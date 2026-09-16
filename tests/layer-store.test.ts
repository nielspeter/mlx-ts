// layerStore: decoder layers read from disk one at a time.
//
// Two kinds of check. The store's own contract — which tensor belongs where,
// what survives a layer's release, what fails and how — on small files written
// here. And the one that matters: a Qwen3 reading its layers through a store
// gives the same logits as the same Qwen3 holding every layer, bit for bit, on a
// tiny 4-bit checkpoint built here too, so CI needs no download.
//   bun test tests/layer-store.test.ts
import { expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  evalAll,
  fromF32,
  fromI32,
  fromU32,
  type KV,
  layerStore,
  loadSafetensors,
  type MX,
  Qwen3,
  saveSafetensors,
  streamTokens,
} from "../src/index.ts";

const dir = mkdtempSync(join(tmpdir(), "mlx-ts-layer-store-"));
const f32 = (xs: number[], shape = [xs.length]) => fromF32(Float32Array.from(xs), shape);
const vals = (x: MX) => [...x.toF32()];

function smallCheckpoint(name: string): string {
  const path = join(dir, name);
  saveSafetensors(path, {
    "model.embed_tokens.weight": f32([1, 2, 3, 4], [2, 2]),
    "model.layers.0.w": f32([10, 11]),
    "model.layers.1.w": f32([20, 21]),
    "model.norm.weight": f32([5, 6]),
  });
  return path;
}

// ---- the store's contract -------------------------------------------------

test("counts the layers and serves shared tensors", () => {
  const store = layerStore(smallCheckpoint("basic.safetensors"));
  expect(store.numLayers).toBe(2);
  expect(vals(store.shared.mx("model.norm.weight"))).toEqual([5, 6]);
  store.done();
});

test("what a layer computes is still readable after the layer is released", () => {
  const store = layerStore(smallCheckpoint("release.safetensors"));
  const [sum] = store.withLayer(1, (L) => [L.mx("model.layers.1.w").add(L.mx("model.layers.1.w"))]);
  // The layer's map is gone by now. The sum was evaluated before it went, which
  // is the whole contract: release first and the result would have nothing to
  // read from.
  expect(vals(sum)).toEqual([40, 42]);
  store.done();
});

test("a weight returned as-is is left alive for the caller", () => {
  const store = layerStore(smallCheckpoint("keep.safetensors"));
  const [w] = store.withLayer(0, (L) => [L.mx("model.layers.0.w")]);
  expect(vals(w)).toEqual([10, 11]);
  store.done();
});

test("reading a tensor from the wrong place says where it lives", () => {
  const store = layerStore(smallCheckpoint("misplaced.safetensors"));
  expect(() => store.withLayer(0, (L) => [L.mx("model.layers.1.w")])).toThrow("belongs to layer 1");
  expect(() => store.withLayer(0, (L) => [L.mx("model.norm.weight")])).toThrow("shared tensor");
  expect(() => store.shared.mx("model.layers.0.w")).toThrow("withLayer(0");
  expect(() => store.shared.mx("model.nope")).toThrow("no such tensor");
  expect(() => store.withLayer(2, () => [])).toThrow("out of range");
  store.done();
});

test("a layer whose computation throws is still released, and the store keeps working", () => {
  const store = layerStore(smallCheckpoint("throws.safetensors"));
  expect(() =>
    store.withLayer(0, (L) => {
      L.mx("model.layers.0.w");
      throw new Error("boom");
    }),
  ).toThrow("boom");
  const [w] = store.withLayer(0, (L) => [L.mx("model.layers.0.w")]);
  expect(vals(w)).toEqual([10, 11]);
  store.done();
});

function twoShards(prefix: string, placeLayer1In: "a" | "b"): string {
  const a = `${prefix}-a.safetensors`, b = `${prefix}-b.safetensors`;
  saveSafetensors(join(dir, a), { "model.embed_tokens.weight": f32([1, 2]), "model.layers.0.w": f32([10, 11]) });
  saveSafetensors(join(dir, b), { "model.layers.1.w": f32([20, 21]), "model.norm.weight": f32([5, 6]) });
  const index = join(dir, `${prefix}.index.json`);
  writeFileSync(index, JSON.stringify({
    weight_map: {
      "model.embed_tokens.weight": a,
      "model.layers.0.w": a,
      "model.layers.1.w": placeLayer1In === "a" ? a : b,
      "model.norm.weight": b,
    },
  }));
  return index;
}

test("a sharded checkpoint is read through its index, each layer from the shard holding it", () => {
  const store = layerStore(twoShards("sharded", "b"));
  expect(store.numLayers).toBe(2);
  expect(vals(store.shared.mx("model.embed_tokens.weight"))).toEqual([1, 2]);
  expect(vals(store.shared.mx("model.norm.weight"))).toEqual([5, 6]);
  expect(vals(store.withLayer(0, (L) => [L.mx("model.layers.0.w")])[0])).toEqual([10, 11]);
  expect(vals(store.withLayer(1, (L) => [L.mx("model.layers.1.w")])[0])).toEqual([20, 21]);
  store.done();
});

test("an index that places a tensor in a shard lacking it fails when the store opens", () => {
  expect(() => layerStore(twoShards("stale", "a"))).toThrow("does not contain it");
});

test("a gap in the layer numbering is refused", () => {
  const path = join(dir, "gap.safetensors");
  saveSafetensors(path, { "model.layers.0.w": f32([1]), "model.layers.2.w": f32([2]) });
  expect(() => layerStore(path)).toThrow("layer 1 is missing");
});

test("a checkpoint with no decoder layers is refused", () => {
  const path = join(dir, "flat.safetensors");
  saveSafetensors(path, { "encoder.w": f32([1]) });
  expect(() => layerStore(path)).toThrow("no decoder layers");
});

test("a file that is not safetensors is refused by name, before any allocation", () => {
  const path = join(dir, "not-a-checkpoint.safetensors");
  writeFileSync(path, "hello, this is not a checkpoint");
  expect(() => layerStore(path)).toThrow("not a safetensors file");
});

// ---- a real model through the store ----------------------------------------

// eos is outside the vocabulary, so generation always runs to `max` — random
// weights would otherwise be free to end a comparison at token zero.
const CFG = {
  hidden_size: 32, num_hidden_layers: 3, num_attention_heads: 2, num_key_value_heads: 1, head_dim: 16,
  rms_norm_eps: 1e-6, rope_theta: 10000, vocab_size: 64, eos_token_id: 64,
  quantization: { group_size: 32, bits: 4 }, tie_word_embeddings: true,
};

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0);
}

/**
 * A 4-bit Qwen3 with the real key layout and shapes, at toy size. Any bit
 * pattern is a valid 4-bit packing, so the weights can be arbitrary; scales and
 * biases are kept small so activations stay finite through three layers.
 */
function tinyQwen3(name: string, head: "tied" | "copy-of-embedding" | "own" = "tied"): string {
  const { hidden_size: D, num_attention_heads: nH, num_key_value_heads: nKV, head_dim: Dh, vocab_size: V } = CFG;
  const I = 64, gs = CFG.quantization.group_size;
  let seed = 1;
  const rec: Record<string, MX> = {};
  const uniform = (n: number, lo: number, hi: number) => {
    const r = lcg(seed++);
    return Float32Array.from({ length: n }, () => lo + (r() / 2 ** 32) * (hi - lo));
  };
  const quant = (p: string, out: number, inp: number) => {
    const r = lcg(seed++);
    rec[`${p}.weight`] = fromU32(Uint32Array.from({ length: out * (inp / 8) }, r), [out, inp / 8]);
    rec[`${p}.scales`] = fromF32(uniform(out * (inp / gs), 0.01, 0.05), [out, inp / gs]);
    rec[`${p}.biases`] = fromF32(uniform(out * (inp / gs), -0.2, 0), [out, inp / gs]);
  };
  const norm = (n: string, size: number) => { rec[n] = fromF32(uniform(size, 0.8, 1.2), [size]); };

  quant("model.embed_tokens", V, D);
  norm("model.norm.weight", D);
  for (let i = 0; i < CFG.num_hidden_layers; i++) {
    const p = `model.layers.${i}`;
    norm(`${p}.input_layernorm.weight`, D);
    norm(`${p}.post_attention_layernorm.weight`, D);
    norm(`${p}.self_attn.q_norm.weight`, Dh);
    norm(`${p}.self_attn.k_norm.weight`, Dh);
    quant(`${p}.self_attn.q_proj`, nH * Dh, D);
    quant(`${p}.self_attn.k_proj`, nKV * Dh, D);
    quant(`${p}.self_attn.v_proj`, nKV * Dh, D);
    quant(`${p}.self_attn.o_proj`, D, nH * Dh);
    quant(`${p}.mlp.gate_proj`, I, D);
    quant(`${p}.mlp.up_proj`, I, D);
    quant(`${p}.mlp.down_proj`, D, I);
  }
  // Added after every layer, so the layers are identical across head variants.
  if (head === "copy-of-embedding") {
    for (const part of ["weight", "scales", "biases"]) rec[`lm_head.${part}`] = rec[`model.embed_tokens.${part}`];
  } else if (head === "own") {
    quant("lm_head", V, D);
  }
  const path = join(dir, name);
  saveSafetensors(path, rec);
  return path;
}

/** Logits after each step, sharing one KV cache — a prefill, then decode steps. */
function logitsSequence(model: Qwen3, steps: number[][]): number[][] {
  const cache: KV[] = Array(model.numLayers).fill(null);
  const out: number[][] = [];
  let offset = 0;
  for (const ids of steps) {
    const logits = model.logitsLastMX(fromI32(Int32Array.from(ids), [1, ids.length]), 1, ids.length, offset, cache, 0);
    evalAll(logits, ...cache.flatMap((c) => (c ? [c.k, c.v] : [])));
    out.push([...logits.toF32()]);
    offset += ids.length;
  }
  return out;
}

const STEPS = [[1, 2, 3, 4, 5], [6], [7]];

for (const [head, cfg] of [
  ["tied", CFG],
  ["own", { ...CFG, tie_word_embeddings: false }],
] as const) {
  test(`a streamed Qwen3 gives the same logits as a resident one, bit for bit (${head} head)`, () => {
    const path = tinyQwen3(`bitexact-${head}.safetensors`, head);
    const resident = logitsSequence(new Qwen3(cfg, loadSafetensors(path)), STEPS);
    const model = new Qwen3(cfg, layerStore(path));
    const streamed = logitsSequence(model, STEPS);
    model.close();
    // Guards against the comparison passing on two runs that both produced
    // nothing meaningful.
    expect(resident.flat().every(Number.isFinite)).toBe(true);
    expect(new Set(resident[0]).size).toBeGreaterThan(1);
    expect(streamed).toEqual(resident);
  });
}

test("generation through streamTokens is identical streamed and resident", async () => {
  const path = tinyQwen3("generate.safetensors");
  const run = async (model: Qwen3) => {
    const ids: number[] = [];
    for await (const { token } of streamTokens(model, [1, 2, 3], { max: 12 })) ids.push(token);
    return ids;
  };
  const resident = await run(new Qwen3(CFG, loadSafetensors(path)));
  const model = new Qwen3(CFG, layerStore(path));
  const streamed = await run(model);
  model.close();
  expect(resident).toHaveLength(12);
  expect(streamed).toEqual(resident);
});

test("an untied checkpoint projects through lm_head, not the embedding", () => {
  const untied = { ...CFG, tie_word_embeddings: false };
  const [tied] = logitsSequence(new Qwen3(CFG, loadSafetensors(tinyQwen3("head-tied.safetensors"))), [[1, 2, 3]]);
  const [copy] = logitsSequence(
    new Qwen3(untied, loadSafetensors(tinyQwen3("head-copy.safetensors", "copy-of-embedding"))), [[1, 2, 3]]);
  const [own] = logitsSequence(new Qwen3(untied, loadSafetensors(tinyQwen3("head-own.safetensors", "own"))), [[1, 2, 3]]);
  // A head identical to the embedding must reproduce the tied logits, and a
  // different head must change them: together, lm_head is read, and wired the
  // same way the tied projection is. Before this, an untied config was ignored
  // and both would have matched the tied logits.
  expect(copy).toEqual(tied);
  expect(own).not.toEqual(tied);
});

test("a store with a different layer count than the config is refused", () => {
  const store = layerStore(tinyQwen3("count.safetensors"));
  expect(() => new Qwen3({ ...CFG, num_hidden_layers: 4 }, store)).toThrow("config has 4 layers");
  store.done();
});
