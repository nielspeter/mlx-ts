// The Qwen2 backbone on synthetic weights.
//
// validation/qwen2-generate.ts checks it against mlx-lm on a real checkpoint;
// this pins the structure with nothing downloaded — in particular the two
// things that separate Qwen2 from the Qwen3 next to it in this repo, and which
// produce plausible-but-wrong output rather than an error: q/k/v carry a bias,
// and there is no per-head q/k normalisation.
//   bun test tests/qwen2.test.ts
import { expect, test } from "bun:test";
import { fromI32, type KV } from "../src/index.ts";
import { Qwen2, type Qwen2Config } from "../src/models/qwen2.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

const D = 8, NH = 4, NKV = 2, NL = 2, FF = 16, V = 32, DH = D / NH;
const cfg: Qwen2Config = {
  hidden_size: D, num_hidden_layers: NL, num_attention_heads: NH,
  num_key_value_heads: NKV, rms_norm_eps: 1e-6, rope_theta: 10000,
  vocab_size: V, eos_token_id: V - 1, tie_word_embeddings: true,
};

function spec(extra: Record<string, number[]> = {}): Record<string, number[]> {
  const s: Record<string, number[]> = {
    "model.embed_tokens.weight": [V, D],
    "model.norm.weight": [D],
    ...extra,
  };
  for (let l = 0; l < NL; l++) {
    const p = `model.layers.${l}`;
    s[`${p}.input_layernorm.weight`] = [D];
    s[`${p}.post_attention_layernorm.weight`] = [D];
    s[`${p}.self_attn.q_proj.weight`] = [NH * DH, D];
    s[`${p}.self_attn.q_proj.bias`] = [NH * DH];      // Qwen3 has no such bias
    s[`${p}.self_attn.k_proj.weight`] = [NKV * DH, D];
    s[`${p}.self_attn.k_proj.bias`] = [NKV * DH];
    s[`${p}.self_attn.v_proj.weight`] = [NKV * DH, D];
    s[`${p}.self_attn.v_proj.bias`] = [NKV * DH];
    s[`${p}.self_attn.o_proj.weight`] = [D, NH * DH];
    s[`${p}.mlp.gate_proj.weight`] = [FF, D];
    s[`${p}.mlp.up_proj.weight`] = [FF, D];
    s[`${p}.mlp.down_proj.weight`] = [D, FF];
  }
  return s;
}

const model = (over?: Partial<Qwen2Config>, sp = spec()) =>
  new Qwen2({ ...cfg, ...over }, fakeWeights(sp));

const run = (m: Qwen2, ids: number[], cache: KV[] = new Array(NL).fill(null), offset = 0) =>
  m.logitsLastMX(fromI32(Int32Array.from(ids), [1, ids.length]), 1, ids.length, offset, cache, 0);

test("a forward pass yields one logit per vocabulary entry", () => {
  expect(run(model(), [1, 2, 3]).shape).toEqual([1, V]);
});

test("the same ids give the same logits", () => {
  expect([...run(model(), [1, 2, 3]).toF32()]).toEqual([...run(model(), [1, 2, 3]).toF32()]);
});

test("different ids give different logits", () => {
  expect([...run(model(), [1, 2, 3]).toF32()]).not.toEqual([...run(model(), [1, 2, 4]).toF32()]);
});

test("grouped-query attention: fewer kv heads than query heads", () => {
  // k and v are NKV*DH wide while q is NH*DH; getting the head split wrong is a
  // shape error, which is the point of exercising NKV < NH here.
  expect(NKV).toBeLessThan(NH);
  expect(run(model(), [1, 2]).shape).toEqual([1, V]);
});

test("the KV cache grows by one entry per layer and lengthens per step", () => {
  const m = model();
  const cache: KV[] = new Array(NL).fill(null);
  run(m, [1, 2, 3], cache, 0);
  expect(cache[0]!.k.shape[2]).toBe(3);
  run(m, [4], cache, 3);
  expect(cache[0]!.k.shape[2]).toBe(4);
  for (let l = 0; l < NL; l++) expect(cache[l]).not.toBeNull();
});

test("decoding with the cache matches decoding the whole prefix at once", () => {
  // The cached path and the full-prefix path must agree, or generation drifts
  // from the second token onward while still looking sane.
  const whole = run(model(), [1, 2, 3]).toF32();
  const m = model();
  const cache: KV[] = new Array(NL).fill(null);
  run(m, [1, 2], cache, 0);
  const stepped = run(m, [3], cache, 2).toF32();
  for (let i = 0; i < whole.length; i++) expect(stepped[i]).toBeCloseTo(whole[i], 4);
});

test("a tied checkpoint needs no lm_head", () => {
  expect(run(model({ tie_word_embeddings: true }), [1]).shape).toEqual([1, V]);
});

test("an untied checkpoint uses lm_head", () => {
  const withHead = spec({ "lm_head.weight": [V, D] });
  expect(run(model({ tie_word_embeddings: false }, withHead), [1]).shape).toEqual([1, V]);
});

test("a missing weight is reported by name", () => {
  const partial = spec();
  delete partial["model.layers.1.mlp.down_proj.weight"];
  expect(() => model(undefined, partial)).toThrow(/down_proj/);
});

test("the bias Qwen3 lacks is required here", () => {
  const partial = spec();
  delete partial["model.layers.0.self_attn.q_proj.bias"];
  expect(() => model(undefined, partial)).toThrow(/q_proj\.bias/);
});
