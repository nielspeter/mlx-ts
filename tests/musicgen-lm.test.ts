// MusicGen's language model on synthetic weights.
//
// validation/musicgen-lm.ts compares its logits against Hugging Face, which
// needs the checkpoint downloaded. This runs the same step() on a tiny config
// to pin the structure around it: the shape of the logits, and the KV cache
// contract — which is where the 55 GB leak lived, and where Owned now holds
// the ownership transfer.
//   bun test tests/musicgen-lm.test.ts
import { expect, test } from "bun:test";
import { fromF32, fromU32, type LayerKV, MusicGenLM, type MusicGenConfig, Owned } from "../src/index.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

const D = 8, H = 2, NL = 2, FF = 16, K = 2, V = 16, MAXPOS = 8, COND = 6;
const cfg: MusicGenConfig = {
  hidden_size: D, num_hidden_layers: NL, num_attention_heads: H, ffn_dim: FF,
  num_codebooks: K, bos_token_id: V, vocab_size: V, max_position_embeddings: MAXPOS,
};

function spec(): Record<string, number[]> {
  const dec = "decoder.model.decoder";
  const s: Record<string, number[]> = {
    [`${dec}.embed_positions.weights`]: [MAXPOS, D],
    [`${dec}.layer_norm.weight`]: [D], [`${dec}.layer_norm.bias`]: [D],
  };
  for (let k = 0; k < K; k++) {
    s[`${dec}.embed_tokens.${k}.weight`] = [V + 1, D];   // +1 for BOS
    s[`decoder.lm_heads.${k}.weight`] = [V, D];
  }
  for (let l = 0; l < NL; l++) {
    const p = `${dec}.layers.${l}`;
    for (const n of ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"]) {
      s[`${p}.${n}.weight`] = [D]; s[`${p}.${n}.bias`] = [D];
    }
    // Projections carry no bias in this checkpoint; the model detects that by
    // the lookup throwing, so leaving them out is the realistic shape.
    for (const att of ["self_attn", "encoder_attn"]) {
      for (const proj of ["q_proj", "k_proj", "v_proj", "out_proj"]) {
        s[`${p}.${att}.${proj}.weight`] = [D, D];
      }
    }
    s[`${p}.fc1.weight`] = [FF, D];
    s[`${p}.fc2.weight`] = [D, FF];
  }
  return s;
}

const lm = () => new MusicGenLM(cfg, fakeWeights(spec()));
const tokens = (ids: number[]) => fromU32(Uint32Array.from(ids), [1, 1, K]);
const cond = () => fromF32(new Float32Array(1 * COND * D).fill(0.1), [1, COND, D]);

test("a step yields logits per codebook", () => {
  using cache = new Owned<LayerKV>(NL);
  const out = lm().step(tokens([1, 2]), cond(), cache, 0);
  expect(out.shape).toEqual([1, 1, V, K]);
});

test("the same step twice gives the same logits", () => {
  const model = lm(), c = cond();
  using c1 = new Owned<LayerKV>(NL);
  using c2 = new Owned<LayerKV>(NL);
  const a = model.step(tokens([1, 2]), c, c1, 0).toF32();
  const b = model.step(tokens([1, 2]), c, c2, 0).toF32();
  expect([...a]).toEqual([...b]);
});

test("the KV cache grows by one token per step", () => {
  const model = lm(), c = cond();
  using cache = new Owned<LayerKV>(NL);
  model.step(tokens([1, 2]), c, cache, 0);
  expect(cache.get(0)!.k.shape[2]).toBe(1);
  model.step(tokens([3, 4]), c, cache, 1);
  expect(cache.get(0)!.k.shape[2]).toBe(2);
  model.step(tokens([5, 6]), c, cache, 2);
  expect(cache.get(0)!.k.shape[2]).toBe(3);
});

test("every layer keeps its own cache entry", () => {
  using cache = new Owned<LayerKV>(NL);
  lm().step(tokens([1, 2]), cond(), cache, 0);
  for (let l = 0; l < NL; l++) expect(cache.get(l)).not.toBeNull();
});

test("history changes the logits, so the cache is really read", () => {
  const model = lm(), c = cond();
  using fresh = new Owned<LayerKV>(NL);
  using warm = new Owned<LayerKV>(NL);
  const atZero = model.step(tokens([1, 2]), c, fresh, 0).toF32();
  model.step(tokens([5, 6]), c, warm, 0);
  const afterHistory = model.step(tokens([1, 2]), c, warm, 1).toF32();
  expect([...atZero]).not.toEqual([...afterHistory]);
});
