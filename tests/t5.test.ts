// T5's encoder on synthetic weights.
//
// The mirror of tests/clip-encoder.test.ts: T5 is bidirectional where CLIP is
// causal, and the relative position bias lives on block 0 only and is shared by
// the rest. Both are structural, so a numeric comparison against a real
// checkpoint would not tell them apart from a subtly different model.
//   bun test tests/t5.test.ts
import { expect, test } from "bun:test";
import { fromI32, T5Encoder, type T5Config } from "../src/index.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

const D = 8, H = 2, NL = 2, FF = 16, V = 16, BUCKETS = 32;
const cfg: T5Config = {
  d_model: D, d_kv: D / H, d_ff: FF, num_layers: NL, num_heads: H,
  relative_attention_num_buckets: BUCKETS, relative_attention_max_distance: 128,
  layer_norm_epsilon: 1e-6,
};

function spec(): Record<string, number[]> {
  const s: Record<string, number[]> = {
    "text_encoder.shared.weight": [V, D],
    "text_encoder.encoder.final_layer_norm.weight": [D],
    // Only block 0 carries the bias table; every block reads it from there.
    "text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": [BUCKETS, H],
  };
  for (let b = 0; b < NL; b++) {
    const p = `text_encoder.encoder.block.${b}`;
    s[`${p}.layer.0.layer_norm.weight`] = [D];
    s[`${p}.layer.1.layer_norm.weight`] = [D];
    for (const n of ["q", "k", "v", "o"]) s[`${p}.layer.0.SelfAttention.${n}.weight`] = [D, D];
    s[`${p}.layer.1.DenseReluDense.wi.weight`] = [FF, D];
    s[`${p}.layer.1.DenseReluDense.wo.weight`] = [D, FF];
  }
  return s;
}

const encode = (ids: number[]) => {
  const t5 = new T5Encoder(cfg, fakeWeights(spec()));
  return t5.encode(fromI32(Int32Array.from(ids), [1, ids.length])).toF32();
};

test("encoding returns one vector per token", () => {
  const t5 = new T5Encoder(cfg, fakeWeights(spec()));
  expect(t5.encode(fromI32(Int32Array.from([1, 2, 3]), [1, 3])).shape).toEqual([1, 3, D]);
});

test("the same ids encode to the same states", () => {
  expect([...encode([1, 2, 3])]).toEqual([...encode([1, 2, 3])]);
});

test("attention is bidirectional: a later token changes earlier states too", () => {
  // The opposite of CLIP. T5 conditions on the whole sequence, so masking it
  // by accident would leave the first positions untouched — which is exactly
  // what this catches.
  const a = encode([1, 2, 3]);
  const b = encode([1, 2, 9]);
  expect([...a.slice(0, D)]).not.toEqual([...b.slice(0, D)]);
});

test("sequence length changes the relative position bias", () => {
  // Same leading tokens, longer sequence: the bias table is indexed by relative
  // distance, so the shared prefix must not encode identically.
  const short = encode([1, 2]);
  const long = encode([1, 2, 3]);
  expect([...short.slice(0, D)]).not.toEqual([...long.slice(0, D)]);
});

test("a missing block weight is reported by name", () => {
  const partial = spec();
  delete partial["text_encoder.encoder.block.1.layer.1.DenseReluDense.wo.weight"];
  const t5 = new T5Encoder(cfg, fakeWeights(partial));
  expect(() => t5.encode(fromI32(Int32Array.from([1, 2]), [1, 2]))).toThrow(/DenseReluDense/);
});

test("the prefix is configurable, for checkpoints that nest differently", () => {
  const renamed = Object.fromEntries(
    Object.entries(spec()).map(([k, v]) => [k.replace("text_encoder.", "encoder_only."), v]));
  const t5 = new T5Encoder(cfg, fakeWeights(renamed), "encoder_only");
  expect(t5.encode(fromI32(Int32Array.from([1, 2]), [1, 2])).shape).toEqual([1, 2, D]);
});
