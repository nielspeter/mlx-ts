// CLIP's text encoder on synthetic weights.
//
// validation/clip-encode.ts already checks the numbers against mlx-examples,
// but that needs 1.6 GB downloaded. A tiny config runs the same forward pass in
// milliseconds, and lets this assert the one thing a numeric comparison cannot
// isolate: that attention is actually causal. Dropping the mask still produces
// plausible conditioning, so nothing downstream would complain.
//   bun test tests/clip-encoder.test.ts
import { expect, test } from "bun:test";
import { ClipTextEncoder, type ClipConfig, fromU32 } from "../src/index.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

const D = 8, H = 2, NL = 2, V = 16, MAXPOS = 8;
const cfg: ClipConfig = {
  hidden_size: D, num_hidden_layers: NL, num_attention_heads: H,
  max_position_embeddings: MAXPOS, vocab_size: V, layer_norm_eps: 1e-5,
  hidden_act: "quick_gelu",
};

function spec(): Record<string, number[]> {
  const s: Record<string, number[]> = {
    "text_model.embeddings.token_embedding.weight": [V, D],
    "text_model.embeddings.position_embedding.weight": [MAXPOS, D],
    "text_model.final_layer_norm.weight": [D],
    "text_model.final_layer_norm.bias": [D],
  };
  for (let l = 0; l < NL; l++) {
    const p = `text_model.encoder.layers.${l}`;
    for (const n of ["layer_norm1", "layer_norm2"]) {
      s[`${p}.${n}.weight`] = [D]; s[`${p}.${n}.bias`] = [D];
    }
    for (const n of ["q_proj", "k_proj", "v_proj", "out_proj"]) {
      s[`${p}.self_attn.${n}.weight`] = [D, D]; s[`${p}.self_attn.${n}.bias`] = [D];
    }
    s[`${p}.mlp.fc1.weight`] = [4 * D, D]; s[`${p}.mlp.fc1.bias`] = [4 * D];
    s[`${p}.mlp.fc2.weight`] = [D, 4 * D]; s[`${p}.mlp.fc2.bias`] = [D];
  }
  return s;
}

const encode = (ids: number[]) => {
  const clip = new ClipTextEncoder(cfg, fakeWeights(spec()));
  return clip.encode(fromU32(Uint32Array.from(ids), [1, ids.length])).toF32();
};

test("encoding returns one vector per token", () => {
  const clip = new ClipTextEncoder(cfg, fakeWeights(spec()));
  const h = clip.encode(fromU32(Uint32Array.from([1, 2, 3, 4]), [1, 4]));
  expect(h.shape).toEqual([1, 4, D]);
});

test("the same ids encode to the same states", () => {
  expect([...encode([1, 2, 3, 4])]).toEqual([...encode([1, 2, 3, 4])]);
});

test("different ids encode differently", () => {
  expect([...encode([1, 2, 3, 4])]).not.toEqual([...encode([1, 2, 3, 5])]);
});

test("attention is causal: a later token cannot change an earlier state", () => {
  // This is the whole point of the file. CLIP is trained with a causal mask,
  // and without one the encoder still returns sensible-looking numbers.
  const a = encode([1, 2, 3, 4]);
  const b = encode([1, 2, 3, 9]);            // only the last token differs
  const upto = 3 * D;                        // states for tokens 0..2
  expect([...a.slice(0, upto)]).toEqual([...b.slice(0, upto)]);
  // ...and the position that did change must actually differ.
  expect([...a.slice(upto)]).not.toEqual([...b.slice(upto)]);
});

test("changing an early token changes everything after it", () => {
  const a = encode([1, 2, 3, 4]);
  const b = encode([5, 2, 3, 4]);
  expect([...a]).not.toEqual([...b]);
});

test("a missing weight is reported by name", () => {
  const partial = spec();
  delete partial["text_model.final_layer_norm.weight"];
  const clip = new ClipTextEncoder(cfg, fakeWeights(partial));
  expect(() => clip.encode(fromU32(Uint32Array.from([1]), [1, 1])))
    .toThrow(/final_layer_norm/);
});
