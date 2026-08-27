// CLIP's vision tower on synthetic weights.
//
// The transformer is shared with the text encoder, so what needs pinning here
// is the front and the back: that a stride-P convolution really does cut the
// image into patches, that the class token leads the sequence and is what gets
// pooled, and — most importantly — that attention is NOT causal. The two towers
// share layer code and differ by that one flag, so it is the thing most likely
// to be wrong and least likely to look wrong.
//   bun test tests/clip-vision.test.ts
import { expect, test } from "bun:test";
import { ClipVisionEncoder, type ClipVisionConfig, fromF32 } from "../src/index.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

const D = 8, H = 2, NL = 2, IMG = 8, P = 4;   // 8/4 = 2 -> 4 patches
const PATCHES = (IMG / P) * (IMG / P);
const cfg: ClipVisionConfig = {
  hidden_size: D, num_hidden_layers: NL, num_attention_heads: H,
  image_size: IMG, patch_size: P, layer_norm_eps: 1e-5, hidden_act: "quick_gelu",
};

function spec(): Record<string, number[]> {
  const s: Record<string, number[]> = {
    "vision_model.embeddings.class_embedding": [D],
    "vision_model.embeddings.patch_embedding.weight": [D, 3, P, P],
    "vision_model.embeddings.position_embedding.weight": [PATCHES + 1, D],
    "vision_model.pre_layrnorm.weight": [D], "vision_model.pre_layrnorm.bias": [D],
    "vision_model.post_layernorm.weight": [D], "vision_model.post_layernorm.bias": [D],
  };
  for (let l = 0; l < NL; l++) {
    const p = `vision_model.encoder.layers.${l}`;
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

/** An image whose top-left patch can be altered independently of the rest. */
function image(topLeft: number, rest: number): MXLike {
  const px = new Float32Array(IMG * IMG * 3).fill(rest);
  for (let y = 0; y < P; y++) {
    for (let x = 0; x < P; x++) {
      for (let c = 0; c < 3; c++) px[(y * IMG + x) * 3 + c] = topLeft;
    }
  }
  return fromF32(px, [1, IMG, IMG, 3]);
}
type MXLike = ReturnType<typeof fromF32>;

const enc = () => new ClipVisionEncoder(cfg, fakeWeights(spec()));

test("the patch count follows image size over patch size", () => {
  expect(enc().patches).toBe(PATCHES);
});

test("encoding pools to one vector per image", () => {
  expect(enc().encode(image(0.5, 0.1)).shape).toEqual([1, D]);
});

test("encoding is deterministic", () => {
  expect([...enc().encode(image(0.5, 0.1)).toF32()])
    .toEqual([...enc().encode(image(0.5, 0.1)).toF32()]);
});

test("attention is not causal: a late patch still changes the pooled vector", () => {
  // The class token sits at position 0. Under a causal mask it could not see
  // any patch, so every image would pool identically — which is exactly the
  // failure this catches, and it produces no error, just a useless encoder.
  const a = enc().encode(image(0.5, 0.1)).toF32();
  const b = enc().encode(image(0.5, 0.9)).toF32();   // only later patches differ
  expect([...a]).not.toEqual([...b]);
});

test("changing the first patch changes the result too", () => {
  const a = enc().encode(image(0.1, 0.1)).toF32();
  const b = enc().encode(image(0.9, 0.1)).toF32();
  expect([...a]).not.toEqual([...b]);
});

test("embed projects into the text space", () => {
  const proj = fromF32(new Float32Array(5 * D).fill(0.1), [5, D]);   // [proj_dim, D]
  expect(enc().embed(image(0.5, 0.1), proj).shape).toEqual([1, 5]);
});

test("a missing weight is reported by name", () => {
  const partial = spec();
  delete partial["vision_model.post_layernorm.weight"];
  const e = new ClipVisionEncoder(cfg, fakeWeights(partial));
  expect(() => e.encode(image(0.5, 0.1))).toThrow(/post_layernorm/);
});
