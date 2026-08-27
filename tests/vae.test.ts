// The VAE decoder on synthetic weights.
//
// validation/vae-decode.ts checks the numbers against mlx-examples with the
// real checkpoint; this runs the same code at toy size to pin the structure —
// that latents come out 8x larger in each spatial dimension, that the residual
// shortcut appears exactly where channel counts change, and that upsampling
// really is nearest-neighbour.
//   bun test tests/vae.test.ts
import { expect, test } from "bun:test";
import { fromF32, upsampleNearest, VaeDecoder, type VaeConfig } from "../src/index.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

const CH = [4, 8], L = 1, G = 2, LATENT = 4, OUT = 3;
const cfg: VaeConfig = {
  block_out_channels: CH, layers_per_block: L, norm_num_groups: G,
  latent_channels: LATENT, out_channels: OUT, scaling_factor: 0.18215,
};

/** Mirrors what src/models/vae.ts asks for, at toy widths. */
function spec(): Record<string, number[]> {
  const s: Record<string, number[]> = {
    "post_quant_conv.weight": [LATENT, LATENT, 1, 1], "post_quant_conv.bias": [LATENT],
    "decoder.conv_in.weight": [8, LATENT, 3, 3], "decoder.conv_in.bias": [8],
    "decoder.conv_norm_out.weight": [4], "decoder.conv_norm_out.bias": [4],
    "decoder.conv_out.weight": [OUT, 4, 3, 3], "decoder.conv_out.bias": [OUT],
  };
  const resnet = (p: string, cin: number, cout: number) => {
    s[`${p}.norm1.weight`] = [cin]; s[`${p}.norm1.bias`] = [cin];
    s[`${p}.norm2.weight`] = [cout]; s[`${p}.norm2.bias`] = [cout];
    s[`${p}.conv1.weight`] = [cout, cin, 3, 3]; s[`${p}.conv1.bias`] = [cout];
    s[`${p}.conv2.weight`] = [cout, cout, 3, 3]; s[`${p}.conv2.bias`] = [cout];
    // Only present when the width changes — the model detects it by the lookup
    // throwing, so adding it unconditionally would hide that branch.
    if (cin !== cout) {
      s[`${p}.conv_shortcut.weight`] = [cout, cin, 1, 1];
      s[`${p}.conv_shortcut.bias`] = [cout];
    }
  };
  resnet("decoder.mid_block.resnets.0", 8, 8);
  resnet("decoder.mid_block.resnets.1", 8, 8);
  const a = "decoder.mid_block.attentions.0";
  s[`${a}.group_norm.weight`] = [8]; s[`${a}.group_norm.bias`] = [8];
  for (const n of ["query", "key", "value", "proj_attn"]) {
    s[`${a}.${n}.weight`] = [8, 8]; s[`${a}.${n}.bias`] = [8];
  }
  // Up blocks run coarse to fine: 8 -> 8, then 8 -> 4.
  for (let j = 0; j <= L; j++) resnet(`decoder.up_blocks.0.resnets.${j}`, 8, 8);
  s["decoder.up_blocks.0.upsamplers.0.conv.weight"] = [8, 8, 3, 3];
  s["decoder.up_blocks.0.upsamplers.0.conv.bias"] = [8];
  resnet("decoder.up_blocks.1.resnets.0", 8, 4);
  for (let j = 1; j <= L; j++) resnet(`decoder.up_blocks.1.resnets.${j}`, 4, 4);
  return s;
}

const decode = (h: number, w: number) => {
  const vae = new VaeDecoder(cfg, fakeWeights(spec()));
  return vae.decode(fromF32(new Float32Array(1 * h * w * LATENT).fill(0.25), [1, h, w, LATENT]));
};

test("upsampleNearest repeats each pixel, it does not interpolate", () => {
  const x = fromF32(Float32Array.from([1, 2, 3, 4]), [1, 2, 2, 1]);
  expect([...upsampleNearest(x, 2).toF32()]).toEqual([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
});

test("a scale of 1 leaves the image alone", () => {
  const x = fromF32(Float32Array.from([1, 2, 3, 4]), [1, 2, 2, 1]);
  expect([...upsampleNearest(x, 1).toF32()]).toEqual([1, 2, 3, 4]);
});

test("two up blocks means one upsample: latents come out twice as large", () => {
  // The real decoder has four blocks and so multiplies by 8; this toy config
  // has two, and must multiply by exactly 2.
  const img = decode(4, 4);
  expect(img.shape).toEqual([1, 8, 8, OUT]);
});

test("the output is a three-channel image", () => {
  expect(decode(2, 2).shape[3]).toBe(OUT);
});

test("decoding is deterministic", () => {
  expect([...decode(2, 2).toF32()]).toEqual([...decode(2, 2).toF32()]);
});

test("a missing weight is reported by name", () => {
  const partial = spec();
  delete partial["decoder.conv_out.weight"];
  const vae = new VaeDecoder(cfg, fakeWeights(partial));
  expect(() => vae.decode(fromF32(new Float32Array(16).fill(0.25), [1, 2, 2, LATENT])))
    .toThrow(/conv_out/);
});
