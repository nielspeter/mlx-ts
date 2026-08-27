// BiCodec's decode path on synthetic weights.
//
// validation/bicodec-decode.ts checks the numbers against mlx-audio with the
// real checkpoint; this runs the same code at toy size to pin the structure —
// the exact 320x upsampling, the mixed-radix FSQ arithmetic, and the two
// weight-norm layouts, which differ between a convolution and its transpose and
// are the easiest thing in the file to get backwards.
//   bun test tests/bicodec.test.ts
import { expect, test } from "bun:test";
import {
  BiCodecPrenet, BiCodecQuantizer, fromI32, SpeakerDetokenizer, WaveGenerator,
  weightNormConv, weightNormConvTranspose,
} from "../src/index.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

test("weight-norm unfusing gives a unit-norm kernel times the gain", () => {
  // g is [C_out, 1, 1] and the norm runs over the other two axes, so every
  // output filter comes out with exactly the norm g asks for.
  const W = fakeWeights({ "c.weight_v": [3, 4, 5], "c.weight_g": [3, 1, 1] });
  const w = weightNormConv(W, "c");
  expect(w.shape).toEqual([3, 5, 4]);            // [C_out, K, C_in] for MLX conv1d

  const f = w.toF32(), g = W.mx("c.weight_g").toF32();
  for (let o = 0; o < 3; o++) {
    let sq = 0;
    for (let i = o * 20; i < (o + 1) * 20; i++) sq += f[i] * f[i];
    expect(Math.sqrt(sq)).toBeCloseTo(Math.abs(g[o]), 4);
  }
  W.done();
});

test("the transposed conv normalises over the *input* channel instead", () => {
  // Stored [C_in, C_out, K] with g per input channel — the opposite axis from
  // weightNormConv. Reusing that one here loads without complaint and is wrong.
  const W = fakeWeights({ "t.weight_v": [3, 4, 5], "t.weight_g": [3, 1, 1] });
  const w = weightNormConvTranspose(W, "t");
  expect(w.shape).toEqual([4, 5, 3]);            // [C_out, K, C_in]

  const f = w.toF32(), g = W.mx("t.weight_g").toF32();
  for (let ci = 0; ci < 3; ci++) {
    let sq = 0;
    for (let o = 0; o < 4; o++) for (let k = 0; k < 5; k++) {
      const v = f[(o * 5 + k) * 3 + ci];
      sq += v * v;
    }
    expect(Math.sqrt(sq)).toBeCloseTo(Math.abs(g[ci]), 4);
  }
  W.done();
});

test("the quantizer widens 8-dim codes to the feature size", () => {
  const W = fakeWeights({
    "quantizer.codebook.weight": [64, 8],
    "quantizer.out_project.weight_v": [16, 8, 1],
    "quantizer.out_project.weight_g": [16, 1, 1],
    "quantizer.out_project.bias": [16],
  });
  const out = new BiCodecQuantizer(W).detokenize(fromI32(Int32Array.from([3, 1, 4, 1]), [1, 4]));
  expect(out.shape).toEqual([1, 4, 16]);
  // Equal ids must give equal rows — it is a lookup, not a position-dependent one.
  const f = out.toF32();
  for (let c = 0; c < 16; c++) expect(f[1 * 16 + c]).toBeCloseTo(f[3 * 16 + c], 5);
  W.done();
});

test("the speaker FSQ decodes ids as mixed-radix digits", () => {
  // levels [4,4,4] -> basis [1,4,16]; each digit maps to (level - 2) / 2.
  const N = 3, D = 3, OUT = 6;
  const W = fakeWeights({
    "speaker_encoder.quantizer.project_out.weight": [5, D],
    "speaker_encoder.quantizer.project_out.bias": [5],
    "speaker_encoder.project.weight": [OUT, N * 5],
    "speaker_encoder.project.bias": [OUT],
  });
  const d = new SpeakerDetokenizer(W, [4, 4, 4]);
  const a = d.detokenize([[0, 1, 2]]);
  expect(a.shape).toEqual([1, OUT]);
  // 21 = 1 + 4 + 16, i.e. digits (1,1,1) — distinct from digits (0,1,2) = 33.
  const same = d.detokenize([[21, 21, 21]]).toF32();
  const diff = d.detokenize([[0, 1, 2]]).toF32();
  expect(same.some((v, i) => Math.abs(v - diff[i]) > 1e-6)).toBe(true);
  W.done();
});

test("the wave generator upsamples by exactly 320", () => {
  const rates = [8, 5, 4, 2], kernels = [16, 11, 8, 4];
  const C_IN = 6, CH = 16;
  const s: Record<string, number[]> = {
    "decoder.model.0.weight_v": [CH, C_IN, 7], "decoder.model.0.weight_g": [CH, 1, 1],
    "decoder.model.0.bias": [CH],
    "decoder.model.5.alpha": [1, CH >> 4, 1],
    "decoder.model.6.weight_v": [1, CH >> 4, 7], "decoder.model.6.weight_g": [1, 1, 1],
    "decoder.model.6.bias": [1],
  };
  for (let i = 0; i < 4; i++) {
    const cin = CH >> i, cout = CH >> (i + 1), p = `decoder.model.${i + 1}`;
    s[`${p}.block.0.alpha`] = [1, cin, 1];
    s[`${p}.block.1.weight_v`] = [cin, cout, kernels[i]];
    s[`${p}.block.1.weight_g`] = [cin, 1, 1];
    s[`${p}.block.1.bias`] = [cout];
    for (const u of [2, 3, 4]) {
      const q = `${p}.block.${u}`;
      s[`${q}.block.0.alpha`] = [1, cout, 1];
      s[`${q}.block.1.weight_v`] = [cout, cout, 7]; s[`${q}.block.1.weight_g`] = [cout, 1, 1];
      s[`${q}.block.1.bias`] = [cout];
      s[`${q}.block.2.alpha`] = [1, cout, 1];
      s[`${q}.block.3.weight_v`] = [cout, cout, 1]; s[`${q}.block.3.weight_g`] = [cout, 1, 1];
      s[`${q}.block.3.bias`] = [cout];
    }
  }
  const W = fakeWeights(s);
  const T = 4;
  const x = fromI32(Int32Array.from({ length: T * C_IN }, (_, i) => i % 3), [1, T, C_IN])
    .astype(10);                                   // FLOAT32
  const wav = new WaveGenerator(W).forward(x);
  // 8*5*4*2 = 320 samples per frame, and no off-by-one: mlx-audio emits one
  // extra sample per stage because it passes `groups` into `output_padding`.
  expect(wav.shape).toEqual([1, T * 320, 1]);
  // Tanh at the end, so nothing can leave [-1, 1].
  for (const v of wav.toF32()) expect(Math.abs(v)).toBeLessThanOrEqual(1);
  W.done();
});

test("the prenet conditions on the speaker and keeps the frame count", () => {
  const D = 8, H = 4, FF = 6, COND = 8, LAYERS = 2, SHALLOW = 2;
  const s: Record<string, number[]> = {
    "prenet.linear_pre.weight": [H, D], "prenet.linear_pre.bias": [H],
    "prenet.linear.weight": [D, H], "prenet.linear.bias": [D],
  };
  const convnext = (p: string, ada: boolean) => {
    s[`${p}.dwconv.weight`] = [H, 1, 7]; s[`${p}.dwconv.bias`] = [H];
    if (ada) {
      s[`${p}.norm.scale.weight`] = [H, COND]; s[`${p}.norm.scale.bias`] = [H];
      s[`${p}.norm.shift.weight`] = [H, COND]; s[`${p}.norm.shift.bias`] = [H];
    } else {
      s[`${p}.norm.weight`] = [H]; s[`${p}.norm.bias`] = [H];
    }
    s[`${p}.pwconv1.weight`] = [FF, H]; s[`${p}.pwconv1.bias`] = [FF];
    s[`${p}.pwconv2.weight`] = [H, FF]; s[`${p}.pwconv2.bias`] = [H];
    s[`${p}.gamma`] = [H];
  };
  const backbone = (p: string, n: number, ada: boolean) => {
    s[`${p}.embed.weight`] = [H, H, 7]; s[`${p}.embed.bias`] = [H];
    if (ada) {
      s[`${p}.norm.scale.weight`] = [H, COND]; s[`${p}.norm.scale.bias`] = [H];
      s[`${p}.norm.shift.weight`] = [H, COND]; s[`${p}.norm.shift.bias`] = [H];
    } else {
      s[`${p}.norm.weight`] = [H]; s[`${p}.norm.bias`] = [H];
    }
    for (let i = 0; i < n; i++) convnext(`${p}.convnext.${i}`, ada);
    s[`${p}.final_layer_norm.weight`] = [H]; s[`${p}.final_layer_norm.bias`] = [H];
  };
  for (let i = 0; i < 2; i++) backbone(`prenet.downsample.${i}.1`, SHALLOW, false);
  backbone("prenet.vocos_backbone", LAYERS, true);

  const W = fakeWeights(s);
  const T = 5;
  const zq = fromI32(Int32Array.from({ length: T * D }, (_, i) => i % 5), [1, T, D]).astype(10);
  const cond = fromI32(Int32Array.from({ length: COND }, (_, i) => i % 3), [1, COND]).astype(10);
  const other = fromI32(Int32Array.from({ length: COND }, (_, i) => (i * 2) % 3), [1, COND]).astype(10);

  const out = new BiCodecPrenet(W, LAYERS).forward(zq, cond);
  expect(out.shape).toEqual([1, T, D]);   // sample_ratios [1,1] keeps the frames

  // The speaker really is a conditioning input, not a decoration: change it and
  // the features change. If norm.scale were read as a plain LayerNorm weight
  // this would come out identical.
  const swapped = new BiCodecPrenet(W, LAYERS).forward(zq, other).toF32();
  const base = out.toF32();
  expect(base.some((v, i) => Math.abs(v - swapped[i]) > 1e-6)).toBe(true);
  W.done();
});
