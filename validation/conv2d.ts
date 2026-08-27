// TS side of the conv2d / convTranspose2d parity check.
//
// These are the primitives Stable Diffusion's UNet and VAE are built from, so
// they get pinned before anything is built on top: a quiet layout or padding
// mismatch here would show up as a subtly wrong image fifty diffusion steps
// later, which is the worst place to debug it.
//   python3 reference/reference-conv2d.py && bun validation/conv2d.ts
import { fromF32, tidy, type MX } from "../src/index.ts";

const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

const N = 2, H = 9, W = 7, CIN = 4, COUT = 6, KH = 3, KW = 3;

const show = (tag: string, a: MX) => {
  const f = a.toF32();
  let sum = 0;
  for (const v of f) sum += v;
  const first4 = Array.from(f.slice(0, 4)).map((v) => +v.toFixed(6));
  console.log(`${tag} shape=[${a.shape.join(", ")}] sum=${sum.toFixed(6)} first4=[${first4.join(", ")}]`);
};

tidy(() => {
  const x = fromF32(det(N * H * W * CIN, 1), [N, H, W, CIN]);
  const w = fromF32(det(COUT * KH * KW * CIN, 2), [COUT, KH, KW, CIN]);

  show("conv2d_basic  ", x.conv2d(w, [1, 1], [0, 0], [1, 1], 1));
  show("conv2d_stride ", x.conv2d(w, [2, 2], [1, 1], [1, 1], 1));
  show("conv2d_dilate ", x.conv2d(w, [1, 1], [2, 2], [2, 2], 1));

  const wg = fromF32(det(COUT * KH * KW * (CIN / 2), 3), [COUT, KH, KW, CIN / 2]);
  show("conv2d_groups ", x.conv2d(wg, [1, 1], [1, 1], [1, 1], 2));

  const wt = fromF32(det(CIN * KH * KW * COUT, 4), [CIN, KH, KW, COUT]);
  const y = x.conv2d(w, [2, 2], [1, 1]);
  show("convT_stride2 ", y.convTranspose2d(wt, [2, 2], [1, 1], [1, 1], [1, 1]));
});
