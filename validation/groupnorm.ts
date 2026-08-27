// TS side of the GroupNorm parity check.
//
// SD's VAE and UNet normalise with GroupNorm at every block, and its weights
// come from PyTorch — so the grouping has to match mlx.nn.GroupNorm's
// pytorch_compatible path, not the other one. The two differ only in how
// channels are split, which is invisible until an image comes out subtly wrong.
//   python3 reference/reference-groupnorm.py && bun validation/groupnorm.ts
import { fromF32, GroupNorm, tidy, type MX } from "../src/index.ts";

const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

const N = 2, H = 5, W = 4, C = 8, G = 4;

const show = (tag: string, a: MX) => {
  const f = a.toF32();
  let sum = 0; for (const v of f) sum += v;
  console.log(`${tag} shape=[${a.shape.join(", ")}] sum=${sum.toFixed(5)} ` +
              `first4=[${Array.from(f.slice(0, 4)).map((v) => +v.toFixed(5)).join(", ")}]`);
};

tidy(() => {
  const x = fromF32(det(N * H * W * C, 1), [N, H, W, C]);
  const weight = fromF32(det(C, 2), [C]);
  const bias = fromF32(det(C, 3), [C]);
  show("groupnorm", new GroupNorm(G, 1e-5).forward(x, weight, bias));
  show("groupnorm_g1", new GroupNorm(1, 1e-5).forward(x, weight, bias));
});
