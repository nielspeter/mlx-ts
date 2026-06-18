// TS port of benchmarks/python/layer_norm_bench.py — forward only (no grad/compile).
// Hand-written layer_norm vs fused mx.fast.layer_norm, looped 32x, across dtypes/sizes.
//   bun benchmarks/layer-norm.ts

import { MX, scalar } from "../mx.ts";
import { randomUniform, astype, rsqrt, meanAxis, varAxis, layerNorm, FLOAT32, FLOAT16, BFLOAT16, timeFn, loop } from "./time-utils.ts";

const EPS = 1e-5, L = 1024, LOOP = 32;

function manualLayerNorm(x: MX, w: MX | null, b: MX | null): MX {
  const xf = astype(x, FLOAT32);
  const mu = meanAxis(xf, -1, true);
  const v = varAxis(xf, -1, true);
  let y = xf.sub(mu).mul(rsqrt(v.add(scalar(EPS))));
  if (w) y = y.mul(w);
  if (b) y = y.add(b);
  return y;
}

function timeLayerNorm(N: number, dt: number, label: string) {
  const x = astype(randomUniform([8, L, N]), dt);
  const w = astype(randomUniform([N]), dt);
  const b = astype(randomUniform([N]), dt);
  x.eval(); w.eval(); b.eval();
  console.log(`-- ${label} N=${N} --`);
  timeFn(() => loop(x, LOOP, (y) => manualLayerNorm(y, w, b)), "layer_norm (manual, 32x)");
  timeFn(() => loop(x, LOOP, (y) => layerNorm(y, w, b, EPS)), "layer_norm (fast, 32x)");
}

for (const [dt, label] of [[FLOAT32, "float32"], [FLOAT16, "float16"], [BFLOAT16, "bfloat16"]] as [number, string][])
  for (const n of [1024, 4096, 8192]) timeLayerNorm(n, dt, label);
