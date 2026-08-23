// TS port of benchmarks/python/rms_norm_bench.py — forward only (no grad/compile).
// Compares a hand-written rms_norm against the fused mx.fast.rms_norm, looped 32x
// like the Python bench's inner loop.
//   bun benchmarks/rms-norm.ts

import { MX, scalar } from "../src/core/mx.ts";
import { randomUniform, astype, square, rsqrt, meanAxis, FLOAT16, timeFn, loop } from "./time-utils.ts";

const EPS = 1e-5;

// manual rms_norm in fp32 (mirrors the Python reference path)
function rmsNorm(x: MX, w: MX | null): MX {
  const xf = astype(x, 10); // float32
  const n = rsqrt(meanAxis(square(xf), -1, true).add(scalar(EPS)));
  let y = xf.mul(n);
  if (w) y = y.mul(w);
  return y;
}

const x = astype(randomUniform([8, 1024, 4096]), FLOAT16);
const w = astype(randomUniform([4096]), FLOAT16);
x.eval(); w.eval();

const LOOP = 32;
timeFn(() => loop(x, LOOP, (y) => rmsNorm(y, w)), "rms_norm (manual, 32x)");
timeFn(() => loop(x, LOOP, (y) => y.rmsNorm(w, EPS)), "rms_norm (fast, 32x)");
