// TS port of benchmarks/python/time_utils.py — same protocol: 5 warmup iters,
// then 100 timed iters of eval(fn(args)), reported in msec. Each iteration runs
// under tidy() so per-iter intermediates are freed (FinalizationRegistry won't
// fire inside a tight loop — see FINDINGS §6.5).
//
// Op/random helpers wrap the generated FFI functions into MX-returning calls so
// bench files read like the Python originals without bloating mx.ts. mlx-ts
// always runs on the default stream (Metal/GPU on Apple silicon).

import { ptr } from "../src/ffi/index.ts";
import { m, stream } from "../src/ffi/generated.ts";
import * as g from "../src/ffi/generated.ts";
import { MX, tidy, evalAll, activeMemoryMB } from "../mx.ts";

// mlx_dtype enum: float16=9, float32=10, float64=11, bfloat16=12, uint32=3
export const FLOAT32 = 10, FLOAT16 = 9, BFLOAT16 = 12, UINT32 = 3;
const wrap = (h: number) => new MX(h);

// ---- random / dtype ----
export function randomUniform(shape: number[]): MX {
  const lo = m.mlx_array_new_float(0) as number, hi = m.mlx_array_new_float(1) as number;
  const r = new BigUint64Array(1); r[0] = BigInt((m.mlx_array_new() as number) ?? 0);
  m.mlx_random_uniform(ptr(r), lo, hi, ptr(new Int32Array(shape)), BigInt(shape.length), FLOAT32, 0, stream);
  m.mlx_array_free(lo); m.mlx_array_free(hi);
  return wrap(Number(r[0]));
}
export function randomNormal(shape: number[]): MX {
  const r = new BigUint64Array(1); r[0] = BigInt((m.mlx_array_new() as number) ?? 0);
  m.mlx_random_normal(ptr(r), ptr(new Int32Array(shape)), BigInt(shape.length), FLOAT32, 0.0, 1.0, 0, stream);
  return wrap(Number(r[0]));
}
export const astype = (a: MX, dtype: number) => wrap(g.astype(a.h, dtype));

// ---- elementwise / reductions (the ops the Python benches use) ----
export const exp = (a: MX) => wrap(g.exp(a.h));
export const negative = (a: MX) => wrap(g.negative(a.h));
export const square = (a: MX) => wrap(g.square(a.h));
export const rsqrt = (a: MX) => wrap(g.rsqrt(a.h));
export const maximum = (a: MX, b: MX) => wrap(g.maximum(a.h, b.h));
export const maxAxis = (a: MX, axis: number, keepdims = false) => wrap(g.maxAxis(a.h, axis, keepdims));
export const minAxis = (a: MX, axis: number, keepdims = false) => wrap(g.minAxis(a.h, axis, keepdims));
export const meanAxis = (a: MX, axis: number, keepdims = false) => wrap(g.meanAxis(a.h, axis, keepdims));
export const varAxis = (a: MX, axis: number, keepdims = false) => wrap(g.varAxis(a.h, axis, keepdims, 0));
export const logsumexpAxis = (a: MX, axis: number, keepdims = false) => wrap(g.logsumexpAxis(a.h, axis, keepdims));
export const broadcastTo = (a: MX, shape: number[]) => wrap(g.broadcastTo(a.h, shape));

// ---- fast layer_norm (rms_norm/rope/sdpa already live on MX) ----
export function layerNorm(x: MX, w: MX | null, b: MX | null, eps: number): MX {
  const r = new BigUint64Array(1); r[0] = BigInt((m.mlx_array_new() as number) ?? 0);
  m.mlx_fast_layer_norm(ptr(r), x.h, w ? w.h : 0, b ? b.h : 0, eps, stream);
  return wrap(Number(r[0]));
}

// ---- quantize: w[out,in] -> {wq, scales, biases} (affine, group_size 64, 4 bit) ----
export function quantize(w: MX, groupSize = 64, bits = 4): { wq: MX; scales: MX; biases: MX } {
  const vh = g.quantize(w.h, groupSize, bits, "affine", null) as number;
  const at = (i: number) => { const o = new BigUint64Array(1); m.mlx_vector_array_get(ptr(o), vh, BigInt(i)); return wrap(Number(o[0])); };
  const out = { wq: at(0), scales: at(1), biases: at(2) };
  m.mlx_vector_array_free(vh);
  return out;
}

// ---- memory guardrail ----
// Benches touch at most a few hundred MB of inputs, so anything near a GB is a
// harness bug (e.g. pinning a deep loop's intermediates). Abort well below RAM
// rather than let the process OOM the machine.
const MEM_CEIL_MB = 6000;
function guardMem(where: string): void {
  const mb = activeMemoryMB();
  if (mb > MEM_CEIL_MB) throw new Error(`[bench guard] active memory ${mb | 0} MB > ${MEM_CEIL_MB} MB at "${where}" — aborting to avoid OOM`);
}

// Feed output -> input n times, freeing each intermediate so peak memory stays at
// ~one iteration. Wrapping a deep loop in a SINGLE tidy() instead pins every
// intermediate at once (handles stay live, so MLX can't stream-free during eval),
// which reached tens of GB and OOM'd — see benchmarks/README.md. Each step runs
// under its own tidy; the prior output is freed once the next is built (safe:
// mlx-c refcounts keep graph inputs alive until eval — FINDINGS §6.6/§6.8).
export function loop(x: MX, n: number, step: (y: MX) => MX): MX {
  let y = x;
  for (let i = 0; i < n; i++) {
    const ny = tidy(() => step(y));
    if (y !== x) y.free();
    y = ny;
    guardMem("loop");
  }
  return y;
}

// ---- timing ----
const evalResult = (r: MX | MX[]) => Array.isArray(r) ? evalAll(...r) : r.eval();

export function timeFn(fn: () => MX | MX[], msg: string, iters = 100, warmup = 5): void {
  guardMem(`${msg} (start)`);
  for (let i = 0; i < warmup; i++) tidy(() => { evalResult(fn()); return []; });
  guardMem(`${msg} (after warmup)`);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) tidy(() => { evalResult(fn()); return []; });
  const msec = (performance.now() - t0) / iters;
  console.log(`Timing ${msg} ... ${msec.toFixed(5)} msec`);
}
