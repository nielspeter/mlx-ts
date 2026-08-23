// SPIKE: mlx-c over Deno's built-in FFI. Must reproduce the spike-ffi-bun.ts
// fingerprint exactly ([2.25, 4, 7.5, 6.25], sum 20) and answer the same three
// questions: ABI, handle representation, zero-copy readback.
//   deno run --allow-ffi --allow-read --allow-env spike-ffi-deno.ts
import { LIBMLXC } from "../src/ffi/native-lib.ts";

const FLOAT32 = 10;
const lib = Deno.dlopen(LIBMLXC, {
  mlx_default_gpu_stream_new: { parameters: [], result: "pointer" },
  mlx_array_new:             { parameters: [], result: "pointer" },
  mlx_array_new_data:        { parameters: ["buffer", "buffer", "i32", "i32"], result: "pointer" },
  mlx_matmul:                { parameters: ["buffer", "pointer", "pointer", "pointer"], result: "i32" },
  mlx_array_eval:            { parameters: ["pointer"], result: "i32" },
  mlx_array_data_float32:    { parameters: ["pointer"], result: "pointer" },
  mlx_array_ndim:            { parameters: ["pointer"], result: "u64" },
  mlx_array_free:            { parameters: ["pointer"], result: "i32" },
} as const);
const m = lib.symbols;

const stream = m.mlx_default_gpu_stream_new();
// Deno's "buffer" params take a TypedArray directly — no ptr() equivalent needed.
const arr = (data: Float32Array, shape: number[]) =>
  m.mlx_array_new_data(data, new Int32Array(shape), shape.length, FLOAT32);

// --- 1+2: ABI and handle representation --------------------------------
const A = new Float32Array([1, 2, 3, 4, 5, 6]);
const B = new Float32Array([0.5, -1, 2, 0.25, -0.75, 1.5]);
const a = arr(A, [2, 3]), b = arr(B, [3, 2]);

// Deno hands pointers back as opaque PointerObjects, not numbers — so the
// `mlx_array*` out-slot needs an explicit address round-trip in both
// directions. (bun:ffi needed a `?? 0` here instead, for its NULL -> null.)
const res = new BigUint64Array(1);
res[0] = BigInt(Deno.UnsafePointer.value(m.mlx_array_new()));
m.mlx_matmul(res, a, b, stream);
const c = Deno.UnsafePointer.create(res[0]);
m.mlx_array_eval(c);

const cp = m.mlx_array_data_float32(c);
const out = new Float32Array(new Deno.UnsafePointerView(cp!).getArrayBuffer(4 * 4));
const sum = out.reduce((s, x) => s + x, 0);

// --- 3: is readback a view or a copy? ----------------------------------
const N = 2_000_000;
const big = arr(new Float32Array(N).fill(1.5), [N]);
m.mlx_array_eval(big);
const view = new Deno.UnsafePointerView(m.mlx_array_data_float32(big)!);

const t0 = performance.now();
const v1 = new Float32Array(view.getArrayBuffer(N * 4));
const viewMs = performance.now() - t0;
const v2 = new Float32Array(view.getArrayBuffer(N * 4));
v1[0] = 12345;
const aliases = v2[0] === 12345;

// --- 4: per-call FFI overhead ------------------------------------------
// Measured twice: Deno boxes a u64 return into a BigInt (an allocation per
// call). u32 reads w0 instead of x0 — exact for ndim, and comparable to what
// koffi does by default for size_t.
const lib32 = Deno.dlopen(LIBMLXC, {
  mlx_array_ndim: { parameters: ["pointer"], result: "u32" },
} as const);
const bench = (fn: (h: any) => unknown) => {
  const N = 500_000;
  let sink = 0;
  // The result has to be observed, or the JIT is free to elide the call and
  // "measure" nothing. Truthiness is cheap and doesn't convert a BigInt.
  const run = () => { const t = performance.now(); for (let i = 0; i < N; i++) if (fn(c)) sink++; return performance.now() - t; };
  run(); run();                                       // warm the JIT
  const ms = Math.min(run(), run(), run());           // best of 3
  if (sink === 0) throw new Error("bench: call was elided");
  return (ms * 1e6) / N;
};
const nsPerCall = bench(lib32.symbols.mlx_array_ndim);
const nsBigInt = bench(m.mlx_array_ndim);

console.log(`runtime        : deno ${Deno.version.deno}`);
console.log(`handle repr    : ${typeof c} (${Deno.UnsafePointer.value(c)})  empty handle -> ${JSON.stringify(m.mlx_array_new())}`);
console.log(`matmul [2,2]   : [${[...out].join(", ")}]  sum=${sum}  ${sum === 20 ? "OK" : "MISMATCH"}`);
console.log(`readback       : ${aliases ? "zero-copy view" : "COPY"}  (${(N * 4 / 1e6).toFixed(1)} MB in ${viewMs.toFixed(3)} ms)`);
console.log(`ffi overhead   : ${nsPerCall.toFixed(1)} ns/call (u32 return), ${nsBigInt.toFixed(1)} ns/call (u64 -> BigInt)`);

[a, b, c, big].forEach((h) => m.mlx_array_free(h));
