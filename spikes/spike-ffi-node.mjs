// SPIKE: mlx-c over Node via koffi (a prebuilt native addon — no C/C++ of ours,
// no build step for users). Must reproduce the spike-ffi-bun.ts fingerprint
// ([2.25, 4, 7.5, 6.25], sum 20) and answer the same three questions.
//   node spike-ffi-node.mjs
import koffi from "koffi";
import { LIBMLXC } from "./native-lib.ts";

const FLOAT32 = 10;
const lib = koffi.load(LIBMLXC);

// koffi takes C prototypes verbatim. `mlx_array` is `struct { void* ctx; }`,
// declared here as `void *` — the same single-pointer-in-a-register assumption
// bun:ffi and Deno make.
const mlx_default_gpu_stream_new = lib.func("void *mlx_default_gpu_stream_new()");
const mlx_array_new       = lib.func("void *mlx_array_new()");
const mlx_array_new_data  = lib.func("void *mlx_array_new_data(const void *data, const int *shape, int dim, int dtype)");
const mlx_matmul          = lib.func("int mlx_matmul(void *res, void *a, void *b, void *s)");
const mlx_array_eval      = lib.func("int mlx_array_eval(void *arr)");
const mlx_array_data_f32  = lib.func("const float *mlx_array_data_float32(void *arr)");
const mlx_array_ndim      = lib.func("size_t mlx_array_ndim(void *arr)");
const mlx_array_free      = lib.func("int mlx_array_free(void *arr)");

const stream = mlx_default_gpu_stream_new();
const arr = (data, shape) => mlx_array_new_data(data, new Int32Array(shape), shape.length, FLOAT32);

// --- 1+2: ABI and handle representation --------------------------------
const A = new Float32Array([1, 2, 3, 4, 5, 6]);
const B = new Float32Array([0.5, -1, 2, 0.25, -0.75, 1.5]);
const a = arr(A, [2, 3]), b = arr(B, [3, 2]);

// koffi hands pointers back as opaque externals. The `mlx_array*` out-slot is
// an 8-byte Buffer; encode/decode bridge between the slot and a callable
// pointer value.
const res = Buffer.alloc(8);
koffi.encode(res, "void *", mlx_array_new());
mlx_matmul(res, a, b, stream);
const c = koffi.decode(res, "void *");
mlx_array_eval(c);

const cp = mlx_array_data_f32(c);
// koffi.view(ptr, len) returns an ArrayBuffer over the native memory directly
// (koffi.decode() is the copying alternative).
const out = new Float32Array(koffi.view(cp, 4 * 4));
const sum = out.reduce((s, x) => s + x, 0);

// --- 3: is readback a view or a copy? ----------------------------------
const N = 2_000_000;
const big = arr(new Float32Array(N).fill(1.5), [N]);
mlx_array_eval(big);
const bp = mlx_array_data_f32(big);

const t0 = performance.now();
const v1 = new Float32Array(koffi.view(bp, N * 4));
const viewMs = performance.now() - t0;
const v2 = new Float32Array(koffi.view(bp, N * 4));
v1[0] = 12345;
const aliases = v2[0] === 12345;

// --- 4: per-call FFI overhead ------------------------------------------
// Measured twice for parity with the Bun/Deno spikes, which box a u64 return
// into a BigInt. koffi returns size_t as a plain number already, so these two
// should land close together — that difference is the point.
const mlx_array_ndim_u32 = lib.func("unsigned int mlx_array_ndim(void *arr)");
const bench = (fn) => {
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
const nsPerCall = bench(mlx_array_ndim_u32);
const nsBigInt = bench(mlx_array_ndim);

console.log(`runtime        : node ${process.versions.node} + koffi ${koffi.version}`);
console.log(`handle repr    : ${typeof c} (${koffi.address(c)})  empty handle -> ${JSON.stringify(mlx_array_new())}`);
console.log(`matmul [2,2]   : [${[...out].join(", ")}]  sum=${sum}  ${sum === 20 ? "OK" : "MISMATCH"}`);
console.log(`readback       : ${aliases ? "zero-copy view" : "COPY"}  (${(N * 4 / 1e6).toFixed(1)} MB in ${viewMs.toFixed(3)} ms)`);
console.log(`ffi overhead   : ${nsPerCall.toFixed(1)} ns/call (u32 return), ${nsBigInt.toFixed(1)} ns/call (size_t return)`);

[a, b, c, big].forEach((h) => mlx_array_free(h));
