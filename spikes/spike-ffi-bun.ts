// SPIKE: is the mlx-c FFI layer portable off Bun? — the Bun baseline.
//
// mlx-ts is Bun-FFI-only (README ❌ list). Node has the users, Deno has the
// closest FFI. Before committing to a backend split (src/core/ffi/{bun,node,
// deno}.ts) this measures the three things that actually decide it:
//
//   1. ABI      — `mlx_array` is `struct { void* ctx; }`; does each runtime's
//                 FFI pass/return it in a register like a bare pointer?
//   2. handles  — bun:ffi gives pointers back as JS *numbers*, which the whole
//                 codebase assumes (`export type Arr = number`). What do the
//                 others give, and what does normalising cost?
//   3. readback — `toArrayBuffer()` is a zero-copy *view* over native memory.
//                 mx.ts reads logits through it every token and loader.ts maps
//                 multi-GB safetensors with it. A backend that copies instead
//                 is not a drop-in.
//
// All three spikes run the same [2,3] @ [3,2] matmul -> [2.25, 4, 7.5, 6.25],
// sum exactly 20, so a mismatch is visible without a tolerance argument.
//   bun spike-ffi-bun.ts
import { dlopen, ptr, toArrayBuffer as bunToArrayBuffer, type Pointer } from "bun:ffi";
import { LIBMLXC } from "../src/ffi/native-lib.ts";

// This spike exists to show that a handle is a plain JS number — which is the
// premise src/ffi/ is built on. bun:ffi types pointers as a branded `Pointer`,
// so narrow once here instead of casting at every call site.
const addr = (h: unknown) => (h ?? 0) as unknown as Pointer;
const toArrayBuffer = (p: number, off: number, len: number) => bunToArrayBuffer(addr(p), off, len);

const FLOAT32 = 10;
const { symbols: raw } = dlopen(LIBMLXC, {
  mlx_default_gpu_stream_new: { args: [], returns: "ptr" },
  mlx_array_new:             { args: [], returns: "ptr" },
  mlx_array_new_data:        { args: ["ptr", "ptr", "i32", "i32"], returns: "ptr" },
  mlx_matmul:                { args: ["ptr", "ptr", "ptr", "ptr"], returns: "i32" },
  mlx_array_eval:            { args: ["ptr"], returns: "i32" },
  mlx_array_data_float32:    { args: ["ptr"], returns: "ptr" },
  mlx_array_ndim:            { args: ["ptr"], returns: "u64" },
  mlx_array_free:            { args: ["ptr"], returns: "i32" },
});
// Handles cross as numbers here, deliberately — see the note above.
const m = raw as unknown as Record<keyof typeof raw, (...a: any[]) => any>;

const stream = m.mlx_default_gpu_stream_new();
const arr = (data: Float32Array, shape: number[]) =>
  m.mlx_array_new_data(ptr(data), ptr(new Int32Array(shape)), shape.length, FLOAT32);

// --- 1+2: ABI and handle representation --------------------------------
const A = new Float32Array([1, 2, 3, 4, 5, 6]);
const B = new Float32Array([0.5, -1, 2, 0.25, -0.75, 1.5]);
const a = arr(A, [2, 3]), b = arr(B, [3, 2]);

const res = new BigUint64Array(1);
// An *empty* handle has ctx==NULL, which bun:ffi hands back as `null`, not 0 —
// hence the `?? 0` that `asBig()` does throughout generated.ts.
res[0] = BigInt((m.mlx_array_new() as number) ?? 0);   // res is `mlx_array*` -> a u64 slot
m.mlx_matmul(ptr(res), a, b, stream);
const c = Number(res[0]);
m.mlx_array_eval(c);

const cp = Number(m.mlx_array_data_float32(c));
const out = new Float32Array(toArrayBuffer(cp, 0, 4 * 4));
const sum = out.reduce((s, x) => s + x, 0);

// --- 3: is readback a view or a copy? ----------------------------------
// Definitive test: two "views" of the same address. Write through one; if the
// other sees it, they alias native memory. If not, the runtime copied.
const N = 2_000_000;                            // 8 MB — a copy would show up
const big = arr(new Float32Array(N).fill(1.5), [N]);
m.mlx_array_eval(big);
const bp = Number(m.mlx_array_data_float32(big));

const t0 = performance.now();
const v1 = new Float32Array(toArrayBuffer(bp, 0, N * 4));
const viewMs = performance.now() - t0;
const v2 = new Float32Array(toArrayBuffer(bp, 0, N * 4));
v1[0] = 12345;
const aliases = v2[0] === 12345;

// --- 4: per-call FFI overhead ------------------------------------------
// MLX is lazy: a decode step is many cheap enqueue calls, so dispatch cost
// matters more than FLOPs suggest. mlx_array_ndim is a trivial C accessor.
//
// Measured twice, because the return type is not neutral: bun:ffi and Deno box
// a u64 into a BigInt (an allocation per call), while koffi hands size_t back
// as a plain number. Declaring the return u32 reads w0 instead of x0 — exact
// for ndim, and the apples-to-apples number.
const { symbols: m32 } = dlopen(LIBMLXC, { mlx_array_ndim: { args: ["ptr"], returns: "u32" } });
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
const nsPerCall = bench(m32.mlx_array_ndim);
const nsBigInt = bench(m.mlx_array_ndim);

console.log(`runtime        : bun ${Bun.version}`);
console.log(`handle repr    : ${typeof c} (${c})  empty handle -> ${JSON.stringify(m.mlx_array_new())}`);
console.log(`matmul [2,2]   : [${[...out].join(", ")}]  sum=${sum}  ${sum === 20 ? "OK" : "MISMATCH"}`);
console.log(`readback       : ${aliases ? "zero-copy view" : "COPY"}  (${(N * 4 / 1e6).toFixed(1)} MB in ${viewMs.toFixed(3)} ms)`);
console.log(`ffi overhead   : ${nsPerCall.toFixed(1)} ns/call (u32 return), ${nsBigInt.toFixed(1)} ns/call (u64 -> BigInt)`);

[a, b, c, big].forEach((h) => m.mlx_array_free(h));
