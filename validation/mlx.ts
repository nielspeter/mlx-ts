// Minimal Bun-FFI binding over mlx-c (libmlxc.dylib).
// Enough surface for a Qwen3 decoder block: matmul, rms_norm, rope, sdpa, silu.
//
// ABI note: every mlx-c handle is `struct { void* ctx; }`. On Apple-silicon
// ARM64 a single-pointer struct is passed/returned in a register exactly like a
// bare pointer, so we model every handle as `ptr` (a JS number). An *empty*
// handle has ctx==NULL, which Bun returns as `null` -> normalize to 0.

import { open, ptr } from "../src/ffi/index.ts";
import { LIBMLXC as LIB } from "../src/ffi/native-lib.ts";

export const m = open(LIB, {
  mlx_default_gpu_stream_new: { args: [], returns: "ptr" },
  // constructors / lifetime
  mlx_array_new:      { args: [], returns: "ptr" },
  mlx_array_new_data: { args: ["ptr", "ptr", "i32", "i32"], returns: "ptr" },
  mlx_array_free:     { args: ["ptr"], returns: "i32" },
  // eval + read-back
  mlx_array_eval:         { args: ["ptr"], returns: "i32" },
  mlx_array_item_float32: { args: ["ptr", "ptr"], returns: "i32" },
  mlx_array_tostring:     { args: ["ptr", "ptr"], returns: "i32" },
  mlx_string_new:  { args: [], returns: "ptr" },
  mlx_string_data: { args: ["ptr"], returns: "cstring" },
  // core ops
  mlx_matmul:    { args: ["ptr", "ptr", "ptr", "ptr"], returns: "i32" },
  mlx_add:       { args: ["ptr", "ptr", "ptr", "ptr"], returns: "i32" },
  mlx_multiply:  { args: ["ptr", "ptr", "ptr", "ptr"], returns: "i32" },
  mlx_sigmoid:   { args: ["ptr", "ptr", "ptr"], returns: "i32" },
  mlx_sum:       { args: ["ptr", "ptr", "bool", "ptr"], returns: "i32" },
  mlx_reshape:        { args: ["ptr", "ptr", "ptr", "u64", "ptr"], returns: "i32" },
  mlx_transpose_axes: { args: ["ptr", "ptr", "ptr", "u64", "ptr"], returns: "i32" },
  // fast ops (the inference-critical path)
  mlx_fast_rms_norm: { args: ["ptr", "ptr", "ptr", "f32", "ptr"], returns: "i32" },
  // rope: res, x, dims, traditional(bool), base(optional_float as u64),
  //       scale(f32), offset(i32), freqs(ptr|null), stream
  mlx_fast_rope: { args: ["ptr", "ptr", "i32", "bool", "u64", "f32", "i32", "ptr", "ptr"], returns: "i32" },
  // sdpa: res, q, k, v, scale(f32), mask_mode(char*), mask(ptr|null), sinks(ptr|null), stream
  mlx_fast_scaled_dot_product_attention: { args: ["ptr", "ptr", "ptr", "ptr", "f32", "ptr", "ptr", "ptr", "ptr"], returns: "i32" },
});

export const MLX_FLOAT32 = 10;
export const stream = m.mlx_default_gpu_stream_new() as number;

const asBig = (x: unknown) => BigInt((x as number) ?? 0);

// Keep JS-side buffers alive for the life of the process (MLX may read lazily).
const KEEP: unknown[] = [];

// Run an mlx-c `int fn(mlx_array* res, ...)` and return the new handle.
function out(fnName: keyof typeof m, ...rest: unknown[]): number {
  const slot = new BigUint64Array(1);
  slot[0] = asBig(m.mlx_array_new());
  (m[fnName] as (...a: unknown[]) => number)(ptr(slot), ...rest);
  return Number(slot[0]);
}

// ---- public array type + helpers ----------------------------------------
export type Arr = number;

export function array(data: Float32Array, shape: number[]): Arr {
  KEEP.push(data);
  const sh = new Int32Array(shape);
  KEEP.push(sh);
  return m.mlx_array_new_data(ptr(data), ptr(sh), shape.length, MLX_FLOAT32) as number;
}

export const matmul = (a: Arr, b: Arr): Arr => out("mlx_matmul", a, b, stream);
export const add = (a: Arr, b: Arr): Arr => out("mlx_add", a, b, stream);
export const mul = (a: Arr, b: Arr): Arr => out("mlx_multiply", a, b, stream);
export const sigmoid = (a: Arr): Arr => out("mlx_sigmoid", a, stream);
export const silu = (a: Arr): Arr => mul(a, sigmoid(a)); // x * sigmoid(x)

export const rmsNorm = (x: Arr, w: Arr, eps: number): Arr =>
  out("mlx_fast_rms_norm", x, w, eps, stream);

export function reshape(a: Arr, shape: number[]): Arr {
  const sh = new Int32Array(shape);
  KEEP.push(sh);
  return out("mlx_reshape", a, ptr(sh), BigInt(shape.length), stream);
}

export function transpose(a: Arr, axes: number[]): Arr {
  const ax = new Int32Array(axes);
  KEEP.push(ax);
  return out("mlx_transpose_axes", a, ptr(ax), BigInt(axes.length), stream);
}

// mlx_optional_float { float value; bool has_value; } packed into a u64:
// low 32 bits = float bits of `value`, byte 4 = has_value.
function optFloat(value: number): bigint {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0];
  return BigInt(bits) | (1n << 32n);
}

export const rope = (x: Arr, dims: number, base: number): Arr =>
  out("mlx_fast_rope", x, dims, false, optFloat(base), 1.0, 0, 0 /* null freqs */, stream);

const CAUSAL = new Uint8Array([...new TextEncoder().encode("causal"), 0]);
KEEP.push(CAUSAL);
export const sdpaCausal = (q: Arr, k: Arr, v: Arr, scale: number): Arr =>
  out("mlx_fast_scaled_dot_product_attention", q, k, v, scale, ptr(CAUSAL), 0, 0, stream);

// ---- read-back ----------------------------------------------------------
export function item(a: Arr): number {
  m.mlx_array_eval(a);
  const o = new Float32Array(1);
  m.mlx_array_item_float32(ptr(o), a);
  return o[0];
}
export const sumAll = (a: Arr): number => item(out("mlx_sum", a, false, stream));

export function show(a: Arr): string {
  const s = new BigUint64Array(1);
  s[0] = asBig(m.mlx_string_new());
  m.mlx_array_tostring(ptr(s), a);
  return m.mlx_string_data(Number(s[0])) as unknown as string;
}
