// MX: an ergonomic, self-freeing array handle over mlx-c.
//
// - Every MX wraps one mlx-c handle and is registered with a FinalizationRegistry
//   so the handle is freed when the JS object is GC'd. (mlx-c handles are
//   refcounted, so freeing a JS handle that MLX still needs internally is safe.)
// - Ops call the generated `m` symbol table directly with local buffers, so no
//   shape/data buffers leak (MLX copies shapes synchronously; data buffers are
//   pinned to the owning MX instance).

import { ptr, toArrayBuffer } from "bun:ffi";
import { m, stream } from "./generated.ts";

const FLOAT32 = 10, INT32 = 7, UINT32 = 3;
const reg = new FinalizationRegistry<number>((h) => { if (h) m.mlx_array_free(h); });

// FinalizationRegistry is a *backstop*: it only fires after a GC, which never
// happens inside a tight synchronous decode loop, so memory would grow
// unbounded. `tidy()` is the deterministic fix — it tracks every MX created in
// its scope and frees all but the ones returned (cf. tf.tidy / mlx refcounting).
let ARENA: Set<MX> | null = null;
function collect(v: any, into: Set<MX>) {
  if (v instanceof MX) into.add(v);
  else if (Array.isArray(v)) for (const x of v) collect(x, into);
  else if (v && typeof v === "object") for (const x of Object.values(v)) collect(x, into);
}
export function tidy<T>(fn: () => T): T {
  const parent = ARENA, cur = new Set<MX>();
  ARENA = cur;
  let result: T;
  try { result = fn(); } finally { ARENA = parent; }
  const keep = new Set<MX>(); collect(result, keep);
  for (const x of cur) if (!keep.has(x)) x.free();
  if (parent) for (const k of keep) parent.add(k);
  return result;
}

function slot(): BigUint64Array { const r = new BigUint64Array(1); r[0] = BigInt((m.mlx_array_new() as number) ?? 0); return r; }
const optI = (v: number | null): bigint => v === null ? 0n : BigInt.asUintN(32, BigInt(v)) | (1n << 32n);
const optF = (v: number | null): bigint => v === null ? 0n : BigInt(new Uint32Array(new Float32Array([v]).buffer)[0]) | (1n << 32n);
const cstr = (s: string) => ptr(new Uint8Array([...new TextEncoder().encode(s), 0]));
const AFFINE = new Uint8Array([...new TextEncoder().encode("affine"), 0]);
const CAUSAL = new Uint8Array([...new TextEncoder().encode("causal"), 0]);
const NONE = new Uint8Array([0]);

export class MX {
  h: number;
  private pin?: ArrayBufferView; // keep source buffer alive for zero-copy arrays
  constructor(h: number, pin?: ArrayBufferView) { this.h = h; this.pin = pin; reg.register(this, h, this); ARENA?.add(this); }
  free() { reg.unregister(this); if (this.h) m.mlx_array_free(this.h); this.h = 0; }
  [Symbol.dispose]() { this.free(); } // `using a = x.add(y)` frees deterministically at scope end

  private r(fn: any, ...a: any[]): MX { const s = slot(); fn(ptr(s), ...a); return new MX(Number(s[0])); }

  // elementwise / linear
  matmul(o: MX) { return this.r(m.mlx_matmul, this.h, o.h, stream); }
  add(o: MX) { return this.r(m.mlx_add, this.h, o.h, stream); }
  mul(o: MX) { return this.r(m.mlx_multiply, this.h, o.h, stream); }
  div(o: MX) { return this.r(m.mlx_divide, this.h, o.h, stream); }
  sub(o: MX) { return this.r(m.mlx_subtract, this.h, o.h, stream); }
  sigmoid() { return this.r(m.mlx_sigmoid, this.h, stream); }
  silu() { return this.mul(this.sigmoid()); }
  tanh() { return this.r(m.mlx_tanh, this.h, stream); }
  sqrt() { return this.r(m.mlx_sqrt, this.h, stream); }
  meanAll() { const n = this.size; return this.sumAxes(this.shape.map((_, i) => i), false).div(scalar(n)); }

  // detach into a freshly-allocated buffer (stack always copies) — materializes
  // a tensor out of a shard's mmap so the shard can be unmapped.
  copy(): MX { const c = stack([this], 0).reshape(this.shape); c.eval(); return c; }

  // shape
  reshape(sh: number[]) { return this.r(m.mlx_reshape, this.h, ptr(new Int32Array(sh)), BigInt(sh.length), stream); }
  transpose(ax: number[]) { return this.r(m.mlx_transpose_axes, this.h, ptr(new Int32Array(ax)), BigInt(ax.length), stream); }
  // slice [start, stop) along leading axes (stride 1) — e.g. split a fused QKV weight
  slice(start: number[], stop: number[]) {
    const st = new Int32Array(start.length).fill(1);
    return this.r(m.mlx_slice, this.h, ptr(new Int32Array(start)), BigInt(start.length), ptr(new Int32Array(stop)), BigInt(stop.length), ptr(st), BigInt(st.length), stream);
  }

  // elementwise (activations)
  erf() { return this.r(m.mlx_erf, this.h, stream); }
  // exact GELU: x·0.5·(1+erf(x/√2)). Constants are hoisted (module-load singletons,
  // see below) so a training loop doesn't re-create three scalar arrays per call.
  gelu() { return this.mul(GH).mul(this.div(GR).erf().add(G1)); }
  astype(dtype: number) { return this.r(m.mlx_astype, this.h, dtype, stream); }

  // conv1d: this=input [N,L,C_in], w=[C_out,K,C_in] -> [N,L',C_out]
  conv1d(w: MX, stride: number, padding: number) { return this.r(m.mlx_conv1d, this.h, w.h, stride, padding, 1, 1, stream); }

  // fast ops
  rmsNorm(w: MX, eps: number) { return this.r(m.mlx_fast_rms_norm, this.h, w.h, eps, stream); }
  layerNorm(w: MX, b: MX, eps: number) { const s = slot(); m.mlx_fast_layer_norm(ptr(s), this.h, w.h, b.h, eps, stream); return new MX(Number(s[0])); }
  rope(dims: number, base: number, offset: number) { return this.r(m.mlx_fast_rope, this.h, dims, false, optF(base), 1.0, offset, 0, stream); }
  static sdpa(q: MX, k: MX, v: MX, scale: number, causal: boolean) {
    const s = slot(); m.mlx_fast_scaled_dot_product_attention(ptr(s), q.h, k.h, v.h, scale, ptr(causal ? CAUSAL : NONE), 0, 0, stream); return new MX(Number(s[0]));
  }

  // gather / quant
  takeAxis(idx: MX, axis: number) { return this.r(m.mlx_take_axis, this.h, idx.h, axis, stream); }
  takeAlong(idx: MX, axis: number) { return this.r(m.mlx_take_along_axis, this.h, idx.h, axis, stream); }
  qmm(wq: MX, scales: MX, biases: MX, gs: number, bits: number) {
    const s = slot(); m.mlx_quantized_matmul(ptr(s), this.h, wq.h, scales.h, biases.h, true, optI(gs), optI(bits), ptr(AFFINE), stream); return new MX(Number(s[0]));
  }
  static dequantize(wq: MX, scales: MX, biases: MX, gs: number, bits: number) {
    const s = slot(); m.mlx_dequantize(ptr(s), wq.h, scales.h, biases.h, optI(gs), optI(bits), ptr(AFFINE), 0, 0, stream); return new MX(Number(s[0]));
  }

  // MoE routing + expert dispatch
  neg() { return this.mul(scalar(-1)); }
  argpartition(kth: number, axis: number) { return this.r(m.mlx_argpartition_axis, this.h, kth, axis, stream); }
  sumAxes(axes: number[], keepdims: boolean) { return this.r(m.mlx_sum_axes, this.h, ptr(new Int32Array(axes)), BigInt(axes.length), keepdims, stream); }
  gatherQmm(wq: MX, scales: MX, biases: MX, rhsIdx: MX, gs: number, bits: number) {
    const s = slot(); m.mlx_gather_qmm(ptr(s), this.h, wq.h, scales.h, biases.h, 0, rhsIdx.h, true, optI(gs), optI(bits), ptr(AFFINE), false, stream); return new MX(Number(s[0]));
  }

  // reductions / sampling primitives
  argmax(axis: number) { return this.r(m.mlx_argmax_axis, this.h, axis, false, stream); }
  softmax(axis: number) { return this.r(m.mlx_softmax_axis, this.h, axis, true, stream); }
  logsumexp(axis: number, keepdims: boolean) { return this.r(m.mlx_logsumexp_axis, this.h, axis, keepdims, stream); }
  argsort(axis: number) { return this.r(m.mlx_argsort_axis, this.h, axis, stream); }
  cumsum(axis: number) { return this.r(m.mlx_cumsum, this.h, axis, false, true, stream); }
  log() { return this.r(m.mlx_log, this.h, stream); }
  greater(o: MX) { return this.r(m.mlx_greater, this.h, o.h, stream); }
  where(a: MX, b: MX) { return this.r(m.mlx_where, this.h, a.h, b.h, stream); } // this=cond
  divScalar(x: number) { return this.div(scalar(x)); }

  // cache concat (this ++ o along axis)
  concat(o: MX, axis: number) {
    const vbuf = new BigUint64Array([BigInt(this.h), BigInt(o.h)]);
    const vh = m.mlx_vector_array_new_data(ptr(vbuf), 2n) as number;
    const s = slot(); m.mlx_concatenate_axis(ptr(s), vh, axis, stream); m.mlx_vector_array_free(vh); return new MX(Number(s[0]));
  }

  // materialize / read back
  eval() { m.mlx_array_eval(this.h); return this; }
  get size() { return Number(m.mlx_array_size(this.h)); }
  get shape() { const n = Number(m.mlx_array_ndim(this.h)); const p = Number(m.mlx_array_shape(this.h)); return n ? Array.from(new Int32Array(toArrayBuffer(p, 0, n * 4))) : []; }
  itemF() { this.eval(); const o = new Float32Array(1); m.mlx_array_item_float32(ptr(o), this.h); return o[0]; }
  itemU() { this.eval(); const o = new Uint32Array(1); m.mlx_array_item_uint32(ptr(o), this.h); return o[0]; }
  toU32() { this.eval(); const n = this.size; const p = Number(m.mlx_array_data_uint32(this.h)); return Array.from(new Uint32Array(toArrayBuffer(p, 0, n * 4))); }
  toF32() { this.eval(); const n = this.size; const p = Number(m.mlx_array_data_float32(this.h)); return Array.from(new Float32Array(toArrayBuffer(p, 0, n * 4))); }
}

// constructors
export function fromF32(data: Float32Array, shape: number[]): MX {
  return new MX(m.mlx_array_new_data(ptr(data), ptr(new Int32Array(shape)), shape.length, FLOAT32) as number, data);
}
export function fromI32(data: Int32Array, shape: number[]): MX {
  return new MX(m.mlx_array_new_data(ptr(data), ptr(new Int32Array(shape)), shape.length, INT32) as number, data);
}
// uint32 — e.g. packed quantized weights and gather indices (bit pattern matters,
// so these can't round-trip through float32).
export function fromU32(data: Uint32Array, shape: number[]): MX {
  return new MX(m.mlx_array_new_data(ptr(data), ptr(new Int32Array(shape)), shape.length, UINT32) as number, data);
}
export const scalar = (x: number): MX => new MX(m.mlx_array_new_float(x) as number);
// GELU constant singletons — allocated once at module load (ARENA is null here, so
// they're never captured/freed by a tidy() scope). Used by MX.prototype.gelu().
const GH = scalar(0.5), G1 = scalar(1), GR = scalar(Math.SQRT2);
// stack a list of arrays along a new axis (e.g. per-expert weights -> [E, ...])
export function stack(arrs: MX[], axis: number): MX {
  const buf = new BigUint64Array(arrs.map((a) => BigInt(a.h)));
  const vh = m.mlx_vector_array_new_data(ptr(buf), BigInt(arrs.length)) as number;
  const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_array_new() as number) ?? 0);
  m.mlx_stack_axis(ptr(s), vh, axis, stream); m.mlx_vector_array_free(vh); return new MX(Number(s[0]));
}

// batched eval: force everything before reading (keeps the lazy graph from
// growing). One mlx_eval over a vector, not N per-array evals — each per-array
// eval is a separate GPU sync, so batching is ~Nx faster for multi-output steps
// (measured: 20-output take 1.84 -> 0.14 ms). Used on the decode hot path
// (token + all KV-cache arrays).
export function evalAll(...xs: MX[]) {
  if (xs.length === 0) return;
  if (xs.length === 1) { m.mlx_array_eval(xs[0].h); return; }
  const buf = new BigUint64Array(xs.map((x) => BigInt(x.h)));
  const vh = m.mlx_vector_array_new_data(ptr(buf), BigInt(xs.length)) as number;
  m.mlx_eval(vh); m.mlx_vector_array_free(vh);
}
// async eval: queue the graph on the stream and return immediately (no host sync),
// so the next step's graph can be built while the GPU runs this one.
export function asyncEval(...xs: MX[]) {
  const buf = new BigUint64Array(xs.map((x) => BigInt(x.h)));
  const vh = m.mlx_vector_array_new_data(ptr(buf), BigInt(xs.length)) as number;
  m.mlx_async_eval(vh); m.mlx_vector_array_free(vh);
}
export function seed(s: number) { if (m.mlx_random_seed) m.mlx_random_seed(BigInt(s)); }

// Write a {name -> MX} record to a .safetensors file (the save side of loader.ts).
// Keys become tensor names; pass a flattened param tree (e.g. "blocks.0.wq").
// Used for training checkpoints (pretrain -> SFT/inference handoff).
export function saveSafetensors(path: string, record: Record<string, MX>) {
  const vals = Object.values(record); evalAll(...vals);          // materialize before writing
  const map = m.mlx_map_string_to_array_new() as number;
  for (const [k, v] of Object.entries(record)) {
    const key = new Uint8Array([...new TextEncoder().encode(k), 0]); // keep alive across the call
    m.mlx_map_string_to_array_insert(map, ptr(key), v.h);
  }
  const meta = m.mlx_map_string_to_string_new() as number;        // empty metadata
  const file = new Uint8Array([...new TextEncoder().encode(path), 0]);
  m.mlx_save_safetensors(ptr(file), map, meta);
  m.mlx_map_string_to_array_free?.(map);
}
const memOf = (fn: any): number => { const o = new BigUint64Array(1); fn(ptr(o)); return Number(o[0]) / 1e6; };
export const activeMemoryMB = (): number => memOf(m.mlx_get_active_memory);
export const peakMemoryMB = (): number => memOf(m.mlx_get_peak_memory);
export const cacheMemoryMB = (): number => memOf(m.mlx_get_cache_memory);
export const clearCache = (): void => { m.mlx_clear_cache(); };        // return reuse-cache buffers to the OS
export const resetPeakMemory = (): void => { m.mlx_reset_peak_memory(); };
export const setMemoryLimit = (mb: number): void => { const o = new BigUint64Array(1); m.mlx_set_memory_limit(ptr(o), BigInt(Math.round(mb * 1e6))); };
export const setWiredLimit = (mb: number): void => { const o = new BigUint64Array(1); if (m.mlx_set_wired_limit) m.mlx_set_wired_limit(ptr(o), BigInt(Math.round(mb * 1e6))); };

function categorical(logits: MX, axis: number): MX {
  const s = slot(); m.mlx_random_categorical(ptr(s), logits.h, axis, 0, stream); return new MX(Number(s[0]));
}

// Inverted dropout (device-side): keep each element w.p. (1-p) via a Bernoulli
// mask, scale survivors by 1/(1-p). `seedNum` derives the RNG key — pass a value
// that varies per call (e.g. step*K + site) so masks differ across steps but are
// reproducible (MLX's bernoulli is deterministic given key + shape, so a Python
// mirror using the same seeds gets identical masks). No-op at p<=0 (eval/inference).
export function dropout(x: MX, p: number, seedNum: number): MX {
  if (p <= 0) return x;
  const keep = 1 - p;
  const ks = slot(); m.mlx_random_key(ptr(ks), BigInt(seedNum)); const key = new MX(Number(ks[0]));
  const sh = x.shape;
  const s = slot(); m.mlx_random_bernoulli(ptr(s), scalar(keep).h, ptr(new Int32Array(sh)), BigInt(sh.length), key.h, stream);
  return x.mul(new MX(Number(s[0])).astype(FLOAT32)).divScalar(keep);
}

// top-k: keep the k highest logits along `ax`, mask the rest to ~-inf.
function topKFilter(logits: MX, k: number, ax: number): MX {
  const vocab = logits.shape[ax];
  if (k <= 0 || k >= vocab) return logits;
  const sortedAsc = logits.takeAlong(logits.argsort(ax), ax);                       // values ascending
  const thr = sortedAsc.takeAxis(fromI32(Int32Array.from([vocab - k]), [1]), ax);   // k-th largest -> [B,1]
  return thr.greater(logits).where(scalar(-1e9), logits);                           // drop logits below it
}

// Repetition penalty (CTRL/HF): logits of already-seen tokens are divided by
// `penalty` when positive and multiplied when negative — discouraging repeats.
// Apply to logits before sampling; `prev` is the token history (prompt + gen).
export function applyRepetitionPenalty(logits: MX, prev: number[], penalty: number): MX {
  if (penalty === 1 || prev.length === 0) return logits;
  const vocab = logits.shape[logits.shape.length - 1];
  const factor = new Float32Array(vocab).fill(1);
  for (const id of prev) if (id >= 0 && id < vocab) factor[id] = penalty;
  const f = fromF32(factor, [vocab]);
  return logits.greater(scalar(0)).where(logits.div(f), logits.mul(f));
}

// ---- sampling: logits [B, vocab] -> token ids [B] ----
// Filters compose as temp -> top-k -> top-p -> categorical (apply repetition
// penalty to the logits beforehand, e.g. via applyRepetitionPenalty).
export function sample(logits: MX, temp: number, topP: number, topK = 0): MX {
  const B = logits.shape[0], ax = 1;               // logits is [B, vocab]
  if (temp === 0) return logits.argmax(ax);        // greedy -> [B]
  let scaled = logits.divScalar(temp);
  if (topK > 0) scaled = topKFilter(scaled, topK, ax);
  if (topP > 0 && topP < 1) {
    const probs = scaled.softmax(ax);
    const idx = probs.argsort(ax);                 // ascending order
    const sp = probs.takeAlong(idx, ax);           // probs in ascending order
    const keep = sp.cumsum(ax).greater(scalar(1 - topP)); // keep the top-p mass (high end)
    const masked = keep.where(sp, scalar(0));
    const sortedTok = categorical(masked.log(), ax).reshape([B, 1]); // index into sorted
    return idx.takeAlong(sortedTok, ax).reshape([B]); // map back to vocab id
  }
  return categorical(scaled, ax);
}
