// MX: an ergonomic, self-freeing array handle over mlx-c.
//
// - Every MX wraps one mlx-c handle and is registered with a FinalizationRegistry
//   so the handle is freed when the JS object is GC'd. (mlx-c handles are
//   refcounted, so freeing a JS handle that MLX still needs internally is safe.)
// - Ops call the generated `m` symbol table directly with local buffers, so no
//   shape/data buffers leak (MLX copies shapes synchronously; data buffers are
//   pinned to the owning MX instance).

import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { m, pendingError, stream, throwIfError } from "../ffi/generated.ts";
import { ptr, view as toArrayBuffer } from "../ffi/index.ts";

// MLX's dtype enum, for astype(). Exported because a parity check needs to read
// a bf16 checkpoint at full precision: bf16 is fine for generation but too
// coarse to compare against a float32 reference.
export const FLOAT32 = 10;
const INT32 = 7,
  UINT32 = 3;
const reg = new FinalizationRegistry<number>((h) => {
  if (h) m.mlx_array_free(h);
});

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
  const parent = ARENA,
    cur = new Set<MX>();
  ARENA = cur;
  let result: T;
  try {
    result = fn();
  } catch (e) {
    // Nothing survives a scope that failed. Cleanup placed after a try/finally
    // is skipped exactly here — on the error path, where a long-lived process
    // needs deterministic freeing most.
    ARENA = parent;
    for (const x of cur) x.free();
    throw e;
  }
  ARENA = parent;
  const keep = new Set<MX>();
  collect(result, keep);
  for (const x of cur) if (!keep.has(x)) x.free();
  // Hand the parent only what this scope created. `keep` can also hold arrays
  // from outside — a nested tidy() returning one of its inputs, say — and those
  // belong to whoever made them. Adopting one lets the outer scope free an
  // array its caller still holds.
  if (parent) for (const k of keep) if (cur.has(k)) parent.add(k);
  return result;
}

// Take arrays out of the enclosing tidy() arena so they survive it. The caller
// becomes responsible for free()ing them.
//
// This exists for state that must outlive the step that produced it — an
// optimizer's moment buffers, say. Without it an optimizer used inside tidy()
// has its state freed underneath it on the next step, which is why the
// hand-rolled AdamW in training/ keeps its state at module scope.
export function escape<T>(v: T): T {
  if (ARENA) {
    const s = new Set<MX>();
    collect(v, s);
    for (const x of s) ARENA.delete(x);
  }
  return v;
}

/** Free every MX reachable from `v`. Arrays and plain objects are walked. */
export function freeAll(v: unknown): void {
  const s = new Set<MX>();
  collect(v, s);
  for (const x of s) x.free();
}

/**
 * A slot table that owns what it holds — the other half of the memory model.
 *
 * `tidy()` frees what a scope created and `escape()` lifts a value out of that
 * scope, but escape() only transfers ownership; it does not say who frees the
 * value being replaced. That part was hand-rolled at both of its call sites —
 * an optimizer's moment buffers and a KV cache — and getting it wrong in the
 * second one leaked ~10 MB per step and reached 55 GB on a 39 GB machine.
 *
 * Here it happens once, in set(). The value escapes the caller's arena and the
 * previous occupant is freed, minus anything the new value still holds:
 * replacing `{k, v}` with `{k, v2}` must not free the `k` that is still live.
 *
 *   using cache = new Owned<LayerKV>(nLayers);
 *   cache.set(l, { k, v });        // frees the pair it replaced
 *                                  // ...and the whole table at scope exit
 */
export class Owned<T> {
  private slots: (T | null)[];

  constructor(n: number) {
    this.slots = new Array<T | null>(n).fill(null);
  }

  get length(): number {
    return this.slots.length;
  }

  get(i: number): T | null {
    return this.slots[i] ?? null;
  }

  /** Take ownership of `v`, freeing what slot `i` held. Returns `v`. */
  set(i: number, v: T): T {
    // Grow on demand: an optimizer discovers how many leaves it has by walking
    // the tree, so it cannot size the table up front.
    while (this.slots.length <= i) this.slots.push(null);
    const prev = this.slots[i];
    this.slots[i] = escape(v);
    if (prev != null && prev !== v) {
      const keep = new Set<MX>();
      collect(v, keep);
      const old = new Set<MX>();
      collect(prev, old);
      for (const x of old) if (!keep.has(x)) x.free();
    }
    return v;
  }

  /** Free every slot and empty the table. Safe to call more than once. */
  free(): void {
    for (let i = 0; i < this.slots.length; i++) {
      const v = this.slots[i];
      this.slots[i] = null;
      if (v != null) freeAll(v);
    }
  }

  [Symbol.dispose]() {
    this.free();
  }
}

/**
 * Wrap a result handle, raising anything mlx-c reported first.
 *
 * Every native op has to consume the error state, not just the ones that go
 * through MX.r(). An op that leaves an error unconsumed does not fail itself —
 * it poisons the *next* unrelated call, which then throws someone else's
 * message under its own name.
 */
function mk(op: string, s: BigUint64Array): MX {
  if (pendingError() !== null) throwIfError(op, Number(s[0]));
  return new MX(Number(s[0]));
}

function slot(): BigUint64Array {
  const r = new BigUint64Array(1);
  r[0] = BigInt((m.mlx_array_new() as number) ?? 0);
  return r;
}
const optI = (v: number | null): bigint => (v === null ? 0n : BigInt.asUintN(32, BigInt(v)) | (1n << 32n));
const optF = (v: number | null): bigint =>
  v === null ? 0n : BigInt(new Uint32Array(new Float32Array([v]).buffer)[0]) | (1n << 32n);
const AFFINE = new Uint8Array([...new TextEncoder().encode("affine"), 0]);
const CAUSAL = new Uint8Array([...new TextEncoder().encode("causal"), 0]);
const NONE = new Uint8Array([0]);

export class MX {
  h: number;
  // No reference to the source TypedArray: mlx_array_new_data copies at
  // construction — verified by mutating the buffer before any eval and reading
  // the original values back — so retaining one only pinned JS memory.
  constructor(h: number) {
    this.h = h;
    reg.register(this, h, this);
    ARENA?.add(this);
  }
  free() {
    reg.unregister(this);
    if (this.h) m.mlx_array_free(this.h);
    this.h = 0;
  }
  [Symbol.dispose]() {
    this.free();
  } // `using a = x.add(y)` frees deterministically at scope end

  // Every op funnels through here, so this is where mlx-c failures become
  // TypeScript exceptions. One null comparison per operation; without it an
  // invalid shape reaches mlx-c's default handler, which prints and exits.
  private r(fn: any, ...a: any[]): MX {
    const s = slot();
    fn(ptr(s), ...a);
    if (pendingError() !== null) throwIfError(fn.name || "mlx-c op", Number(s[0]));
    return new MX(Number(s[0]));
  }

  // elementwise / linear
  matmul(o: MX) {
    return this.r(m.mlx_matmul, this.h, o.h, stream);
  }
  add(o: MX) {
    return this.r(m.mlx_add, this.h, o.h, stream);
  }
  mul(o: MX) {
    return this.r(m.mlx_multiply, this.h, o.h, stream);
  }
  div(o: MX) {
    return this.r(m.mlx_divide, this.h, o.h, stream);
  }
  sub(o: MX) {
    return this.r(m.mlx_subtract, this.h, o.h, stream);
  }
  sigmoid() {
    return this.r(m.mlx_sigmoid, this.h, stream);
  }
  silu() {
    return this.mul(this.sigmoid());
  }
  tanh() {
    return this.r(m.mlx_tanh, this.h, stream);
  }
  sqrt() {
    return this.r(m.mlx_sqrt, this.h, stream);
  }
  sin() {
    return this.r(m.mlx_sin, this.h, stream);
  }
  square() {
    return this.r(m.mlx_square, this.h, stream);
  }
  /** Elementwise max against a scalar — ReLU is `x.maximum(0)`. */
  maximum(v: number) {
    return this.r(m.mlx_maximum, this.h, scalar(v).h, stream);
  }
  meanAll() {
    const n = this.size;
    return this.sumAxes(
      this.shape.map((_, i) => i),
      false,
    ).div(scalar(n));
  }

  // detach into a freshly-allocated buffer (stack always copies) — materializes
  // a tensor out of a shard's mmap so the shard can be unmapped.
  copy(): MX {
    const c = stack([this], 0).reshape(this.shape);
    c.eval();
    return c;
  }

  // shape
  reshape(sh: number[]) {
    return this.r(m.mlx_reshape, this.h, ptr(new Int32Array(sh)), BigInt(sh.length), stream);
  }
  broadcastTo(sh: number[]) {
    return this.r(m.mlx_broadcast_to, this.h, ptr(new Int32Array(sh)), BigInt(sh.length), stream);
  }
  transpose(ax: number[]) {
    return this.r(m.mlx_transpose_axes, this.h, ptr(new Int32Array(ax)), BigInt(ax.length), stream);
  }
  // slice [start, stop) along leading axes (stride 1) — e.g. split a fused QKV weight
  slice(start: number[], stop: number[]) {
    const st = new Int32Array(start.length).fill(1);
    return this.r(
      m.mlx_slice,
      this.h,
      ptr(new Int32Array(start)),
      BigInt(start.length),
      ptr(new Int32Array(stop)),
      BigInt(stop.length),
      ptr(st),
      BigInt(st.length),
      stream,
    );
  }

  // elementwise (activations)
  erf() {
    return this.r(m.mlx_erf, this.h, stream);
  }
  // exact GELU: x·0.5·(1+erf(x/√2)). Constants are hoisted (module-load singletons,
  // see below) so a training loop doesn't re-create three scalar arrays per call.
  gelu() {
    return this.mul(GH).mul(this.div(GR).erf().add(G1));
  }
  astype(dtype: number) {
    return this.r(m.mlx_astype, this.h, dtype, stream);
  }

  // conv1d: this=input [N,L,C_in], w=[C_out,K,C_in] -> [N,L',C_out]
  // groups > 1 gives a depthwise convolution, which ConvNeXt blocks need.
  conv1d(w: MX, stride: number, padding: number, dilation = 1, groups = 1) {
    return this.r(m.mlx_conv1d, this.h, w.h, stride, padding, dilation, groups, stream);
  }

  // The 1-D upsampler behind every neural vocoder: same channels-last layout as
  // conv1d, with output padding to break the tie between the output lengths a
  // strided convolution leaves ambiguous.
  convTranspose1d(w: MX, stride = 1, padding = 0, dilation = 1, outputPadding = 0, groups = 1) {
    return this.r(
      m.mlx_conv_transpose1d,
      this.h,
      w.h,
      stride,
      padding,
      dilation,
      outputPadding,
      groups,
      stream,
    );
  }

  // conv2d: this=input [N,H,W,C_in], w=[C_out,KH,KW,C_in] -> [N,H',W',C_out].
  // Channels-last, like conv1d and like MLX itself — PyTorch checkpoints are
  // stored [C_out,C_in,KH,KW], so a port transposes the weights on load.
  conv2d(
    w: MX,
    stride: [number, number] = [1, 1],
    padding: [number, number] = [0, 0],
    dilation: [number, number] = [1, 1],
    groups = 1,
  ) {
    return this.r(
      m.mlx_conv2d,
      this.h,
      w.h,
      stride[0],
      stride[1],
      padding[0],
      padding[1],
      dilation[0],
      dilation[1],
      groups,
      stream,
    );
  }

  // The upsampling half of a decoder: same layout, plus output padding to pick
  // between the output sizes a strided convolution leaves ambiguous.
  convTranspose2d(
    w: MX,
    stride: [number, number] = [1, 1],
    padding: [number, number] = [0, 0],
    dilation: [number, number] = [1, 1],
    outputPadding: [number, number] = [0, 0],
    groups = 1,
  ) {
    return this.r(
      m.mlx_conv_transpose2d,
      this.h,
      w.h,
      stride[0],
      stride[1],
      padding[0],
      padding[1],
      dilation[0],
      dilation[1],
      outputPadding[0],
      outputPadding[1],
      groups,
      stream,
    );
  }

  // fast ops
  rmsNorm(w: MX, eps: number) {
    return this.r(m.mlx_fast_rms_norm, this.h, w.h, eps, stream);
  }
  // w/b are optional: mlx-c reads an empty handle (0) as "not provided", which is
  // how GroupNorm normalises before applying its own affine parameters.
  layerNorm(w: MX | null, b: MX | null, eps: number) {
    const s = slot();
    m.mlx_fast_layer_norm(ptr(s), this.h, w ? w.h : 0, b ? b.h : 0, eps, stream);
    return mk("mlx_fast_layer_norm", s);
  }
  rope(dims: number, base: number, offset: number) {
    return this.r(m.mlx_fast_rope, this.h, dims, false, optF(base), 1.0, offset, 0, stream);
  }
  static sdpa(q: MX, k: MX, v: MX, scale: number, causal: boolean) {
    const s = slot();
    m.mlx_fast_scaled_dot_product_attention(
      ptr(s),
      q.h,
      k.h,
      v.h,
      scale,
      ptr(causal ? CAUSAL : NONE),
      0,
      0,
      stream,
    );
    return mk("mlx_fast_scaled_dot_product_attention", s);
  }

  // gather / quant
  takeAxis(idx: MX, axis: number) {
    return this.r(m.mlx_take_axis, this.h, idx.h, axis, stream);
  }
  takeAlong(idx: MX, axis: number) {
    return this.r(m.mlx_take_along_axis, this.h, idx.h, axis, stream);
  }
  qmm(wq: MX, scales: MX, biases: MX, gs: number, bits: number) {
    const s = slot();
    m.mlx_quantized_matmul(
      ptr(s),
      this.h,
      wq.h,
      scales.h,
      biases.h,
      true,
      optI(gs),
      optI(bits),
      ptr(AFFINE),
      stream,
    );
    return mk("mlx_quantized_matmul", s);
  }
  static dequantize(wq: MX, scales: MX, biases: MX, gs: number, bits: number) {
    const s = slot();
    m.mlx_dequantize(ptr(s), wq.h, scales.h, biases.h, optI(gs), optI(bits), ptr(AFFINE), 0, 0, stream);
    return mk("mlx_dequantize", s);
  }

  // MoE routing + expert dispatch
  neg() {
    return this.mul(scalar(-1));
  }
  argpartition(kth: number, axis: number) {
    return this.r(m.mlx_argpartition_axis, this.h, kth, axis, stream);
  }
  sumAxes(axes: number[], keepdims: boolean) {
    return this.r(m.mlx_sum_axes, this.h, ptr(new Int32Array(axes)), BigInt(axes.length), keepdims, stream);
  }
  gatherQmm(wq: MX, scales: MX, biases: MX, rhsIdx: MX, gs: number, bits: number) {
    const s = slot();
    m.mlx_gather_qmm(
      ptr(s),
      this.h,
      wq.h,
      scales.h,
      biases.h,
      0,
      rhsIdx.h,
      true,
      optI(gs),
      optI(bits),
      ptr(AFFINE),
      false,
      stream,
    );
    return mk("mlx_gather_qmm", s);
  }

  // reductions / sampling primitives
  argmax(axis: number) {
    return this.r(m.mlx_argmax_axis, this.h, axis, false, stream);
  }
  softmax(axis: number) {
    return this.r(m.mlx_softmax_axis, this.h, axis, true, stream);
  }
  logsumexp(axis: number, keepdims: boolean) {
    return this.r(m.mlx_logsumexp_axis, this.h, axis, keepdims, stream);
  }
  argsort(axis: number) {
    return this.r(m.mlx_argsort_axis, this.h, axis, stream);
  }
  cumsum(axis: number) {
    return this.r(m.mlx_cumsum, this.h, axis, false, true, stream);
  }
  log() {
    return this.r(m.mlx_log, this.h, stream);
  }
  greater(o: MX) {
    return this.r(m.mlx_greater, this.h, o.h, stream);
  }
  where(a: MX, b: MX) {
    return this.r(m.mlx_where, this.h, a.h, b.h, stream);
  } // this=cond
  divScalar(x: number) {
    return this.div(scalar(x));
  }
  mulScalar(x: number) {
    return this.mul(scalar(x));
  }
  addScalar(x: number) {
    return this.add(scalar(x));
  }

  // cache concat (this ++ o along axis)
  concat(o: MX, axis: number) {
    const vbuf = new BigUint64Array([BigInt(this.h), BigInt(o.h)]);
    const vh = m.mlx_vector_array_new_data(ptr(vbuf), 2n) as number;
    const s = slot();
    m.mlx_concatenate_axis(ptr(s), vh, axis, stream);
    m.mlx_vector_array_free(vh);
    return mk("mlx_concatenate_axis", s);
  }

  // materialize / read back
  // Errors surface here as well as at graph construction: a bad shape is caught
  // when the op is built, an allocation failure only when it runs.
  eval() {
    m.mlx_array_eval(this.h);
    if (pendingError() !== null) throwIfError("mlx_array_eval");
    return this;
  }
  get size() {
    return Number(m.mlx_array_size(this.h));
  }
  get shape() {
    const n = Number(m.mlx_array_ndim(this.h));
    const p = Number(m.mlx_array_shape(this.h));
    return n ? Array.from(new Int32Array(toArrayBuffer(p, 0, n * 4))) : [];
  }
  /**
   * Row-major strides, in elements. A `transpose()` produces a *view*: the shape
   * changes but the buffer does not, so anything that reads the raw pointer has
   * to check this first or it reads the pre-transpose order under the new shape.
   *
   * Describes the array as currently materialised, so `eval()` first — MLX does
   * not fix the layout of a pending graph, and an unevaluated transpose still
   * reports its source's strides.
   */
  get strides(): number[] {
    const n = Number(m.mlx_array_ndim(this.h));
    const p = Number(m.mlx_array_strides(this.h));
    return n ? Array.from(new BigUint64Array(toArrayBuffer(p, 0, n * 8)), Number) : [];
  }

  /** Whether the buffer is laid out in the order `shape` implies; see `strides`. */
  get contiguous(): boolean {
    const sh = this.shape,
      st = this.strides;
    let expect = 1;
    for (let i = sh.length - 1; i >= 0; i--) {
      if (sh[i] !== 1 && st[i] !== expect) return false;
      expect *= sh[i];
    }
    return true;
  }

  itemF() {
    this.eval();
    const o = new Float32Array(1);
    m.mlx_array_item_float32(ptr(o), this.h);
    return o[0];
  }
  itemU() {
    this.eval();
    const o = new Uint32Array(1);
    m.mlx_array_item_uint32(ptr(o), this.h);
    return o[0];
  }
  toU32(): number[] {
    this.eval();
    if (!this.contiguous) {
      const c = this.copy();
      try {
        return c.toU32();
      } finally {
        c.free();
      }
    }
    const n = this.size;
    const p = Number(m.mlx_array_data_uint32(this.h));
    return Array.from(new Uint32Array(toArrayBuffer(p, 0, n * 4)));
  }
  /** MLX's dtype enum for this array — FLOAT32 is 10, INT32 7, UINT32 3. */
  get dtype(): number {
    return m.mlx_array_dtype(this.h) as number;
  }

  // Casts first when the array is not already float32. mlx_array_data_float32
  // on a bfloat16 array returns null, and reading from it segfaults rather than
  // erroring — which is exactly what a bf16 checkpoint's logits do.
  toF32(): number[] {
    if (this.dtype !== FLOAT32) {
      const c = this.astype(FLOAT32);
      try {
        return c.toF32();
      } finally {
        c.free();
      }
    }
    this.eval();
    if (!this.contiguous) {
      const c = this.copy();
      try {
        return c.toF32();
      } finally {
        c.free();
      }
    }
    const n = this.size;
    const p = Number(m.mlx_array_data_float32(this.h));
    return Array.from(new Float32Array(toArrayBuffer(p, 0, n * 4)));
  }
}

// constructors
/**
 * Validate a host array against the shape it is given.
 *
 * mlx-c takes a pointer and a shape but no byte length, so a shape larger than
 * the buffer makes native code read past the TypedArray — silently, returning
 * whatever follows it. A fractional dimension is truncated by the Int32Array
 * cast, and a negative one reaches the allocator as a huge unsigned size. None
 * of that should be reachable from a public constructor.
 *
 * Returns the shape buffer padded to length 1, because ptr() cannot address a
 * zero-length buffer and a 0-d array is legal: product([]) === 1.
 */
function shapeBuffer(shape: number[], n: number, what: string): Int32Array {
  let product = 1;
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0) {
      throw new Error(`${what}: shape [${shape}] has a non-integer or negative dimension`);
    }
    product *= d;
    if (!Number.isSafeInteger(product)) {
      throw new Error(`${what}: shape [${shape}] overflows a safe integer`);
    }
  }
  if (product !== n) {
    throw new Error(`${what}: shape [${shape}] needs ${product} elements, got ${n}`);
  }
  const buf = new Int32Array(Math.max(1, shape.length));
  buf.set(shape);
  return buf;
}

export function fromF32(data: Float32Array, shape: number[]): MX {
  const sh = shapeBuffer(shape, data.length, "fromF32");
  return new MX(m.mlx_array_new_data(ptr(data), ptr(sh), shape.length, FLOAT32) as number);
}
export function fromI32(data: Int32Array, shape: number[]): MX {
  const sh = shapeBuffer(shape, data.length, "fromI32");
  return new MX(m.mlx_array_new_data(ptr(data), ptr(sh), shape.length, INT32) as number);
}
// uint32 — e.g. packed quantized weights and gather indices (bit pattern matters,
// so these can't round-trip through float32).
export function fromU32(data: Uint32Array, shape: number[]): MX {
  const sh = shapeBuffer(shape, data.length, "fromU32");
  return new MX(m.mlx_array_new_data(ptr(data), ptr(sh), shape.length, UINT32) as number);
}
export const scalar = (x: number): MX => new MX(m.mlx_array_new_float(x) as number);
// GELU constant singletons — allocated once at module load (ARENA is null here, so
// they're never captured/freed by a tidy() scope). Used by MX.prototype.gelu().
const GH = scalar(0.5),
  G1 = scalar(1),
  GR = scalar(Math.SQRT2);
// stack a list of arrays along a new axis (e.g. per-expert weights -> [E, ...])
export function stack(arrs: MX[], axis: number): MX {
  const buf = new BigUint64Array(arrs.map((a) => BigInt(a.h)));
  const vh = m.mlx_vector_array_new_data(ptr(buf), BigInt(arrs.length)) as number;
  const s = new BigUint64Array(1);
  s[0] = BigInt((m.mlx_array_new() as number) ?? 0);
  m.mlx_stack_axis(ptr(s), vh, axis, stream);
  m.mlx_vector_array_free(vh);
  return mk("mlx_stack_axis", s);
}

// batched eval: force everything before reading (keeps the lazy graph from
// growing). One mlx_eval over a vector, not N per-array evals — each per-array
// eval is a separate GPU sync, so batching is ~Nx faster for multi-output steps
// (measured: 20-output take 1.84 -> 0.14 ms). Used on the decode hot path
// (token + all KV-cache arrays).
export function evalAll(...xs: MX[]) {
  if (xs.length === 0) return;
  if (xs.length === 1) {
    m.mlx_array_eval(xs[0].h);
    return;
  }
  const buf = new BigUint64Array(xs.map((x) => BigInt(x.h)));
  const vh = m.mlx_vector_array_new_data(ptr(buf), BigInt(xs.length)) as number;
  m.mlx_eval(vh);
  m.mlx_vector_array_free(vh);
}
// async eval: queue the graph on the stream and return immediately (no host sync),
// so the next step's graph can be built while the GPU runs this one.
export function asyncEval(...xs: MX[]) {
  const buf = new BigUint64Array(xs.map((x) => BigInt(x.h)));
  const vh = m.mlx_vector_array_new_data(ptr(buf), BigInt(xs.length)) as number;
  m.mlx_async_eval(vh);
  m.mlx_vector_array_free(vh);
}
export function seed(s: number) {
  if (m.mlx_random_seed) m.mlx_random_seed(BigInt(s));
}

/**
 * Standard-normal noise of the given shape — where a diffusion sample starts.
 * With `seedNum` the draw comes from an explicit key, so a prompt reproduces;
 * without one it advances MLX's global RNG like any other random op.
 */
export function randomNormal(shape: number[], seedNum?: number): MX {
  const r = slot();
  let key = 0;
  if (seedNum !== undefined) {
    const ks = slot();
    m.mlx_random_key(ptr(ks), BigInt(seedNum));
    key = Number(ks[0]);
  }
  m.mlx_random_normal(ptr(r), ptr(new Int32Array(shape)), BigInt(shape.length), FLOAT32, 0, 1, key, stream);
  if (key) m.mlx_array_free(key);
  return new MX(Number(r[0]));
}

// Write a {name -> MX} record to a .safetensors file (the save side of loader.ts).
// Keys become tensor names; pass a flattened param tree (e.g. "blocks.0.wq").
// Used for training checkpoints (pretrain -> SFT/inference handoff).
export function saveSafetensors(path: string, record: Record<string, MX>) {
  mkdirSync(dirname(path), { recursive: true }); // checkpoints/ etc. are gitignored
  const vals = Object.values(record);
  evalAll(...vals); // materialize before writing
  const map = m.mlx_map_string_to_array_new() as number;
  for (const [k, v] of Object.entries(record)) {
    const key = new Uint8Array([...new TextEncoder().encode(k), 0]); // keep alive across the call
    m.mlx_map_string_to_array_insert(map, ptr(key), v.h);
  }
  const meta = m.mlx_map_string_to_string_new() as number; // empty metadata
  const file = new Uint8Array([...new TextEncoder().encode(path), 0]);
  m.mlx_save_safetensors(ptr(file), map, meta);
  m.mlx_map_string_to_array_free?.(map);
}
const memOf = (fn: any): number => {
  const o = new BigUint64Array(1);
  fn(ptr(o));
  return Number(o[0]) / 1e6;
};
export const activeMemoryMB = (): number => memOf(m.mlx_get_active_memory);
export const peakMemoryMB = (): number => memOf(m.mlx_get_peak_memory);
export const cacheMemoryMB = (): number => memOf(m.mlx_get_cache_memory);
export const clearCache = (): void => {
  m.mlx_clear_cache();
}; // return reuse-cache buffers to the OS
export const resetPeakMemory = (): void => {
  m.mlx_reset_peak_memory();
};
/**
 * Cap the buffer-reuse cache. MLX keeps freed Metal buffers around to hand
 * back to the next allocation, and by default that pool has no ceiling: a
 * MusicGen-medium generation grew it to 20 GB on top of 8 GB of weights, for
 * 28 GB resident. This is the only knob that bounds it — setMemoryLimit()
 * governs live allocations, which were never the problem.
 *
 * Returns the previous limit, so a caller can put it back.
 */
export const setCacheLimit = (mb: number): number => {
  const o = new BigUint64Array(1);
  m.mlx_set_cache_limit(ptr(o), BigInt(Math.round(mb * 1e6)));
  return Number(o[0]) / 1e6;
};
export const setMemoryLimit = (mb: number): void => {
  const o = new BigUint64Array(1);
  m.mlx_set_memory_limit(ptr(o), BigInt(Math.round(mb * 1e6)));
};
export const setWiredLimit = (mb: number): void => {
  const o = new BigUint64Array(1);
  if (m.mlx_set_wired_limit) m.mlx_set_wired_limit(ptr(o), BigInt(Math.round(mb * 1e6)));
};

function categorical(logits: MX, axis: number): MX {
  const s = slot();
  m.mlx_random_categorical(ptr(s), logits.h, axis, 0, stream);
  return mk("mlx_random_categorical", s);
}

// Inverted dropout (device-side): keep each element w.p. (1-p) via a Bernoulli
// mask, scale survivors by 1/(1-p). `seedNum` derives the RNG key — pass a value
// that varies per call (e.g. step*K + site) so masks differ across steps but are
// reproducible (MLX's bernoulli is deterministic given key + shape, so a Python
// mirror using the same seeds gets identical masks). No-op at p<=0 (eval/inference).
export function dropout(x: MX, p: number, seedNum: number): MX {
  if (p <= 0) return x;
  const keep = 1 - p;
  const ks = slot();
  m.mlx_random_key(ptr(ks), BigInt(seedNum));
  const key = new MX(Number(ks[0]));
  const sh = x.shape;
  const s = slot();
  m.mlx_random_bernoulli(ptr(s), scalar(keep).h, ptr(new Int32Array(sh)), BigInt(sh.length), key.h, stream);
  return x.mul(new MX(Number(s[0])).astype(FLOAT32)).divScalar(keep);
}

// top-k: keep the k highest logits along `ax`, mask the rest to ~-inf.
function topKFilter(logits: MX, k: number, ax: number): MX {
  const vocab = logits.shape[ax];
  if (k <= 0 || k >= vocab) return logits;
  const sortedAsc = logits.takeAlong(logits.argsort(ax), ax); // values ascending
  const thr = sortedAsc.takeAxis(fromI32(Int32Array.from([vocab - k]), [1]), ax); // k-th largest -> [B,1]
  return thr.greater(logits).where(scalar(-1e9), logits); // drop logits below it
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
  const B = logits.shape[0],
    ax = 1; // logits is [B, vocab]
  if (temp === 0) return logits.argmax(ax); // greedy -> [B]
  let scaled = logits.divScalar(temp);
  if (topK > 0) scaled = topKFilter(scaled, topK, ax);
  if (topP > 0 && topP < 1) {
    const probs = scaled.softmax(ax);
    const idx = probs.argsort(ax); // ascending order
    const sp = probs.takeAlong(idx, ax); // probs in ascending order
    const keep = sp.cumsum(ax).greater(scalar(1 - topP)); // keep the top-p mass (high end)
    const masked = keep.where(sp, scalar(0));
    const sortedTok = categorical(masked.log(), ax).reshape([B, 1]); // index into sorted
    return idx.takeAlong(sortedTok, ax).reshape([B]); // map back to vocab id
  }
  return categorical(scaled, ax);
}
