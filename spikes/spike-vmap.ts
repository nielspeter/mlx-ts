// SPIKE: recover vmap over FFI from the mlx_detail_vmap_* primitives — the same
// pair MLX's own public vmap composes (trace once, then replace with the real
// batched inputs). The per-op batching rules live in the C++ core and run inside
// vmap_replace; we only orchestrate. Closure plumbing mirrors train.ts.
//   bun spike-vmap.ts

import { fromF32, MX } from "../src/core/mx.ts";
import { m } from "../src/ffi/generated.ts";
import { callback, open, ptr } from "../src/ffi/index.ts";
import { LIBMLXC } from "../src/ffi/native-lib.ts";

// detail funcs are internal -> not in the generated table; dlopen by hand.
const lib = open(LIBMLXC, {
  mlx_closure_new_func: { args: ["ptr"], returns: "ptr" },
  mlx_detail_vmap_trace: { args: ["ptr", "ptr", "ptr", "ptr", "ptr", "u64"], returns: "i32" },
  mlx_detail_vmap_replace: { args: ["ptr", "ptr", "ptr", "ptr", "ptr", "u64", "ptr", "u64"], returns: "i32" },
});

const arrSlot = () => { const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_array_new() as number) ?? 0); return s; };
const vecSlot = () => { const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_vector_array_new() as number) ?? 0); return s; };
const vget = (vec: number, i: number) => { const s = arrSlot(); m.mlx_vector_array_get(ptr(s), vec, BigInt(i)); return Number(s[0]); };
const vsize = (vec: number) => Number(m.mlx_vector_array_size(vec));
const vecOf = (hs: number[]) => m.mlx_vector_array_new_data(ptr(new BigUint64Array(hs.map((h) => BigInt(h)))), BigInt(hs.length)) as number;
const keepAlive: unknown[] = [];

// vmap(fn, inAxes, outAxes): fn maps single examples -> outputs; returns a
// function over the real batched inputs.
function vmap(fn: (...xs: MX[]) => MX | MX[], inAxes: number[], outAxes: number[]) {
  const cb = callback({ args: ["ptr", "ptr"], returns: "i32" }, (resPtr: number, inVec: number) => {
    const inputs = Array.from({ length: vsize(inVec) }, (_, i) => new MX(vget(inVec, i)));
    const out = fn(...inputs);
    const outs = Array.isArray(out) ? out : [out];
    m.mlx_vector_array_set(resPtr, vecOf(outs.map((o) => o.h)));
    return 0;
  });
  keepAlive.push(cb);
  const closure = lib.mlx_closure_new_func(cb.addr) as number;
  const inAx = new Int32Array(inAxes), outAx = new Int32Array(outAxes);

  return (...inputs: MX[]): MX[] => {
    const inVec = vecOf(inputs.map((x) => x.h));
    const sIn = vecSlot(), sOut = vecSlot();
    lib.mlx_detail_vmap_trace(ptr(sIn), ptr(sOut), closure, inVec, ptr(inAx), BigInt(inAxes.length));
    const res = vecSlot();
    lib.mlx_detail_vmap_replace(ptr(res), inVec, Number(sIn[0]), Number(sOut[0]), ptr(inAx), BigInt(inAxes.length), ptr(outAx), BigInt(outAxes.length));
    return Array.from({ length: vsize(Number(res[0])) }, (_, i) => new MX(vget(Number(res[0]), i)));
  };
}

const approx = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) < 1e-4);

// --- test 1: vmap a single-input fn over axis 0 ---
// f(x:[N]) = sum(x*x) -> scalar ; vmap over X[B,N] -> [B]
const B = 4, N = 5;
const X = fromF32(Float32Array.from({ length: B * N }, (_, i) => (i % 7) - 3), [B, N]); X.eval();
const f = (x: MX) => x.mul(x).sumAxes([0], false);
const v1 = vmap(f, [0], [0])(X)[0];
const manual1 = X.mul(X).sumAxes([1], false);
console.log("test1 vmap(sumsq):", v1.copy().toF32().map((x) => +x.toFixed(2)));
console.log("test1 manual:    ", manual1.copy().toF32().map((x) => +x.toFixed(2)));
const ok1 = approx([...v1.copy().toF32()], [...manual1.copy().toF32()]);

// --- test 2: a non-mapped (shared) input. f(A, x) = A @ x, A shared, x batched ---
// A[M,N] shared (in_axis -1), x[B,N] batched over 0 -> [B,M]
const M = 3;
const A = fromF32(Float32Array.from({ length: M * N }, (_, i) => ((i * 3) % 5) - 2), [M, N]); A.eval();
const g = (a: MX, x: MX) => a.matmul(x.reshape([N, 1])).reshape([M]); // A@x per example
const v2 = vmap(g, [-1, 0], [0])(A, X)[0];          // -1 = not mapped
const manual2 = X.matmul(A.transpose([1, 0]));      // [B,N]@[N,M] = [B,M]
console.log("test2 vmap(A@x) [0]:", v2.copy().toF32().slice(0, M).map((x) => +x.toFixed(2)));
console.log("test2 manual    [0]:", manual2.copy().toF32().slice(0, M).map((x) => +x.toFixed(2)));
const ok2 = approx([...v2.copy().toF32()], [...manual2.copy().toF32()]);

// --- test 3: per-sample gradients = vmap(grad(loss)) — the use case vmap unlocks ---
// single-example grad helper via mlx value_and_grad (same machinery as train.ts)
function gradArr(f: (x: MX) => MX): (x: MX) => MX {
  const cb = callback({ args: ["ptr", "ptr"], returns: "i32" }, (outPtr: number, inH: number) => {
    const x = new MX(vget(inH, 0));
    m.mlx_vector_array_set_value(outPtr, f(x).h);
    return 0;
  });
  keepAlive.push(cb);
  const closure = lib.mlx_closure_new_func(cb.addr) as number;
  const vag = vecSlot(); vag[0] = BigInt((m.mlx_closure_value_and_grad_new() as number) ?? 0);
  m.mlx_value_and_grad(ptr(vag), closure, ptr(new Int32Array([0])), 1n);
  return (x: MX) => {
    const lossV = vecSlot(), gradV = vecSlot();
    m.mlx_closure_value_and_grad_apply(ptr(lossV), ptr(gradV), Number(vag[0]), vecOf([x.h]));
    return new MX(vget(Number(gradV[0]), 0)); // dL/dx for this single example
  };
}
const loss = (x: MX) => x.mul(x).sumAxes([0], false); // per-example sum(x*x), grad = 2x
const perSample = vmap(gradArr(loss), [0], [0])(X)[0]; // [B, N]
const manual3 = X.mul(scalar2(2));                      // 2*X
console.log("test3 vmap(grad) row0:", perSample.copy().toF32().slice(0, N).map((x) => +x.toFixed(2)));
console.log("test3 manual 2x  row0:", manual3.copy().toF32().slice(0, N).map((x) => +x.toFixed(2)));
const ok3 = approx([...perSample.copy().toF32()], [...manual3.copy().toF32()]);

console.log(`test1 (single-input vmap): ${ok1}`);
console.log(`test2 (shared + batched, in_axis -1): ${ok2}`);
console.log(`test3 (per-sample gradients, vmap∘grad): ${ok3}`);
console.log(ok1 && ok2 && ok3 ? "VMAP OK" : "VMAP MISMATCH");

function scalar2(v: number): MX { return fromF32(Float32Array.from([v]), [1]); }
