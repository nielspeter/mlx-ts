// SPIKE: is TRAINING feasible over the FFI stack? Proves the keystone — a real
// SGD loop where the loss is a JS closure, differentiated w.r.t. a multi-param
// vector via mlx_value_and_grad, with the optimizer step in TS. Linear regression
// fit to a known target; loss must fall and final W must match MLX Python.
//   bun spike-train.ts   (then: python3 reference-train.py)

import { open, callback, ptr } from "./src/ffi/index.ts";
import { MX, fromF32, scalar } from "./mx.ts";
import { m } from "./src/ffi/generated.ts";
import { LIBMLXC } from "./src/ffi/native-lib.ts";

// mlx_closure_new_func takes a C function pointer -> not in the generated table; dlopen it.
const clib = open(LIBMLXC, {
  mlx_closure_new_func: { args: ["ptr"], returns: "ptr" },
});

const N = 16, D = 4, LR = 0.1, STEPS = 50;
const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

// data: Y = X @ Wtrue + 0.5  (exactly linear -> loss should go ~0)
const X = fromF32(det(N * D, 1), [N, D]);
const Wtrue = fromF32(det(D, 2), [D, 1]);
const Y = X.matmul(Wtrue).add(scalar(0.5));
[X, Y].forEach((a) => a.eval());

const nb = () => { const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_array_new() as number) ?? 0); return s; };
const nv = () => { const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_vector_array_new() as number) ?? 0); return s; };
const getArr = (vec: number, i: number) => { const s = nb(); m.mlx_vector_array_get(ptr(s), vec, BigInt(i)); return Number(s[0]); };
const vecOf = (hs: number[]) => m.mlx_vector_array_new_data(ptr(new BigUint64Array(hs.map((h) => BigInt(h)))), BigInt(hs.length)) as number;

// loss(params, data) as a JS closure: inputs = [W, b, X, Y]; returns [mse]
const lossCb = callback({ args: ["ptr", "ptr"], returns: "i32" }, (outPtr: number, inH: number) => {
  const get = (i: number) => new MX(getArr(inH, i));
  const W = get(0), b = get(1), Xi = get(2), Yi = get(3);
  const diff = Xi.matmul(W).add(b).sub(Yi);          // pred - Y
  const loss = diff.mul(diff).sumAxes([0, 1], false).div(scalar(N)); // mean squared error
  m.mlx_vector_array_set_value(outPtr, loss.h);
  return 0;
});

// value_and_grad of the loss w.r.t. inputs 0,1 (W, b); X, Y are constants
const closure = clib.mlx_closure_new_func(lossCb.addr) as number;
const vagS = new BigUint64Array(1); vagS[0] = BigInt((m.mlx_closure_value_and_grad_new() as number) ?? 0);
m.mlx_value_and_grad(ptr(vagS), closure, ptr(new Int32Array([0, 1])), 2n);
const vag = Number(vagS[0]);

// SGD loop
let W = fromF32(new Float32Array(D * 1), [D, 1]);   // zeros
let b = fromF32(new Float32Array(1), [1]);
console.log("=== training spike: linear regression over FFI (value_and_grad + SGD) ===");
let loss = 0;
for (let step = 0; step < STEPS; step++) {
  const inputs = vecOf([W.h, b.h, X.h, Y.h]);
  const lossV = nv(), gradV = nv();
  m.mlx_closure_value_and_grad_apply(ptr(lossV), ptr(gradV), vag, inputs);
  loss = new MX(getArr(Number(lossV[0]), 0)).itemF();
  const gW = new MX(getArr(Number(gradV[0]), 0)), gb = new MX(getArr(Number(gradV[0]), 1));
  W = W.sub(gW.mul(scalar(LR))); b = b.sub(gb.mul(scalar(LR)));
  W.eval(); b.eval();
  m.mlx_vector_array_free(inputs); m.mlx_vector_array_free(Number(lossV[0])); m.mlx_vector_array_free(Number(gradV[0]));
  if (step % 10 === 0 || step === STEPS - 1) console.log(`  step ${String(step).padStart(2)}: loss ${loss.toFixed(6)}`);
}
console.log(`final loss: ${loss.toFixed(6)}`);
console.log(`W: [${W.toF32().map((v) => v.toFixed(4)).join(", ")}]`);
lossCb.close();
