// Tree-based value_and_grad (cf. mlx.nn.value_and_grad): differentiate a scalar
// loss w.r.t. a pytree of parameters. Flattens the tree into the vector the
// mlx-c autograd closure needs and unflattens the gradients back into the same
// tree shape — so callers never thread parameters by hand.

import { open, callback, ptr } from "../ffi/index.ts";
import { MX } from "../core/mx.ts";
import { m } from "../ffi/generated.ts";
import { treeFlatten, treeUnflattenLike, type Tree } from "../core/pytree.ts";

// mlx_closure_new_func takes a C function pointer -> not in the generated table.
import { LIBMLXC } from "../ffi/native-lib.ts";
const clib = open(LIBMLXC, {
  mlx_closure_new_func: { args: ["ptr"], returns: "ptr" },
});

const nb = () => { const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_array_new() as number) ?? 0); return s; };
const vget = (vec: number, i: number) => { const s = nb(); m.mlx_vector_array_get(ptr(s), vec, BigInt(i)); return Number(s[0]); };
const vsize = (vec: number) => Number(m.mlx_vector_array_size(vec));
const vecOf = (hs: number[]) => m.mlx_vector_array_new_data(ptr(new BigUint64Array(hs.map((h) => BigInt(h)))), BigInt(hs.length)) as number;
const keepAlive: unknown[] = [];

// loss: (params, ...extraInputs) -> scalar MX.  returns a function that, given a
// params tree (+ the extra inputs), yields { loss: number, grads: tree }.
export function valueAndGrad(template: Tree, loss: (params: Tree, ...extra: MX[]) => MX) {
  const nP = treeFlatten(template).length;

  const cb = callback({ args: ["ptr", "ptr"], returns: "i32" }, (outPtr: number, inH: number) => {
    const total = vsize(inH);
    const paramLeaves = Array.from({ length: nP }, (_, i) => new MX(vget(inH, i)));
    const extras = Array.from({ length: total - nP }, (_, j) => new MX(vget(inH, nP + j)));
    const out = loss(treeUnflattenLike(template, paramLeaves), ...extras);
    m.mlx_vector_array_set_value(outPtr, out.h);
    return 0;
  });
  keepAlive.push(cb);

  const closure = clib.mlx_closure_new_func(cb.addr) as number;
  const vagS = new BigUint64Array(1); vagS[0] = BigInt((m.mlx_closure_value_and_grad_new() as number) ?? 0);
  m.mlx_value_and_grad(ptr(vagS), closure, ptr(new Int32Array(Array.from({ length: nP }, (_, i) => i))), BigInt(nP));
  const vag = Number(vagS[0]);

  return (params: Tree, ...extras: MX[]): { loss: number; grads: Tree } => {
    const inputs = vecOf([...treeFlatten(params).map((p) => p.h), ...extras.map((e) => e.h)]);
    const lossV = nb(), gradV = nb();
    m.mlx_closure_value_and_grad_apply(ptr(lossV), ptr(gradV), vag, inputs);
    const lossVal = new MX(vget(Number(lossV[0]), 0)).itemF();
    const grads = treeUnflattenLike(params, Array.from({ length: nP }, (_, i) => new MX(vget(Number(gradV[0]), i))));
    m.mlx_vector_array_free(inputs); m.mlx_vector_array_free(Number(lossV[0])); m.mlx_vector_array_free(Number(gradV[0]));
    return { loss: lossVal, grads };
  };
}
