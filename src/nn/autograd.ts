// Tree-based value_and_grad (cf. mlx.nn.value_and_grad): differentiate a scalar
// loss w.r.t. a pytree of parameters. Flattens the tree into the vector the
// mlx-c autograd closure needs and unflattens the gradients back into the same
// tree shape — so callers never thread parameters by hand.

import { MX, tidy } from "../core/mx.ts";
import { type Tree, treeFlatten, treeUnflattenLike } from "../core/pytree.ts";
import { m, throwIfError } from "../ffi/generated.ts";
import { callback, open, ptr } from "../ffi/index.ts";

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

// Gradient checkpointing (cf. mlx.core.checkpoint). Inside a valueAndGrad, the
// backward pass needs the intermediates of every op the loss went through, so
// by default all of them stay alive until the gradients are computed — for a
// model that is every layer's activations at once, and on a long input they
// outweigh the weights many times over. A checkpointed function keeps only its
// inputs: when the gradient reaches it, it runs `fn` again from those inputs
// and differentiates that. Memory for a stack of checkpointed layers is one
// layer's intermediates plus each layer's input; the price is running each
// layer's forward twice.
//
// Anything a gradient must flow to has to be an input: arrays `fn` closes over
// are treated as constants. Frozen weights can be captured; trainable ones
// (a LoRA adapter, say) must be passed in.
export function checkpoint(fn: (...inputs: MX[]) => MX): (...inputs: MX[]) => MX {
  const cb = callback({ args: ["ptr", "ptr"], returns: "i32" }, (outPtr: number, inH: number) => {
    const inputs = Array.from({ length: vsize(inH) }, (_, i) => new MX(vget(inH, i)));
    // fn's intermediates get JS handles, and a handle keeps its array alive.
    // Left in the caller's tidy() scope they would live until the whole step
    // ends — every layer's activations, the exact thing checkpointing exists to
    // drop — and be recomputed on top. Free them as soon as fn returns; MLX
    // holds its own references to whatever its graph still needs.
    const out = tidy(() => fn(...inputs));
    m.mlx_vector_array_set_value(outPtr, out.h);
    out.free();
    for (const x of inputs) x.free();
    return 0;
  });
  keepAlive.push(cb);

  const inner = clib.mlx_closure_new_func(cb.addr) as number;
  const outS = new BigUint64Array(1); outS[0] = BigInt((m.mlx_closure_new() as number) ?? 0);
  m.mlx_checkpoint(ptr(outS), inner);
  throwIfError("mlx_checkpoint");
  const checkpointed = Number(outS[0]);

  return (...inputs: MX[]): MX => {
    const vin = vecOf(inputs.map((x) => x.h));
    const resS = new BigUint64Array(1); resS[0] = BigInt((m.mlx_vector_array_new() as number) ?? 0);
    m.mlx_closure_apply(ptr(resS), checkpointed, vin);
    // `fn` runs inside this call (and again inside the backward pass).
    throwIfError("mlx_closure_apply");
    const out = new MX(vget(Number(resS[0]), 0));
    m.mlx_vector_array_free(vin); m.mlx_vector_array_free(Number(resS[0]));
    return out;
  };
}

// Vector-Jacobian product (cf. mlx.core.vjp): run `fn` at `primals` and carry a
// gradient for its output back to a gradient for each input. valueAndGrad does
// this for a whole loss at once; vjp does it for one piece, so a backward pass
// can be taken one layer at a time — each layer's gradient computed, evaluated
// and its weights released before the next layer back is touched. That is what
// training a model whose weights do not fit in memory needs.
//
// Returns a reusable function: build it once, call it per layer and example.
// Gradients flow to every input, so pass only what should be differentiated;
// anything `fn` reads from outside (frozen weights, a target) is a constant.
export function vjpOf(fn: (...inputs: MX[]) => MX): (primals: MX[], cotangent: MX) => { output: MX; grads: MX[] } {
  const cb = callback({ args: ["ptr", "ptr"], returns: "i32" }, (outPtr: number, inH: number) => {
    const inputs = Array.from({ length: vsize(inH) }, (_, i) => new MX(vget(inH, i)));
    // As in checkpoint(): intermediates must not outlive this call.
    const out = tidy(() => fn(...inputs));
    m.mlx_vector_array_set_value(outPtr, out.h);
    out.free();
    for (const x of inputs) x.free();
    return 0;
  });
  keepAlive.push(cb);
  const closure = clib.mlx_closure_new_func(cb.addr) as number;

  return (primals: MX[], cotangent: MX) => {
    const vp = vecOf(primals.map((x) => x.h)), vc = vecOf([cotangent.h]);
    const outS = new BigUint64Array(1); outS[0] = BigInt((m.mlx_vector_array_new() as number) ?? 0);
    const gradS = new BigUint64Array(1); gradS[0] = BigInt((m.mlx_vector_array_new() as number) ?? 0);
    m.mlx_vjp(ptr(outS), ptr(gradS), closure, vp, vc);
    throwIfError("mlx_vjp");
    const output = new MX(vget(Number(outS[0]), 0));
    const grads = primals.map((_, i) => new MX(vget(Number(gradS[0]), i)));
    m.mlx_vector_array_free(vp); m.mlx_vector_array_free(vc);
    m.mlx_vector_array_free(Number(outS[0])); m.mlx_vector_array_free(Number(gradS[0]));
    return { output, grads };
  };
}

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
    // The loss closure runs inside this call, so a failure in the user's forward
    // pass is reported here. Consume it rather than leaving it for the next op.
    throwIfError("mlx_closure_value_and_grad_apply");
    const lossVal = new MX(vget(Number(lossV[0]), 0)).itemF();
    const grads = treeUnflattenLike(params, Array.from({ length: nP }, (_, i) => new MX(vget(Number(gradV[0]), i))));
    m.mlx_vector_array_free(inputs); m.mlx_vector_array_free(Number(lossV[0])); m.mlx_vector_array_free(Number(gradV[0]));
    return { loss: lossVal, grads };
  };
}
