// Training in ~30 lines: value_and_grad over a pytree of parameters, Adam, and
// cross-entropy — the same machinery training/ uses, on a toy problem that
// needs no data files.
//   bun examples/train.ts
import {
  valueAndGrad, Adam, crossEntropy, fromF32, fromU32, tidy,
  type MX, type Tree,
} from "../src/index.ts";

const N = 256, D = 4, C = 3, STEPS = 60;

// Synthetic 3-class problem: labels come from a fixed linear rule, so a linear
// model can actually reach it and the loss curve means something.
const rnd = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => (((i * 7919 + seed * 104729) % 2003) / 2003) - 0.5);
const xData = rnd(N * D, 1);
const trueW = rnd(D * C, 2);
const labels = new Uint32Array(N);
for (let i = 0; i < N; i++) {
  let best = 0, bestVal = -Infinity;
  for (let c = 0; c < C; c++) {
    let acc = 0;
    for (let d = 0; d < D; d++) acc += xData[i * D + d] * trueW[d * C + c];
    if (acc > bestVal) { bestVal = acc; best = c; }
  }
  labels[i] = best;
}

const X = fromF32(xData, [N, D]);
const Y = fromU32(labels, [N, 1]);           // class indices, one per row

// Parameters are a plain object tree of MX — valueAndGrad flattens it for the
// autograd closure and hands the gradients back in the same shape.
let params: Tree = { w: fromF32(rnd(D * C, 3), [D, C]), b: fromF32(new Float32Array(C), [1, C]) };

const forward = (p: Tree, x: MX) => {
  const { w, b } = p as { w: MX; b: MX };
  return x.matmul(w).add(b);
};
const lossFn = (p: Tree, x: MX, y: MX) => crossEntropy(forward(p, x), y);

const step = valueAndGrad(params, lossFn);
const opt = new Adam(0.1);

for (let i = 0; i <= STEPS; i++) {
  // tidy returns the new params, so they survive while every intermediate and
  // gradient from this step is freed.
  const { loss, next } = tidy(() => {
    const { loss, grads } = step(params, X, Y);
    return { loss, next: opt.update(params, grads) };
  });
  params = next;
  if (i % 15 === 0) console.log(`step ${String(i).padStart(3)}  loss ${loss.toFixed(4)}`);
}
