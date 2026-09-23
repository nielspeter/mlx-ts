// Gradient checkpointing must change memory, never the result. A stack of
// adapter-carrying layers — the shape LoRA training has — is differentiated
// with and without checkpoint(), and loss and every gradient must agree.
//   bun test tests/checkpoint.test.ts
import { expect, test } from "bun:test";
import { checkpoint, evalAll, fromF32, MX, valueAndGrad, vjpOf } from "../src/index.ts";
import { type Tree, treeFlatten } from "../src/core/pytree.ts";

const D = 16, R = 4, LAYERS = 4, L = 7;
const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => (((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.4);

// Frozen weights: captured, as a real model's are.
const frozen = Array.from({ length: LAYERS }, (_, i) => fromF32(det(D * D, 50 + i), [D, D]));
const x0 = fromF32(det(L * D, 1), [L, D]);
const params: Tree = {
  layers: Array.from({ length: LAYERS }, (_, i) => ({
    A: fromF32(det(D * R, 10 + i), [D, R]),
    B: fromF32(det(R * D, 20 + i), [R, D]),
  })),
};

// One layer: frozen projection plus a low-rank adapter, then a nonlinearity.
const layer = (i: number) => (h: MX, A: MX, B: MX): MX => h.add(h.matmul(frozen[i]).add(h.matmul(A).matmul(B)).silu());
const plain = Array.from({ length: LAYERS }, (_, i) => layer(i));
const checked = Array.from({ length: LAYERS }, (_, i) => checkpoint(layer(i)));

const lossWith = (fs: ((h: MX, A: MX, B: MX) => MX)[]) => (p: any, x: MX): MX => {
  let h = x;
  for (let i = 0; i < LAYERS; i++) h = fs[i](h, p.layers[i].A, p.layers[i].B);
  return h.square().meanAll();
};

test("a checkpointed function gives the same output as the plain one", () => {
  const a = plain[0](x0, (params as any).layers[0].A, (params as any).layers[0].B);
  const b = checked[0](x0, (params as any).layers[0].A, (params as any).layers[0].B);
  evalAll(a, b);
  const av = a.toF32(), bv = b.toF32();
  expect(bv.length).toBe(av.length);
  for (let i = 0; i < av.length; i++) expect(bv[i]).toBeCloseTo(av[i], 6);
});

test("loss and every gradient agree with and without checkpointing", () => {
  const ref = valueAndGrad(params, lossWith(plain))(params, x0);
  const ck = valueAndGrad(params, lossWith(checked))(params, x0);
  expect(ck.loss).toBeCloseTo(ref.loss, 6);
  const gr = treeFlatten(ref.grads), gc = treeFlatten(ck.grads);
  expect(gc.length).toBe(gr.length);
  let largest = 0;
  for (let k = 0; k < gr.length; k++) {
    const a = gr[k].toF32(), b = gc[k].toF32();
    for (let i = 0; i < a.length; i++) largest = Math.max(largest, Math.abs(a[i] - b[i]));
  }
  // Recomputing the same ops on the same inputs is exact on the GPU.
  expect(largest).toBeLessThan(1e-6);
  // And the gradients are not trivially zero.
  expect(Math.max(...gr.map((g) => Math.max(...g.toF32().map(Math.abs))))).toBeGreaterThan(1e-4);
});

test("an error inside a checkpointed function is thrown, not swallowed", () => {
  const bad = checkpoint((h: MX) => h.matmul(fromF32(new Float32Array(9), [3, 3])));
  expect(() => bad(x0)).toThrow();
});

// A backward pass taken one layer at a time — what training with layers read
// from disk does — must give the gradients a single valueAndGrad gives.
test("a layer-by-layer backward pass with vjpOf matches valueAndGrad", () => {
  const ref = valueAndGrad(params, lossWith(plain))(params, x0);
  const p = params as any;

  // Forward, keeping each layer's input.
  const inputs: MX[] = [];
  let h = x0;
  for (let i = 0; i < LAYERS; i++) { inputs.push(h); h = plain[i](h, p.layers[i].A, p.layers[i].B); }

  // The loss's gradient with respect to the last layer's output...
  const head = vjpOf((y: MX) => y.square().meanAll());
  const top = head([h], fromF32(new Float32Array([1]), []));
  expect(top.output.itemF()).toBeCloseTo(ref.loss, 6);

  // ...carried back through each layer in turn.
  const layerVjps = plain.map((f) => vjpOf(f));
  let g = top.grads[0];
  const grads: MX[][] = [];
  for (let i = LAYERS - 1; i >= 0; i--) {
    const r = layerVjps[i]([inputs[i], p.layers[i].A, p.layers[i].B], g);
    grads[i] = r.grads.slice(1);
    g = r.grads[0];
  }

  const gr = treeFlatten(ref.grads);
  const gl = grads.flat();
  let largest = 0;
  for (let k = 0; k < gr.length; k++) {
    const a = gr[k].toF32(), b = gl[k].toF32();
    for (let i = 0; i < a.length; i++) largest = Math.max(largest, Math.abs(a[i] - b[i]));
  }
  expect(largest).toBeLessThan(1e-6);
});
