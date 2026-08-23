// LoRA fine-tuning of the REAL 4-bit Qwen3-0.6B — now with ergonomic training:
// adapters live in a Module, parameters() builds the pytree, valueAndGrad +
// Adam.update operate on the tree. No hand-threaded parameter indices.
//   bun lora-train.ts

import { MX, fromI32, fromF32, scalar, evalAll } from "../src/core/mx.ts";
import { Qwen3 } from "../src/models/qwen-nn.ts";
import { loadSafetensors } from "../src/io/loader.ts";
import { Module, LoraDelta } from "../src/nn/nn.ts";
import { Adam } from "../src/nn/optim.ts";
import { crossEntropy } from "../src/nn/loss.ts";
import { valueAndGrad } from "../src/nn/autograd.ts";
import { treeFlatten, type Tree } from "../src/core/pytree.ts";

const cfg = await Bun.file("models/config-4bit.json").json();
const model = new Qwen3(cfg, loadSafetensors("models/model-q4.safetensors"));
const { D, NL, nH, nKV, Dh, theta, scale, vocab } = model;
const qDim = nH * Dh, kvDim = nKV * Dh;

const R = 8, ALPHA = 16, LSCALE = ALPHA / R, LR = 1e-3, STEPS = 60;
const SEQ = [785, 6722, 315, 9625, 374, 12095, 13, 576]; // "The capital of France is Paris. The"
const L = SEQ.length;
const idsMX = fromI32(Int32Array.from(SEQ), [L]);
const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => (((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.02);

// adapters on every layer's q_proj and v_proj, held in a Module.
// init: A small deterministic, B = 0 (so the initial delta is 0). Frozen 4-bit base.
class LoraSet extends Module {
  q = Array.from({ length: NL }, (_, i) => new LoraDelta(fromF32(det(D * R, i * 4 + 1), [D, R]), fromF32(new Float32Array(R * qDim), [R, qDim]), LSCALE));
  v = Array.from({ length: NL }, (_, i) => new LoraDelta(fromF32(det(D * R, i * 4 + 3), [D, R]), fromF32(new Float32Array(R * kvDim), [R, kvDim]), LSCALE));
  forward(): MX { throw new Error("LoraSet is trained functionally via valueAndGrad"); }
}
let params: Tree = new LoraSet().parameters();   // { q: [{A,B}, …], v: [{A,B}, …] }

// functional forward: frozen 4-bit base + LoRA deltas from the params tree
function forward(p: any, ids: MX): MX {
  let h = model.embed.forward(ids.reshape([1, L]));
  const delta = (x: MX, A: MX, B: MX) => x.matmul(A).matmul(B).mul(scalar(LSCALE));
  for (let i = 0; i < NL; i++) {
    const W = model.layers[i];
    const y = W.inNorm.forward(h);
    let q = W.q.forward(y).add(delta(y, p.q[i].A, p.q[i].B));
    let k = W.k.forward(y);
    let v = W.v.forward(y).add(delta(y, p.v[i].A, p.v[i].B));
    q = W.qNorm.forward(q.reshape([1, L, nH, Dh])).transpose([0, 2, 1, 3]);
    k = W.kNorm.forward(k.reshape([1, L, nKV, Dh])).transpose([0, 2, 1, 3]);
    v = v.reshape([1, L, nKV, Dh]).transpose([0, 2, 1, 3]);
    q = q.rope(Dh, theta, 0); k = k.rope(Dh, theta, 0);
    const o = MX.sdpa(q, k, v, scale, true).transpose([0, 2, 1, 3]).reshape([1, L, qDim]);
    h = h.add(W.o.forward(o));
    const y2 = W.postNorm.forward(h);
    h = h.add(W.down.forward(W.gate.forward(y2).silu().mul(W.up.forward(y2))));
  }
  return model.embed.asLinear(model.finalNorm.forward(h).reshape([L, D]));  // tied lm_head -> [L, V]
}

const predIdx = fromI32(Int32Array.from({ length: L - 1 }, (_, i) => i), [L - 1]);
const tgtIdx = fromI32(Int32Array.from({ length: L - 1 }, (_, i) => i + 1), [L - 1]);
const lossFn = (p: Tree, ids: MX): MX => {
  const logits = forward(p, ids);                                   // [L, V]
  const targets = ids.reshape([L]).takeAxis(tgtIdx, 0).reshape([L - 1, 1]);
  return crossEntropy(logits.takeAxis(predIdx, 0), targets);
};

const vg = valueAndGrad(params, lossFn);
const opt = new Adam(LR);
console.log(`=== LoRA fine-tune: Qwen3-0.6B-4bit, rank ${R} — pytree params, no hand-threading ===`);
let loss = 0;
for (let step = 0; step < STEPS; step++) {
  const r = vg(params, idsMX);
  loss = r.loss;
  params = opt.update(params, r.grads);
  evalAll(...treeFlatten(params));
  if (step % 5 === 0 || step === STEPS - 1) console.log(`  step ${String(step).padStart(2)}: loss ${loss.toFixed(6)}`);
}
console.log(`final loss: ${loss.toFixed(6)}`);
