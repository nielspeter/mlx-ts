// Minimal nn.Module layer over MX — the ergonomic surface a TS SDK would expose.
// Mirrors mlx.nn: modules own their weights (as MX) and have a forward/call.

import { MX, fromI32, scalar } from "./mx.ts";
import type { Tree } from "./pytree.ts";

export abstract class Module {
  abstract forward(...xs: MX[]): MX;
  call(...xs: MX[]): MX { return this.forward(...xs); }

  // Collect this module's parameters as a pytree: MX fields are leaves, child
  // Modules recurse, arrays map. Non-array/MX/Module fields (numbers, etc.) are
  // skipped. Mirrors mlx.nn Module.parameters().
  parameters(): Tree {
    const out: any = {};
    for (const [k, v] of Object.entries(this)) {
      const p = collectParams(v);
      if (p !== undefined) out[k] = p;
    }
    return out;
  }
}

function collectParams(v: any): Tree | undefined {
  if (v instanceof MX) return v;
  if (v instanceof Module) return v.parameters();
  if (Array.isArray(v)) { const a = v.map(collectParams); return a.some((x) => x !== undefined) ? a.map((x) => x ?? {}) : undefined; }
  return undefined;
}

// A rank-r LoRA adapter delta: x -> (x @ A) @ B * scale. A, B are the params.
export class LoraDelta extends Module {
  A: MX; B: MX; scale: number;
  constructor(A: MX, B: MX, scale: number) { super(); this.A = A; this.B = B; this.scale = scale; }
  forward(x: MX): MX { return x.matmul(this.A).matmul(this.B).mul(scalar(this.scale)); }
}

export class RMSNorm extends Module {
  weight: MX; eps: number;
  constructor(weight: MX, eps: number) { super(); this.weight = weight; this.eps = eps; }
  forward(x: MX) { return x.rmsNorm(this.weight, this.eps); }
}

// Linear with weight pre-transposed to [in, out] so forward is plain matmul.
export class Linear extends Module {
  wt: MX; bias?: MX;
  constructor(wt: MX, bias?: MX) { super(); this.wt = wt; this.bias = bias; }
  forward(x: MX) { const y = x.matmul(this.wt); return this.bias ? y.add(this.bias) : y; }
}

// 4-bit (or n-bit) quantized linear: y = x @ dequant(W).T via quantized_matmul.
export class QuantizedLinear extends Module {
  wq: MX; scales: MX; biases: MX; gs: number; bits: number;
  constructor(wq: MX, scales: MX, biases: MX, gs: number, bits: number) { super(); this.wq = wq; this.scales = scales; this.biases = biases; this.gs = gs; this.bits = bits; }
  forward(x: MX) { return x.qmm(this.wq, this.scales, this.biases, this.gs, this.bits); }
}

export class Embedding extends Module {
  weight: MX;
  constructor(weight: MX) { super(); this.weight = weight; }      // [vocab, D]
  forward(ids: MX) { return this.weight.takeAxis(ids, 0); }
  asLinear(x: MX) { return x.matmul(this.weight.transpose([1, 0])); } // tied lm_head
}

// A stack of quantized expert weights ([E, out, in_packed] + scales/biases).
export type Experts = { wq: MX; scales: MX; biases: MX };

// Mixture-of-Experts SwiGLU block (Qwen3-MoE / SwitchGLU style): router picks
// top-K experts per token, each expert is a quantized SwiGLU, outputs combined
// by the (renormalized) router weights. Built on gather_qmm expert dispatch.
export class MoE extends Module {
  // router: a Module mapping [T,D] -> [T,E] (Linear or QuantizedLinear).
  // normTopK: renormalize the top-K routing weights to sum to 1 (Qwen3-MoE);
  // off for models with norm_topk_prob=false (OLMoE).
  router: Module; gate: Experts; up: Experts; down: Experts; K: number; gs: number; bits: number; normTopK: boolean;
  constructor(router: Module, gate: Experts, up: Experts, down: Experts, K: number, gs: number, bits: number, normTopK = true) { super(); this.router = router; this.gate = gate; this.up = up; this.down = down; this.K = K; this.gs = gs; this.bits = bits; this.normTopK = normTopK; }
  forward(x: MX): MX {
    const [T, D] = x.shape;                              // x: [T, D]
    const gates = this.router.forward(x).softmax(1);     // [T, E]
    const cols = fromI32(Int32Array.from({ length: this.K }, (_, i) => i), [this.K]);
    const inds = gates.neg().argpartition(this.K - 1, 1).takeAxis(cols, 1); // [T, K] expert ids (top-K)
    let w = gates.takeAlong(inds, 1);                    // [T, K] routing weights
    if (this.normTopK) w = w.div(w.sumAxes([1], true));  // renormalize (Qwen3-MoE)
    const xb = x.reshape([T, 1, 1, D]);                  // [T,1,1,D] for gather_qmm dispatch
    const g = xb.gatherQmm(this.gate.wq, this.gate.scales, this.gate.biases, inds, this.gs, this.bits); // [T,K,1,I]
    const u = xb.gatherQmm(this.up.wq, this.up.scales, this.up.biases, inds, this.gs, this.bits);
    const h = g.silu().mul(u);
    const o = h.gatherQmm(this.down.wq, this.down.scales, this.down.biases, inds, this.gs, this.bits);  // [T,K,1,D]
    return o.reshape([T, this.K, D]).mul(w.reshape([T, this.K, 1])).sumAxes([1], false);                 // [T, D]
  }
}

// Quantized embedding: gather the quantized rows and dequantize; tied lm_head
// reuses the packed weight via quantized_matmul.
export class QuantizedEmbedding extends Module {
  wq: MX; scales: MX; biases: MX; gs: number; bits: number;
  constructor(wq: MX, scales: MX, biases: MX, gs: number, bits: number) { super(); this.wq = wq; this.scales = scales; this.biases = biases; this.gs = gs; this.bits = bits; }
  forward(ids: MX) {
    const wr = this.wq.takeAxis(ids, 0), sr = this.scales.takeAxis(ids, 0), br = this.biases.takeAxis(ids, 0);
    return MX.dequantize(wr, sr, br, this.gs, this.bits);
  }
  asLinear(x: MX) { return x.qmm(this.wq, this.scales, this.biases, this.gs, this.bits); }
}
