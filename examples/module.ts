// Building a model: compose nn Modules the way src/models/ does, with weights
// you make up rather than download. No model files needed.
//   bun examples/module.ts
import { activeMemoryMB, fromF32, Linear, Module, MX, RMSNorm, tidy } from "../src/index.ts";

// Deterministic stand-in for real weights, so the output is reproducible.
const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);
const weight = (rows: number, cols: number, seed: number) => fromF32(det(rows * cols, seed), [rows, cols]);
const ones = (n: number) => fromF32(new Float32Array(n).fill(1), [n]);

// A SwiGLU-ish MLP block — the same shape qwen.ts uses, minus the attention.
// Extend Module and implement forward(); `call()` is the public entry point.
class MLP extends Module {
  norm: RMSNorm;
  gate: Linear;
  up: Linear;
  down: Linear;

  constructor(d: number, hidden: number) {
    super();
    this.norm = new RMSNorm(ones(d), 1e-5);
    // Linear holds an already-transposed weight: forward is x @ wt, so [in, out].
    this.gate = new Linear(weight(d, hidden, 1));
    this.up = new Linear(weight(d, hidden, 2));
    this.down = new Linear(weight(hidden, d, 3));
  }

  forward(x: MX): MX {
    const h = this.norm.forward(x);
    return this.down.forward(this.gate.forward(h).silu().mul(this.up.forward(h)));
  }
}

const D = 64, HIDDEN = 128, T = 4;
const model = new MLP(D, HIDDEN);
const x = fromF32(det(T * D, 7), [T, D]);

// tidy() around the forward pass: every intermediate is freed, the result kept.
const y = tidy(() => model.call(x));

console.log(`in  ${JSON.stringify(x.shape)} -> out ${JSON.stringify(y.shape)}`);
const out = y.toF32();
console.log("first row:", Array.from(out.slice(0, 6)).map((v) => +v.toFixed(4)));
console.log("sum:", out.reduce((s, v) => s + v, 0).toFixed(4));
console.log(`active memory: ${activeMemoryMB().toFixed(1)} MB`);
