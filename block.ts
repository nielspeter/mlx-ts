// A full Qwen3 decoder block forward pass, in TypeScript, over mlx-c -> Metal.
// Mirrors mlx-lm's qwen3.py: pre-norm, GQA attention with per-head q/k RMSNorm
// (the Qwen3-specific bit), RoPE, causal SDPA, residual, SwiGLU MLP, residual.
//
// Weights are generated deterministically so reference.py can reproduce the
// exact same numbers and we can check numerical parity. Run: bun block.ts

import {
  array, matmul, add, mul, silu, rmsNorm, reshape, transpose, rope,
  sdpaCausal, sumAll, type Arr,
} from "./mlx.ts";

// ---- config (small, for a fast deterministic check) ----
const B = 1, L = 3;
const D = 64;          // hidden size
const nH = 4, nKV = 2; // GQA: 4 query heads, 2 kv heads
const Dh = 16;         // head dim
const I = 128;         // mlp intermediate
const EPS = 1e-6;
const THETA = 1_000_000; // Qwen3 rope theta
const SCALE = Dh ** -0.5;

// Deterministic data, identical integer math on both JS and Python sides.
function det(n: number, seed: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = (((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.1;
  }
  return a;
}
const W = (n: number, shape: number[], seed: number): Arr => array(det(n, seed), shape);

// ---- weights (raw; Linear is matmul(x, Wt) with Wt laid out [in, out]) ----
const inNorm   = W(D, [D], 1);
const Wq       = W(D * nH * Dh, [D, nH * Dh], 2);
const Wk       = W(D * nKV * Dh, [D, nKV * Dh], 3);
const Wv       = W(D * nKV * Dh, [D, nKV * Dh], 4);
const qNorm    = W(Dh, [Dh], 5);
const kNorm    = W(Dh, [Dh], 6);
const Wo       = W(nH * Dh * D, [nH * Dh, D], 7);
const postNorm = W(D, [D], 8);
const Wgate    = W(D * I, [D, I], 9);
const Wup      = W(D * I, [D, I], 10);
const Wdown    = W(I * D, [I, D], 11);

const x = array(det(B * L * D, 100), [B, L, D]);

// ---- forward ----
function block(h: Arr): Arr {
  // attention
  let y = rmsNorm(h, inNorm, EPS);
  let q = matmul(y, Wq);
  let k = matmul(y, Wk);
  let v = matmul(y, Wv);

  q = transpose(rmsNorm(reshape(q, [B, L, nH, Dh]), qNorm, EPS), [0, 2, 1, 3]);
  k = transpose(rmsNorm(reshape(k, [B, L, nKV, Dh]), kNorm, EPS), [0, 2, 1, 3]);
  v = transpose(reshape(v, [B, L, nKV, Dh]), [0, 2, 1, 3]);

  q = rope(q, Dh, THETA);
  k = rope(k, Dh, THETA);

  let o = sdpaCausal(q, k, v, SCALE);          // [B, nH, L, Dh], GQA handled inside
  o = reshape(transpose(o, [0, 2, 1, 3]), [B, L, nH * Dh]);
  o = matmul(o, Wo);
  h = add(h, o);                                // residual

  // SwiGLU MLP
  const y2 = rmsNorm(h, postNorm, EPS);
  const act = mul(silu(matmul(y2, Wgate)), matmul(y2, Wup));
  const mlp = matmul(act, Wdown);
  return add(h, mlp);                           // residual
}

const out = block(x);
const sum = sumAll(out);
const sumsq = sumAll(mul(out, out));

console.log("Qwen3 decoder block — TS over mlx-c -> Metal");
console.log(`  output shape: [${B}, ${L}, ${D}]`);
console.log(`  sum    = ${sum.toFixed(6)}`);
console.log(`  sum_sq = ${sumsq.toFixed(6)}`);
