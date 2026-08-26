// Same Qwen3 decoder block as block.ts, but built entirely from the
// AUTO-GENERATED wrappers in generated.ts. If codegen is correct, this matches
// reference.py to ~float32.  Run: bun codegen.ts && bun block-gen.ts

import {type Arr,add, 
  array, fastRmsNorm, fastRope, fastScaledDotProductAttention, item, matmul, multiply, reshape,sigmoid, sum, 
  transposeAxes, 
} from "../src/ffi/generated.ts";

const B = 1, L = 3, D = 64, nH = 4, nKV = 2, Dh = 16, I = 128;
const EPS = 1e-6, THETA = 1_000_000, SCALE = Dh ** -0.5;

function det(n: number, seed: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.1;
  return a;
}
const W = (n: number, shape: number[], seed: number): Arr => array(det(n, seed), shape);
const silu = (a: Arr): Arr => multiply(a, sigmoid(a));

const inNorm = W(D, [D], 1), Wq = W(D * nH * Dh, [D, nH * Dh], 2);
const Wk = W(D * nKV * Dh, [D, nKV * Dh], 3), Wv = W(D * nKV * Dh, [D, nKV * Dh], 4);
const qNorm = W(Dh, [Dh], 5), kNorm = W(Dh, [Dh], 6);
const Wo = W(nH * Dh * D, [nH * Dh, D], 7), postNorm = W(D, [D], 8);
const Wgate = W(D * I, [D, I], 9), Wup = W(D * I, [D, I], 10), Wdown = W(I * D, [I, D], 11);
const x = array(det(B * L * D, 100), [B, L, D]);

function block(h: Arr): Arr {
  const y = fastRmsNorm(h, inNorm, EPS);
  let q = matmul(y, Wq), k = matmul(y, Wk), v = matmul(y, Wv);

  q = transposeAxes(fastRmsNorm(reshape(q, [B, L, nH, Dh]), qNorm, EPS), [0, 2, 1, 3]);
  k = transposeAxes(fastRmsNorm(reshape(k, [B, L, nKV, Dh]), kNorm, EPS), [0, 2, 1, 3]);
  v = transposeAxes(reshape(v, [B, L, nKV, Dh]), [0, 2, 1, 3]);

  q = fastRope(q, Dh, false, THETA, 1.0, 0, null);
  k = fastRope(k, Dh, false, THETA, 1.0, 0, null);

  let o = fastScaledDotProductAttention(q, k, v, SCALE, "causal", null, null);
  o = reshape(transposeAxes(o, [0, 2, 1, 3]), [B, L, nH * Dh]);
  h = add(h, matmul(o, Wo));

  const y2 = fastRmsNorm(h, postNorm, EPS);
  const act = multiply(silu(matmul(y2, Wgate)), matmul(y2, Wup));
  return add(h, matmul(act, Wdown));
}

const out = block(x);
const sumAll = (a: Arr) => item(sum(a, false));
console.log("Qwen3 decoder block — TS over GENERATED wrappers -> Metal");
console.log(`  output shape: [${B}, ${L}, ${D}]`);
console.log(`  sum    = ${sumAll(out).toFixed(6)}`);
console.log(`  sum_sq = ${sumAll(multiply(out, out)).toFixed(6)}`);
