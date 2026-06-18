// A small multi-layer Qwen3 model + KV-cache autoregressive decode loop,
// built from the generated wrappers. Greedy decoding; the produced token ids
// must match reference-decode.py exactly (discrete -> any drift shows up).
//
//   bun codegen.ts && bun model-gen.ts
//
// The eval-boundary discipline is the point: MLX is lazy, so after each decode
// step we force the new token AND the per-layer KV caches to concrete arrays.
// Without that, the graph (and memory) grows with every step and nothing runs.

import {
  array, arrayI32, itemU32, evalArray, vec,
  matmul, add, multiply, sigmoid, fastRmsNorm, reshape, transposeAxes,
  fastRope, fastScaledDotProductAttention, takeAxis, argmaxAxis, concatenateAxis,
  type Arr,
} from "./generated.ts";

// ---- config ----
const VOCAB = 32, D = 64, nH = 4, nKV = 2, Dh = 16, I = 128, LAYERS = 2;
const EPS = 1e-6, THETA = 1_000_000, SCALE = Dh ** -0.5;
const B = 1;

function det(n: number, seed: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.1;
  return a;
}
const W = (n: number, shape: number[], seed: number): Arr => array(det(n, seed), shape);
const silu = (a: Arr): Arr => multiply(a, sigmoid(a));

// ---- weights ----
const embed = W(VOCAB * D, [VOCAB, D], 0);
type Layer = ReturnType<typeof makeLayer>;
function makeLayer(l: number) {
  const s = (k: number) => 100 + l * 20 + k; // distinct seeds per layer
  return {
    inNorm: W(D, [D], s(1)),
    Wq: W(D * nH * Dh, [D, nH * Dh], s(2)),
    Wk: W(D * nKV * Dh, [D, nKV * Dh], s(3)),
    Wv: W(D * nKV * Dh, [D, nKV * Dh], s(4)),
    qNorm: W(Dh, [Dh], s(5)),
    kNorm: W(Dh, [Dh], s(6)),
    Wo: W(nH * Dh * D, [nH * Dh, D], s(7)),
    postNorm: W(D, [D], s(8)),
    Wgate: W(D * I, [D, I], s(9)),
    Wup: W(D * I, [D, I], s(10)),
    Wdown: W(I * D, [I, D], s(11)),
  };
}
const layers = Array.from({ length: LAYERS }, (_, l) => makeLayer(l));
const finalNorm = W(D, [D], 900);
const lmHead = W(D * VOCAB, [D, VOCAB], 901);

// ---- KV cache: one {k, v} per layer, null until first use ----
type KV = { k: Arr; v: Arr } | null;
const cache: KV[] = Array(LAYERS).fill(null);

function attnBlock(li: number, h: Arr, Lc: number, offset: number): Arr {
  const W = layers[li];
  const y = fastRmsNorm(h, W.inNorm, EPS);
  let q = transposeAxes(fastRmsNorm(reshape(matmul(y, W.Wq), [B, Lc, nH, Dh]), W.qNorm, EPS), [0, 2, 1, 3]);
  let k = transposeAxes(fastRmsNorm(reshape(matmul(y, W.Wk), [B, Lc, nKV, Dh]), W.kNorm, EPS), [0, 2, 1, 3]);
  let v = transposeAxes(reshape(matmul(y, W.Wv), [B, Lc, nKV, Dh]), [0, 2, 1, 3]);

  q = fastRope(q, Dh, false, THETA, 1.0, offset, null);
  k = fastRope(k, Dh, false, THETA, 1.0, offset, null);

  const prev = cache[li];
  if (prev) {                                  // append new k/v to the cache along seq axis
    k = concatenateAxis(vec([prev.k, k]), 2);
    v = concatenateAxis(vec([prev.v, v]), 2);
  }
  cache[li] = { k, v };

  // prefill (Lc>1) needs a causal mask; single-token decode attends to all -> no mask
  const o0 = fastScaledDotProductAttention(q, k, v, SCALE, Lc > 1 ? "causal" : "", null, null);
  const o = reshape(transposeAxes(o0, [0, 2, 1, 3]), [B, Lc, nH * Dh]);
  h = add(h, matmul(o, W.Wo));

  const y2 = fastRmsNorm(h, W.postNorm, EPS);
  const act = multiply(silu(matmul(y2, W.Wgate)), matmul(y2, W.Wup));
  return add(h, matmul(act, W.Wdown));
}

// Run `Lc` tokens at positions [offset .. offset+Lc), return the next-token id.
function step(ids: number[], offset: number): number {
  const Lc = ids.length;
  let h = takeAxis(embed, arrayI32(Int32Array.from(ids), [B, Lc]), 0); // [B, Lc, D]
  for (let li = 0; li < LAYERS; li++) h = attnBlock(li, h, Lc, offset);
  h = fastRmsNorm(h, finalNorm, EPS);

  // logits for the LAST position only -> argmax -> token id
  const hLast = takeAxis(h, arrayI32(Int32Array.from([Lc - 1]), [1]), 1); // [B, 1, D]
  const logits = matmul(hLast, lmHead);                                   // [B, 1, VOCAB]
  const tok = itemU32(argmaxAxis(logits, 2, false));

  // --- eval boundary: force the token + every KV cache to concrete arrays ---
  evalArray(...cache.flatMap((c) => (c ? [c.k, c.v] : [])));
  return tok;
}

// ---- generate ----
const prompt = [1, 7, 3, 9];
const N_NEW = 12;

const out: number[] = [];
let tok = step(prompt, 0);          // prefill; predicts token at pos = prompt.length
out.push(tok);
let pos = prompt.length;
for (let i = 1; i < N_NEW; i++) {
  tok = step([tok], pos);           // decode one token at position `pos`
  pos += 1;
  out.push(tok);
}

console.log("Qwen3 KV-cache decode — TS over generated wrappers -> Metal");
console.log(`  prompt:    [${prompt.join(", ")}]`);
console.log(`  generated: [${out.join(", ")}]`);
