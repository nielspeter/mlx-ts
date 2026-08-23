// 4-bit quantized Qwen3 decode in TS: loads quantized weights from
// model-quant.safetensors and uses quantizedMatmul for the Linear projections.
// Token ids must match reference-quant.py exactly.
//
//   python3 reference-quant.py && bun codegen.ts && bun model-quant.ts

import {
  arrayI32, itemU32, evalArray, vec,
  add, multiply, sigmoid, fastRmsNorm, reshape, transposeAxes,
  fastRope, fastScaledDotProductAttention, takeAxis, argmaxAxis, concatenateAxis,
  quantizedMatmul, type Arr,
} from "../src/ffi/generated.ts";
import { loadSafetensors, get, entries } from "../src/io/loader.ts";

const VOCAB = 32, D = 64, nH = 4, nKV = 2, Dh = 16, I = 128, LAYERS = 2;
const EPS = 1e-6, THETA = 1_000_000, SCALE = Dh ** -0.5, B = 1;
const GS = 64, BITS = 4;

const w = loadSafetensors("model-quant.safetensors");
console.log(`loaded model-quant.safetensors — ${entries(w).length} tensors`);

// a quantized projection = (packed weight, scales, biases)
type Q = { wq: Arr; scales: Arr; biases: Arr };
const q = (name: string): Q => ({ wq: get(w, `${name}.weight`), scales: get(w, `${name}.scales`), biases: get(w, `${name}.biases`) });
// y = x @ dequant(W).T
const qmm = (x: Arr, p: Q): Arr => quantizedMatmul(x, p.wq, p.scales, p.biases, true, GS, BITS, "affine");

const embed = get(w, "embed");
const finalNorm = get(w, "finalNorm");
const lmHead = q("lmHead");
const layers = Array.from({ length: LAYERS }, (_, l) => {
  const n = (s: string) => get(w, `layers.${l}.${s}`);
  return {
    inNorm: n("inNorm"), qNorm: n("qNorm"), kNorm: n("kNorm"), postNorm: n("postNorm"),
    Wq: q(`layers.${l}.Wq`), Wk: q(`layers.${l}.Wk`), Wv: q(`layers.${l}.Wv`), Wo: q(`layers.${l}.Wo`),
    Wgate: q(`layers.${l}.Wgate`), Wup: q(`layers.${l}.Wup`), Wdown: q(`layers.${l}.Wdown`),
  };
});

const silu = (a: Arr): Arr => multiply(a, sigmoid(a));
type KV = { k: Arr; v: Arr } | null;
const cache: KV[] = Array(LAYERS).fill(null);

function attnBlock(li: number, h: Arr, Lc: number, offset: number): Arr {
  const W = layers[li];
  const y = fastRmsNorm(h, W.inNorm, EPS);
  let qh = transposeAxes(fastRmsNorm(reshape(qmm(y, W.Wq), [B, Lc, nH, Dh]), W.qNorm, EPS), [0, 2, 1, 3]);
  let k = transposeAxes(fastRmsNorm(reshape(qmm(y, W.Wk), [B, Lc, nKV, Dh]), W.kNorm, EPS), [0, 2, 1, 3]);
  let v = transposeAxes(reshape(qmm(y, W.Wv), [B, Lc, nKV, Dh]), [0, 2, 1, 3]);
  qh = fastRope(qh, Dh, false, THETA, 1.0, offset, null);
  k = fastRope(k, Dh, false, THETA, 1.0, offset, null);
  const prev = cache[li];
  if (prev) { k = concatenateAxis(vec([prev.k, k]), 2); v = concatenateAxis(vec([prev.v, v]), 2); }
  cache[li] = { k, v };
  const o0 = fastScaledDotProductAttention(qh, k, v, SCALE, Lc > 1 ? "causal" : "", null, null);
  const o = reshape(transposeAxes(o0, [0, 2, 1, 3]), [B, Lc, nH * Dh]);
  h = add(h, qmm(o, W.Wo));
  const y2 = fastRmsNorm(h, W.postNorm, EPS);
  const act = multiply(silu(qmm(y2, W.Wgate)), qmm(y2, W.Wup));
  return add(h, qmm(act, W.Wdown));
}

function step(ids: number[], offset: number): number {
  const Lc = ids.length;
  let h = takeAxis(embed, arrayI32(Int32Array.from(ids), [B, Lc]), 0);
  for (let li = 0; li < LAYERS; li++) h = attnBlock(li, h, Lc, offset);
  h = fastRmsNorm(h, finalNorm, EPS);
  const hLast = takeAxis(h, arrayI32(Int32Array.from([Lc - 1]), [1]), 1);
  const tok = itemU32(argmaxAxis(qmm(hLast, lmHead), 2, false));
  evalArray(...cache.flatMap((c) => (c ? [c.k, c.v] : [])));
  return tok;
}

const prompt = [1, 7, 3, 9];
const out: number[] = [];
let tok = step(prompt, 0); out.push(tok);
let pos = prompt.length;
for (let i = 1; i < 12; i++) { tok = step([tok], pos); pos += 1; out.push(tok); }

console.log("Qwen3 4-bit KV-cache decode — quantized weights LOADED -> Metal");
console.log(`  generated: [${out.join(", ")}]`);
