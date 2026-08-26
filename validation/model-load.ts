// Qwen3 KV-cache decode, but weights are LOADED from a real .safetensors file
// via mlx_load_safetensors (not constructed in JS). Token ids must match
// reference-decode.py exactly.
//
//   python3 save-model.py && bun codegen.ts && bun model-load.ts

import {
  type Arr,add, argmaxAxis, 
  arrayI32, concatenateAxis,evalArray, fastRmsNorm, 
  fastRope, fastScaledDotProductAttention, itemU32, 
  matmul, multiply, reshape, sigmoid, takeAxis, transposeAxes,vec,
} from "../src/ffi/generated.ts";
import { entries, get, loadSafetensors } from "../src/io/loader.ts";

const VOCAB = 32, D = 64, nH = 4, nKV = 2, Dh = 16, I = 128, LAYERS = 2;
const EPS = 1e-6, THETA = 1_000_000, SCALE = Dh ** -0.5, B = 1;

const w = loadSafetensors("models/model.safetensors");
console.log(`loaded models/model.safetensors — ${entries(w).length} tensors`);

const embed = get(w, "embed");
const finalNorm = get(w, "finalNorm");
const lmHead = get(w, "lmHead");
const layers = Array.from({ length: LAYERS }, (_, l) => {
  const g = (name: string) => get(w, `layers.${l}.${name}`);
  return {
    inNorm: g("inNorm"), Wq: g("Wq"), Wk: g("Wk"), Wv: g("Wv"),
    qNorm: g("qNorm"), kNorm: g("kNorm"), Wo: g("Wo"), postNorm: g("postNorm"),
    Wgate: g("Wgate"), Wup: g("Wup"), Wdown: g("Wdown"),
  };
});

const silu = (a: Arr): Arr => multiply(a, sigmoid(a));
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
  if (prev) { k = concatenateAxis(vec([prev.k, k]), 2); v = concatenateAxis(vec([prev.v, v]), 2); }
  cache[li] = { k, v };
  const o0 = fastScaledDotProductAttention(q, k, v, SCALE, Lc > 1 ? "causal" : "", null, null);
  const o = reshape(transposeAxes(o0, [0, 2, 1, 3]), [B, Lc, nH * Dh]);
  h = add(h, matmul(o, W.Wo));
  const y2 = fastRmsNorm(h, W.postNorm, EPS);
  const act = multiply(silu(matmul(y2, W.Wgate)), matmul(y2, W.Wup));
  return add(h, matmul(act, W.Wdown));
}

function step(ids: number[], offset: number): number {
  const Lc = ids.length;
  let h = takeAxis(embed, arrayI32(Int32Array.from(ids), [B, Lc]), 0);
  for (let li = 0; li < LAYERS; li++) h = attnBlock(li, h, Lc, offset);
  h = fastRmsNorm(h, finalNorm, EPS);
  const hLast = takeAxis(h, arrayI32(Int32Array.from([Lc - 1]), [1]), 1);
  const tok = itemU32(argmaxAxis(matmul(hLast, lmHead), 2, false));
  evalArray(...cache.flatMap((c) => (c ? [c.k, c.v] : [])));
  return tok;
}

const prompt = [1, 7, 3, 9];
const out: number[] = [];
let tok = step(prompt, 0);
out.push(tok);
let pos = prompt.length;
for (let i = 1; i < 12; i++) { tok = step([tok], pos); pos += 1; out.push(tok); }

console.log("Qwen3 KV-cache decode — weights LOADED from safetensors -> Metal");
console.log(`  prompt:    [${prompt.join(", ")}]`);
console.log(`  generated: [${out.join(", ")}]`);
