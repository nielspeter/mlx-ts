// Config-driven Qwen3 inference: load a real downloaded Qwen3-0.6B
// (models/config.json + models/model.safetensors + models/tokenizer.json) and generate text.
// All dims come from models/config.json; weight names follow the HF layout.
//
//   bun qwen.ts "The capital of France is"

import {
  type Arr,add, argmaxAxis, 
  arrayI32, concatenateAxis,evalArray, fastRmsNorm, 
  fastRope, fastScaledDotProductAttention, itemU32, 
  matmul, multiply, reshape, sigmoid, takeAxis, transposeAxes,vec,
} from "../ffi/generated.ts";
import { readJson } from "../io/fs.ts";
import { get, loadSafetensors } from "../io/loader.ts";
import { Tokenizer } from "../text/tokenizer.ts";

const cfg = await readJson("models/config.json");
const NL = cfg.num_hidden_layers;
const nH = cfg.num_attention_heads, nKV = cfg.num_key_value_heads, Dh = cfg.head_dim;
const EPS = cfg.rms_norm_eps, THETA = cfg.rope_theta;
const SCALE = Dh ** -0.5, B = 1, EOS = cfg.eos_token_id;

const w = loadSafetensors("models/model-qwen.safetensors");
const tok = await Tokenizer.fromFile("models/tokenizer.json");

// HF weights are [out, in] (for x @ W.T). Pre-transpose to [in, out] once so the
// decode loop is plain matmul(x, Wt); materialize so it isn't recomputed.
function linT(name: string): Arr { const t = transposeAxes(get(w, name), [1, 0]); evalArray(t); return t; }
const norm = (name: string): Arr => get(w, name);

const embed = get(w, "model.embed_tokens.weight");   // [vocab, D]
const embedT = (() => { const t = transposeAxes(embed, [1, 0]); evalArray(t); return t; })(); // tied lm_head
const finalNorm = norm("model.norm.weight");
const layers = Array.from({ length: NL }, (_, i) => {
  const p = `model.layers.${i}`;
  return {
    inNorm: norm(`${p}.input_layernorm.weight`),
    postNorm: norm(`${p}.post_attention_layernorm.weight`),
    qNorm: norm(`${p}.self_attn.q_norm.weight`),
    kNorm: norm(`${p}.self_attn.k_norm.weight`),
    Wq: linT(`${p}.self_attn.q_proj.weight`), Wk: linT(`${p}.self_attn.k_proj.weight`),
    Wv: linT(`${p}.self_attn.v_proj.weight`), Wo: linT(`${p}.self_attn.o_proj.weight`),
    Wgate: linT(`${p}.mlp.gate_proj.weight`), Wup: linT(`${p}.mlp.up_proj.weight`),
    Wdown: linT(`${p}.mlp.down_proj.weight`),
  };
});

const silu = (a: Arr): Arr => multiply(a, sigmoid(a));
type KV = { k: Arr; v: Arr } | null;
const cache: KV[] = Array(NL).fill(null);

function block(li: number, h: Arr, Lc: number, offset: number): Arr {
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
  return add(h, matmul(multiply(silu(matmul(y2, W.Wgate)), matmul(y2, W.Wup)), W.Wdown));
}

function step(ids: number[], offset: number): number {
  const Lc = ids.length;
  let h = takeAxis(embed, arrayI32(Int32Array.from(ids), [B, Lc]), 0);
  for (let li = 0; li < NL; li++) h = block(li, h, Lc, offset);
  h = fastRmsNorm(h, finalNorm, EPS);
  const hLast = takeAxis(h, arrayI32(Int32Array.from([Lc - 1]), [1]), 1);
  const tk = itemU32(argmaxAxis(matmul(hLast, embedT), 2, false));
  evalArray(...cache.flatMap((c) => (c ? [c.k, c.v] : [])));
  return tk;
}

const prompt = process.argv[2] ?? "The capital of France is";
const N_NEW = 24;
const promptIds = tok.encode(prompt);

const gen: number[] = [];
let tk = step(promptIds, 0);
let pos = promptIds.length;
const t0 = performance.now();
for (let i = 0; tk !== EOS && i < N_NEW; i++) { gen.push(tk); tk = step([tk], pos); pos++; }
const dt = (performance.now() - t0) / 1000;

console.log("=== Qwen3-0.6B (real weights) — TS over mlx-c -> Metal ===");
console.log(`prompt:    ${JSON.stringify(prompt)}`);
console.log(`prompt ids: [${promptIds.join(", ")}]`);
console.log(`gen ids:    [${gen.join(", ")}]`);
console.log(`completion: ${JSON.stringify(tok.decode(gen))}`);
console.log(`(${gen.length} tokens in ${dt.toFixed(2)}s — ${(gen.length / dt).toFixed(1)} tok/s)`);
