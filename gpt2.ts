// Real OpenAI GPT-2-124M inference: load the actual openai-community/gpt2 weights
// (gpt2-model.safetensors + config-gpt2.json + gpt2-tokenizer.json) and generate.
// Architecture is GPT-2 exactly: learned positional embeddings, LayerNorm WITH
// bias, fused QKV (Conv1D), gelu_new (tanh approx), tied lm_head. KV-cached greedy
// decode. Token-exact vs reference-gpt2.py (same weights, MLX Python).
//   bun gpt2.ts "The capital of France is"
import {
  array, arrayI32, itemU32, evalArray, vec,
  matmul, add, multiply, reshape, transposeAxes, fastLayerNorm,
  fastScaledDotProductAttention, takeAxis, argmaxAxis, concatenateAxis,
  tanh, square, slice, type Arr,
} from "./src/ffi/generated.ts";
import { loadSafetensors, get } from "./loader.ts";
import { Tokenizer, GPT2_SPLIT } from "./tokenizer.ts";
import { MX, sample as mxSample, applyRepetitionPenalty, seed as mxSeed } from "./mx.ts";

const cfg = await Bun.file("config-gpt2.json").json();
const D = cfg.n_embd, NL = cfg.n_layer, nH = cfg.n_head, Dh = D / nH;
const EPS = cfg.layer_norm_epsilon, SCALE = Dh ** -0.5, B = 1;
const EOS = 50256, VOCAB = cfg.vocab_size;

// Sampling (env-configurable). Default = pure greedy argmax, which stays
// token-exact vs reference-gpt2.py. Set TEMP>0 / TOP_K / TOP_P / REP to sample.
//   TEMP=0.8 TOP_K=40 REP=1.3 SEED=1 bun gpt2.ts "Once upon a time"
const TEMP = +(process.env.TEMP ?? 0), TOP_K = +(process.env.TOP_K ?? 0);
const TOP_P = +(process.env.TOP_P ?? 0), REP = +(process.env.REP ?? 1);
if (process.env.SEED) mxSeed(+process.env.SEED);
const SAMPLING = TEMP > 0 || TOP_K > 0 || TOP_P > 0 || REP !== 1;
let histIds: number[] = [];   // full context, for the repetition penalty

const w = loadSafetensors("gpt2-model.safetensors");
const tok = await Tokenizer.fromFile("gpt2-tokenizer.json", GPT2_SPLIT);
const g = (n: string): Arr => get(w, n);

// GPT-2 stores Linear as Conv1D weight [in, out] -> matmul(x, W) directly (no
// transpose, unlike Qwen). lm_head is tied to wte, so transpose wte once.
const wte = g("wte.weight"), wpe = g("wpe.weight");        // [vocab,D], [n_pos,D]
const wteT = (() => { const t = transposeAxes(wte, [1, 0]); evalArray(t); return t; })();
const lnfW = g("ln_f.weight"), lnfB = g("ln_f.bias");
const mat = (a: Arr): Arr => { evalArray(a); return a; };  // materialize a sliced weight
const layers = Array.from({ length: NL }, (_, i) => {
  const p = `h.${i}`;
  const caW = g(`${p}.attn.c_attn.weight`), caB = g(`${p}.attn.c_attn.bias`);  // [D,3D],[3D]
  return {
    ln1W: g(`${p}.ln_1.weight`), ln1B: g(`${p}.ln_1.bias`),
    Wq: mat(slice(caW, [0, 0], [D, D], [1, 1])), bq: mat(slice(caB, [0], [D], [1])),
    Wk: mat(slice(caW, [0, D], [D, 2 * D], [1, 1])), bk: mat(slice(caB, [D], [2 * D], [1])),
    Wv: mat(slice(caW, [0, 2 * D], [D, 3 * D], [1, 1])), bv: mat(slice(caB, [2 * D], [3 * D], [1])),
    Wo: g(`${p}.attn.c_proj.weight`), bo: g(`${p}.attn.c_proj.bias`),
    ln2W: g(`${p}.ln_2.weight`), ln2B: g(`${p}.ln_2.bias`),
    Wfc: g(`${p}.mlp.c_fc.weight`), bfc: g(`${p}.mlp.c_fc.bias`),
    Wproj: g(`${p}.mlp.c_proj.weight`), bproj: g(`${p}.mlp.c_proj.bias`),
  };
});

// gelu_new (GPT-2 activation): 0.5x(1+tanh(√(2/π)(x+0.044715x³)))
const C5 = array(Float32Array.from([0.5]), [1]), C1 = array(Float32Array.from([1]), [1]);
const CK = array(Float32Array.from([0.7978845608028654]), [1]), CA = array(Float32Array.from([0.044715]), [1]);
const geluNew = (x: Arr): Arr =>
  multiply(multiply(C5, x), add(C1, tanh(multiply(CK, add(x, multiply(CA, multiply(x, square(x))))))));

type KV = { k: Arr; v: Arr } | null;
const cache: KV[] = Array(NL).fill(null);

function block(li: number, h: Arr, Lc: number): Arr {
  const W = layers[li];
  const x1 = fastLayerNorm(h, W.ln1W, W.ln1B, EPS);
  const head = (Wt: Arr, b: Arr) => transposeAxes(reshape(add(matmul(x1, Wt), b), [B, Lc, nH, Dh]), [0, 2, 1, 3]);
  let q = head(W.Wq, W.bq), k = head(W.Wk, W.bk), v = head(W.Wv, W.bv);
  const prev = cache[li];
  if (prev) { k = concatenateAxis(vec([prev.k, k]), 2); v = concatenateAxis(vec([prev.v, v]), 2); }
  cache[li] = { k, v };
  const o0 = fastScaledDotProductAttention(q, k, v, SCALE, Lc > 1 ? "causal" : "", null, null);
  const o = reshape(transposeAxes(o0, [0, 2, 1, 3]), [B, Lc, D]);
  h = add(h, add(matmul(o, W.Wo), W.bo));
  const x2 = fastLayerNorm(h, W.ln2W, W.ln2B, EPS);
  return add(h, add(matmul(geluNew(add(matmul(x2, W.Wfc), W.bfc)), W.Wproj), W.bproj));
}

function step(ids: number[], offset: number): number {
  const Lc = ids.length;
  const tokEmb = takeAxis(wte, arrayI32(Int32Array.from(ids), [B, Lc]), 0);            // [B,Lc,D]
  const posEmb = takeAxis(wpe, arrayI32(Int32Array.from(ids.map((_, i) => offset + i)), [Lc]), 0);
  let h = add(tokEmb, posEmb);
  for (let li = 0; li < NL; li++) h = block(li, h, Lc);
  h = fastLayerNorm(h, lnfW, lnfB, EPS);
  const hLast = takeAxis(h, arrayI32(Int32Array.from([Lc - 1]), [1]), 1);             // [B,1,D]
  const logits = matmul(hLast, wteT);                                                 // [B,1,vocab]
  let tk: number;
  if (!SAMPLING) {
    tk = itemU32(argmaxAxis(logits, 2, false));                                       // pure greedy (token-exact)
  } else {
    let lm = new MX(reshape(logits, [B, VOCAB]));                                     // [B,vocab]
    if (REP !== 1) lm = applyRepetitionPenalty(lm, histIds, REP);
    tk = mxSample(lm, TEMP, TOP_P, TOP_K).itemU();                                    // temp->top-k->top-p->categorical
  }
  evalArray(...cache.flatMap((c) => (c ? [c.k, c.v] : [])));
  return tk;
}

const prompt = process.argv[2] ?? "The capital of France is";
const N_NEW = 24;
const promptIds = tok.encode(prompt);
histIds = [...promptIds];
const gen: number[] = [];
let tk = step(promptIds, 0), pos = promptIds.length;
const t0 = performance.now();
for (let i = 0; tk !== EOS && i < N_NEW; i++) { gen.push(tk); histIds.push(tk); tk = step([tk], pos); pos++; }
const dt = (performance.now() - t0) / 1000;

const mode = SAMPLING ? `sampling (temp=${TEMP}, top-k=${TOP_K}, top-p=${TOP_P}, rep=${REP})` : "greedy";
console.log("=== GPT-2-124M (real OpenAI weights) — TS over mlx-c -> Metal ===");
console.log(`decode:    ${mode}`);
console.log(`prompt:    ${JSON.stringify(prompt)}`);
console.log(`prompt ids: [${promptIds.join(", ")}]`);
console.log(`gen ids:    [${gen.join(", ")}]`);
console.log(`completion: ${JSON.stringify(tok.decode(gen))}`);
console.log(`(${gen.length} tokens in ${dt.toFixed(2)}s — ${(gen.length / dt).toFixed(1)} tok/s)`);
