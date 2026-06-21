// SFT (supervised fine-tuning) — the nanochat chat stage, on real GPT-2-124M.
// Takes the pretrained OpenAI weights and fine-tunes them into an instruction
// follower on chat-formatted examples, with the SFT-defining detail: the loss is
// computed ONLY on the assistant's response tokens (the prompt is masked out).
// Full fine-tune (all 124M params), real MLX value_and_grad over FFI, AdamW.
//   bun sft.ts
// Validated vs reference-sft.py: same loaded weights + data -> step-0 loss matches;
// both converge. (Training drifts past step 0 — FINDINGS §6 #8 — so the bar is
// "same start, both converge", as for lora-train / nanoGPT.)
import { MX, fromI32, scalar, evalAll, clearCache, tidy } from "./mx.ts";
import { crossEntropy } from "./loss.ts";
import { valueAndGrad } from "./train.ts";
import { treeFlatten, treeUnflattenLike, type Tree } from "./pytree.ts";
import { loadSafetensors, get } from "./loader.ts";
import { Tokenizer, GPT2_SPLIT } from "./tokenizer.ts";

const cfg = await Bun.file("config-gpt2.json").json();
const D = cfg.n_embd, NL = cfg.n_layer, NH = cfg.n_head, Dh = D / NH;
const EPS = cfg.layer_norm_epsilon, ASCALE = Dh ** -0.5, EOS = 50256, VOCAB = cfg.vocab_size;
const ITERS = +(process.env.ITERS ?? 120), LR0 = +(process.env.LR ?? 3e-5), WARMUP = 10;
const B1 = 0.9, B2 = 0.95, EPSA = 1e-8, WD = 0, CLIP = 1.0;   // SFT: small LR, grad clip, no weight decay

const tok = await Tokenizer.fromFile("gpt2-tokenizer.json", GPT2_SPLIT);

// --- load GPT-2 weights into a trainable MX params tree (copy out -> independent
// buffers we own; the fused QKV Conv1D weight is split into q/k/v). ---
const w = loadSafetensors("gpt2-model.safetensors");
const cp = (name: string): MX => new MX(get(w, name) as number).copy();   // materialize
const blocks = Array.from({ length: NL }, (_, i) => {
  const p = `h.${i}`;
  const caW = new MX(get(w, `${p}.attn.c_attn.weight`) as number), caB = new MX(get(w, `${p}.attn.c_attn.bias`) as number);
  return {
    ln1w: cp(`${p}.ln_1.weight`), ln1b: cp(`${p}.ln_1.bias`),
    wq: caW.slice([0, 0], [D, D]).copy(), bq: caB.slice([0], [D]).copy(),
    wk: caW.slice([0, D], [D, 2 * D]).copy(), bk: caB.slice([D], [2 * D]).copy(),
    wv: caW.slice([0, 2 * D], [D, 3 * D]).copy(), bv: caB.slice([2 * D], [3 * D]).copy(),
    wo: cp(`${p}.attn.c_proj.weight`), bo: cp(`${p}.attn.c_proj.bias`),
    ln2w: cp(`${p}.ln_2.weight`), ln2b: cp(`${p}.ln_2.bias`),
    wfc: cp(`${p}.mlp.c_fc.weight`), bfc: cp(`${p}.mlp.c_fc.bias`),
    wproj: cp(`${p}.mlp.c_proj.weight`), bproj: cp(`${p}.mlp.c_proj.bias`),
  };
});
let params: Tree = { wte: cp("wte.weight"), wpe: cp("wpe.weight"), blocks, lnfw: cp("ln_f.weight"), lnfb: cp("ln_f.bias") };
evalAll(...treeFlatten(params));
const nParams = treeFlatten(params).reduce((a, p) => a + p.size, 0);

// gelu_new (GPT-2 activation, tanh approx) — constants hoisted once
const C5 = scalar(0.5), C1 = scalar(1), CK = scalar(0.7978845608028654), CA = scalar(0.044715);
const geluNew = (x: MX): MX => x.mul(C5).mul(x.add(x.mul(x).mul(x).mul(CA)).mul(CK).tanh().add(C1));

// forward: GPT-2 over the params tree. ids:[1,L] -> logits:[1,L,V]
function forward(p: any, ids: MX, L: number): MX {
  const pos = fromI32(Int32Array.from({ length: L }, (_, i) => i), [L]);
  let x = p.wte.takeAxis(ids, 0).add(p.wpe.takeAxis(pos, 0));        // [1,L,D]
  const head = (t: MX, wt: MX, b: MX) => t.matmul(wt).add(b).reshape([1, L, NH, Dh]).transpose([0, 2, 1, 3]);
  for (const blk of p.blocks) {
    const n1 = x.layerNorm(blk.ln1w, blk.ln1b, EPS);
    const q = head(n1, blk.wq, blk.bq), k = head(n1, blk.wk, blk.bk), v = head(n1, blk.wv, blk.bv);
    const att = MX.sdpa(q, k, v, ASCALE, true).transpose([0, 2, 1, 3]).reshape([1, L, D]);
    x = x.add(att.matmul(blk.wo).add(blk.bo));
    const n2 = x.layerNorm(blk.ln2w, blk.ln2b, EPS);
    x = x.add(geluNew(n2.matmul(blk.wfc).add(blk.bfc)).matmul(blk.wproj).add(blk.bproj));
  }
  return x.layerNorm(p.lnfw, p.lnfb, EPS).matmul(p.wte.transpose([1, 0]));  // tied -> [1,L,V]
}

// --- SFT data: chat-formatted, loss on the response only ---
const PROMPT = (q: string) => `User: ${q}\nAssistant:`;
const DATA = [
  { q: "What is the capital of France?", a: "The capital of France is Paris." },
  { q: "What is the capital of Japan?", a: "The capital of Japan is Tokyo." },
  { q: "Who wrote Romeo and Juliet?", a: "Romeo and Juliet was written by William Shakespeare." },
  { q: "What is 2 plus 2?", a: "2 plus 2 equals 4." },
  { q: "What color is the sky on a clear day?", a: "On a clear day the sky is blue." },
  { q: "What is the largest planet?", a: "The largest planet in our solar system is Jupiter." },
];
// each example -> input ids [1,L]; loss only on the completion (a contiguous
// suffix), so we slice to the response range instead of masking — numerically
// clean (no inf*0 from prompt positions whose logprob underflows during training).
const examples = DATA.map(({ q, a }) => {
  const pIds = tok.encode(PROMPT(q));
  const ids = tok.encode(PROMPT(q) + " " + a); ids.push(EOS);       // append end-of-text
  const L = ids.length, c = pIds.length - 1;                        // first completion target index
  const comp = Int32Array.from(ids.slice(c + 1));                   // response tokens + EOS
  return { idsMX: fromI32(Int32Array.from(ids), [1, L]), L, tgt: fromI32(comp, [L - 1 - c, 1]) };
});

const lossFn = (p: Tree, idsMX: MX, tgt: MX, lenMX: MX): MX => {
  const L = lenMX.shape[0], c = L - 1 - tgt.shape[0];              // completion start
  const logits = forward(p, idsMX, L).reshape([L, VOCAB]).slice([c, 0], [L - 1, VOCAB]); // response rows only
  return crossEntropy(logits, tgt);
};
const vg = valueAndGrad(params, lossFn);

// --- AdamW with bias-correction folded into the step size (same as spike-nanogpt) ---
const sB1 = scalar(B1), sB1m = scalar(1 - B1), sB2 = scalar(B2), sB2m = scalar(1 - B2);
const sumsq = (g: MX) => g.mul(g).sumAxes(g.shape.map((_, i) => i), false);
let mS: MX[] | null = null, vS: MX[] | null = null;
const lrAt = (it: number) => it < WARMUP ? LR0 * (it + 1) / WARMUP : LR0;   // warmup then constant

function genReply(p: any, q: string, maxNew = 32): string {
  return tidy(() => {
    let ids = tok.encode(PROMPT(q));
    const out: number[] = [];
    for (let i = 0; i < maxNew; i++) {
      const logits = forward(p, fromI32(Int32Array.from(ids), [1, ids.length]), ids.length).reshape([ids.length, VOCAB]);
      const tk = logits.slice([ids.length - 1, 0], [ids.length, VOCAB]).argmax(1).itemU();
      if (tk === EOS) break;
      ids.push(tk); out.push(tk);
    }
    return tok.decode(out).trim();
  });
}

console.log(`=== SFT GPT-2-124M (${(nParams / 1e6).toFixed(1)}M params, full fine-tune) — real MLX autograd over FFI ===`);
console.log(`before: ${JSON.stringify(genReply(params, DATA[0].q))}`);

let loss = 0, step0 = 0;
for (let it = 0; it < ITERS; it++) {
  const lr = lrAt(it), bc1 = 1 - B1 ** (it + 1), bc2 = 1 - B2 ** (it + 1);
  const ex = examples[it % examples.length];
  const fp = treeFlatten(params), oldM = mS, oldV = vS;
  const lenMX = fromI32(new Int32Array(ex.L), [ex.L]);
  const kept = tidy(() => {
    const r = vg(params, ex.idsMX, ex.tgt, lenMX);
    const fg = treeFlatten(r.grads);
    let total = sumsq(fg[0]); for (let i = 1; i < fg.length; i++) total = total.add(sumsq(fg[i]));
    const gnorm = total.sqrt().itemF();
    const cs = gnorm > CLIP ? CLIP / gnorm : 1;            // global grad-norm clip
    const sCs = cs === 1 ? null : scalar(cs);
    const sqBc2 = Math.sqrt(bc2);
    const sAlpha = scalar(lr * sqBc2 / bc1), sEpsHat = scalar(EPSA * sqBc2), sDecay = scalar(1 - lr * WD);
    const nP: MX[] = [], nM: MX[] = [], nV: MX[] = [];
    fp.forEach((pp, i) => {
      const g = sCs ? fg[i].mul(sCs) : fg[i];
      const mi = mS ? mS[i].mul(sB1).add(g.mul(sB1m)) : g.mul(sB1m);
      const vi = vS ? vS[i].mul(sB2).add(g.mul(g).mul(sB2m)) : g.mul(g).mul(sB2m);
      const core = mi.div(vi.sqrt().add(sEpsHat));
      const pNew = WD && pp.shape.length >= 2 ? pp.mul(sDecay).sub(core.mul(sAlpha)) : pp.sub(core.mul(sAlpha));
      nP.push(pNew); nM.push(mi); nV.push(vi);
    });
    evalAll(...nP, ...nM, ...nV);
    return { p: treeUnflattenLike(params, nP), m: nM, v: nV, loss: r.loss };
  });
  loss = kept.loss;
  if (it === 0) step0 = loss;
  params = kept.p; mS = kept.m; vS = kept.v;
  fp.forEach((x) => x.free()); oldM?.forEach((x) => x.free()); oldV?.forEach((x) => x.free());
  if (it % 20 === 0) { console.log(`  iter ${String(it).padStart(3)}: loss ${loss.toFixed(4)} (lr ${lr.toExponential(1)})`); clearCache(); }
}
console.log(`STEP0 loss=${step0.toFixed(4)}`);
console.log(`FINAL loss=${loss.toFixed(4)}`);

console.log("\n--- after SFT ---");
for (const q of [DATA[0].q, DATA[3].q, "What is the capital of Italy?"])   // last = held-out
  console.log(`Q: ${q}\nA: ${genReply(params, q)}\n`);
