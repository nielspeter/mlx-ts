// RL stage (GRPO) — the nanochat RL step, on real GPT-2-124M. Group Relative
// Policy Optimization: for each prompt, sample a GROUP of completions, score them
// with a verifiable reward, normalize advantage WITHIN the group, and do a
// policy-gradient update. The GRPO objective reduces to advantage-weighted NLL:
//   loss = mean_seq( advantage_seq * crossEntropy(completion) )
// (positive advantage -> minimize NLL -> reinforce; negative -> discourage.)
//
// Task: positivity steering (RLHF-flavored, verifiable) — reward = # of positive
// words in the completion. The policy learns to complete more positively; mean
// reward rises. No labeled completions, just a reward signal.
//   bun rl.ts
import { MX, fromI32, scalar, evalAll, seed, sample, clearCache, tidy } from "../src/core/mx.ts";
import { crossEntropy } from "../src/nn/loss.ts";
import { valueAndGrad } from "./train.ts";
import { treeFlatten, treeUnflattenLike, type Tree } from "../src/core/pytree.ts";
import { loadSafetensors, get } from "../src/io/loader.ts";
import { Tokenizer, GPT2_SPLIT } from "../src/text/tokenizer.ts";

const cfg = await Bun.file("models/config-gpt2.json").json();
const D = cfg.n_embd, NL = cfg.n_layer, NH = cfg.n_head, Dh = D / NH;
const EPS = cfg.layer_norm_epsilon, ASCALE = Dh ** -0.5, EOS = 50256, VOCAB = cfg.vocab_size;
const STEPS = +(process.env.STEPS ?? 25), G = +(process.env.GROUP ?? 8), MAXNEW = +(process.env.MAXNEW ?? 12);
const TEMP = +(process.env.TEMP ?? 1.0), TOPK = +(process.env.TOPK ?? 50), LR0 = +(process.env.LR ?? 1e-5);
const B1 = 0.9, B2 = 0.95, EPSA = 1e-8, CLIP = 1.0;

const tok = await Tokenizer.fromFile("models/gpt2-tokenizer.json", GPT2_SPLIT);

// --- load GPT-2 into a trainable MX params tree (same as sft.ts) ---
const w = loadSafetensors("models/gpt2-model.safetensors");
const cp = (name: string): MX => new MX(get(w, name) as number).copy();
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

const C5 = scalar(0.5), C1 = scalar(1), CK = scalar(0.7978845608028654), CA = scalar(0.044715);
const geluNew = (x: MX): MX => x.mul(C5).mul(x.add(x.mul(x).mul(x).mul(CA)).mul(CK).tanh().add(C1));
function forward(p: any, ids: MX, L: number): MX {
  const pos = fromI32(Int32Array.from({ length: L }, (_, i) => i), [L]);
  let x = p.wte.takeAxis(ids, 0).add(p.wpe.takeAxis(pos, 0));
  const head = (t: MX, wt: MX, b: MX) => t.matmul(wt).add(b).reshape([1, L, NH, Dh]).transpose([0, 2, 1, 3]);
  for (const blk of p.blocks) {
    const n1 = x.layerNorm(blk.ln1w, blk.ln1b, EPS);
    const q = head(n1, blk.wq, blk.bq), k = head(n1, blk.wk, blk.bk), v = head(n1, blk.wv, blk.bv);
    const att = MX.sdpa(q, k, v, ASCALE, true).transpose([0, 2, 1, 3]).reshape([1, L, D]);
    x = x.add(att.matmul(blk.wo).add(blk.bo));
    const n2 = x.layerNorm(blk.ln2w, blk.ln2b, EPS);
    x = x.add(geluNew(n2.matmul(blk.wfc).add(blk.bfc)).matmul(blk.wproj).add(blk.bproj));
  }
  return x.layerNorm(p.lnfw, p.lnfb, EPS).matmul(p.wte.transpose([1, 0]));
}

// --- verifiable reward: count positive words in the completion ---
const PROMPTS = ["The movie was", "I think the food was", "My day today was", "The new restaurant is", "Overall, the experience was"];
const POS = ["great", "good", "amazing", "excellent", "wonderful", "fantastic", "love", "loved", "best", "happy", "beautiful", "perfect", "brilliant", "awesome", "enjoyable", "delightful", "nice", "incredible"];
const reward = (text: string): number => {
  const t = text.toLowerCase();
  return POS.reduce((s, word) => s + (t.match(new RegExp(`\\b${word}\\b`, "g"))?.length ?? 0), 0);
};

// sample one completion (stochastic) from the current policy
let rollSeed = 1234;
function rollout(p: any, promptIds: number[]): number[] {
  return tidy(() => {
    seed(rollSeed++);                                       // vary each rollout
    const ids = [...promptIds], out: number[] = [];
    for (let i = 0; i < MAXNEW; i++) {
      const logits = forward(p, fromI32(Int32Array.from(ids), [1, ids.length]), ids.length).reshape([ids.length, VOCAB]);
      const last = logits.slice([ids.length - 1, 0], [ids.length, VOCAB]);
      const tk = sample(last, TEMP, 0, TOPK).itemU();
      if (tk === EOS) break;
      ids.push(tk); out.push(tk);
    }
    return out;
  });
}

// GRPO loss over the current batch (set per step) — advantage-weighted NLL
type Seq = { idsMX: MX; L: number; cStart: number; tgtMX: MX; adv: number };
let BATCH: Seq[] = [];
function rlLoss(p: Tree): MX {
  let total: MX | null = null;
  for (const b of BATCH) {
    const logits = forward(p, b.idsMX, b.L).reshape([b.L, VOCAB]).slice([b.cStart, 0], [b.L - 1, VOCAB]);
    const term = crossEntropy(logits, b.tgtMX).mul(scalar(b.adv));   // adv>0 -> reinforce, adv<0 -> suppress
    total = total ? total.add(term) : term;
  }
  return total!.div(scalar(BATCH.length));
}
const vg = valueAndGrad(params, rlLoss);

// CHECK mode: validate the GRPO loss path (forward + CE + advantage weighting) on
// a FIXED batch vs reference-rl.py — rollouts are random, but the loss is not.
if (process.env.CHECK) {
  const P = "The movie was";
  const fixed = [{ c: " great and wonderful", adv: 1.0 }, { c: " terrible and awful", adv: -1.0 }];
  BATCH = fixed.map(({ c, adv }) => {
    const pIds = tok.encode(P), comp = tok.encode(c), ids = [...pIds, ...comp];
    return { idsMX: fromI32(Int32Array.from(ids), [1, ids.length]), L: ids.length, cStart: pIds.length - 1, tgtMX: fromI32(Int32Array.from(comp), [comp.length, 1]), adv };
  });
  console.log(`RLLOSS=${rlLoss(params).itemF().toFixed(5)}`);
  process.exit(0);
}

// --- AdamW (bias-correction folded, grad clip) ---
const sB1 = scalar(B1), sB1m = scalar(1 - B1), sB2 = scalar(B2), sB2m = scalar(1 - B2);
const sumsq = (g: MX) => g.mul(g).sumAxes(g.shape.map((_, i) => i), false);
let mS: MX[] | null = null, vS: MX[] | null = null;
const avg = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;

console.log(`=== RL (GRPO) on GPT-2-124M — positivity reward, group size ${G} ===`);
for (let step = 0; step < STEPS; step++) {
  // 1. rollout + reward + group-relative advantage
  const seqs: { ids: number[]; pLen: number; comp: number[]; adv: number }[] = [];
  const allR: number[] = [];
  for (const prompt of PROMPTS) {
    const promptIds = tok.encode(prompt);
    const comps = Array.from({ length: G }, () => rollout(params, promptIds));
    const rs = comps.map((c) => reward(tok.decode(c)));
    const m = avg(rs), sd = Math.sqrt(avg(rs.map((r) => (r - m) ** 2)));
    rs.forEach((r, i) => { if (comps[i].length > 0 && sd > 1e-6) seqs.push({ ids: [...promptIds, ...comps[i]], pLen: promptIds.length, comp: comps[i], adv: (r - m) / (sd + 1e-4) }); });
    allR.push(...rs);
  }
  if (seqs.length === 0) { console.log(`  step ${step}: no advantage signal (reward ${avg(allR).toFixed(2)})`); continue; }

  // 2. policy-gradient update
  BATCH = seqs.map((s) => ({ idsMX: fromI32(Int32Array.from(s.ids), [1, s.ids.length]), L: s.ids.length, cStart: s.pLen - 1, tgtMX: fromI32(Int32Array.from(s.comp), [s.comp.length, 1]), adv: s.adv }));
  const lr = LR0, bc1 = 1 - B1 ** (step + 1), bc2 = 1 - B2 ** (step + 1);
  const fp = treeFlatten(params), oldM = mS, oldV = vS;
  const kept = tidy(() => {
    const r = vg(params);
    const fg = treeFlatten(r.grads);
    let tot = sumsq(fg[0]); for (let i = 1; i < fg.length; i++) tot = tot.add(sumsq(fg[i]));
    const gnorm = tot.sqrt().itemF(), cs = gnorm > CLIP ? CLIP / gnorm : 1, sCs = cs === 1 ? null : scalar(cs);
    const sqBc2 = Math.sqrt(bc2), sAlpha = scalar(lr * sqBc2 / bc1), sEpsHat = scalar(EPSA * sqBc2);
    const nP: MX[] = [], nM: MX[] = [], nV: MX[] = [];
    fp.forEach((pp, i) => {
      const g = sCs ? fg[i].mul(sCs) : fg[i];
      const mi = mS ? mS[i].mul(sB1).add(g.mul(sB1m)) : g.mul(sB1m);
      const vi = vS ? vS[i].mul(sB2).add(g.mul(g).mul(sB2m)) : g.mul(g).mul(sB2m);
      nP.push(pp.sub(mi.div(vi.sqrt().add(sEpsHat)).mul(sAlpha))); nM.push(mi); nV.push(vi);
    });
    evalAll(...nP, ...nM, ...nV);
    return { p: treeUnflattenLike(params, nP), m: nM, v: nV };
  });
  params = kept.p; mS = kept.m; vS = kept.v;
  fp.forEach((x) => x.free()); oldM?.forEach((x) => x.free()); oldV?.forEach((x) => x.free());
  BATCH.forEach((b) => { b.idsMX.free(); b.tgtMX.free(); });
  console.log(`  step ${String(step).padStart(2)}: mean reward ${avg(allR).toFixed(3)}  (${seqs.length} seqs)`);
  if (step % 5 === 0) clearCache();
}

// show sample completions after RL
console.log("\n--- sample completions after RL ---");
for (const prompt of PROMPTS.slice(0, 3)) {
  const c = rollout(params, tok.encode(prompt));
  console.log(`${prompt}${tok.decode(c)}`);
}
