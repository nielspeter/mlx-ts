// base_train (nanochat stage 2): pretrain a GPT from scratch on BPE-tokenized
// text and SAVE a checkpoint — the keystone that lets pretrain -> SFT/inference
// chain. Same engine as spike-nanogpt.ts (multi-layer GPT, AdamW + cosine +
// warmup + grad clip) but over the trained BPE vocab, with a safetensors save.
//   VOCAB-trained first: python3 tok-train.py ; then: bun base-train.ts
import { MX, fromI32, fromF32, scalar, evalAll, seed, sample, clearCache, tidy, saveSafetensors } from "./mx.ts";
import { crossEntropy } from "./loss.ts";
import { valueAndGrad } from "./train.ts";
import { treeFlatten, treeUnflattenLike, type Tree } from "./pytree.ts";
import { loadSafetensors, get } from "./loader.ts";
import { Tokenizer, GPT2_SPLIT } from "./tokenizer.ts";

const NL = +(process.env.N_LAYER ?? 4), NH = +(process.env.N_HEAD ?? 4), D = +(process.env.N_EMBD ?? 128);
const T = +(process.env.BLOCK ?? 64), B = +(process.env.BATCH ?? 16), DH = D / NH, FF = 4 * D;
const ITERS = +(process.env.ITERS ?? 600), WARMUP = 50, EVAL_ITERS = 20;
const LR0 = 3e-3, MIN_LR = 3e-4, WD = 0.1, CLIP = 1.0, B1 = 0.9, B2 = 0.95, EPSA = 1e-8;
const EPS = 1e-5, ASCALE = 1 / Math.sqrt(DH), SEED = 1337;
const CKPT = process.env.CKPT ?? "base-ckpt.safetensors";

// --- data: encode the corpus with the trained BPE tokenizer (TS inference) ---
const tok = await Tokenizer.fromFile("tokenizer-trained.json", GPT2_SPLIT);
const V = tok.vocabSize();
const text = await Bun.file(process.env.CORPUS ?? "input.txt").text();
const t0 = performance.now();
const data = Int32Array.from(tok.encode(text));
console.log(`encoded ${text.length} chars -> ${data.length} BPE tokens (vocab ${V}) in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
const nTrain = Math.floor(0.9 * data.length);
const train = data.subarray(0, nTrain), val = data.subarray(nTrain);

// --- deterministic init (seeded) ---
function rng(s0: number) { let s = s0 >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32; }
const rand = rng(SEED);
const randn = () => { const u = Math.max(rand(), 1e-12), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const RSTD = 0.02 / Math.sqrt(2 * NL);
const mk = (sh: number[], std: number) => { const n = sh.reduce((a, b) => a * b, 1), d = new Float32Array(n); for (let i = 0; i < n; i++) d[i] = randn() * std; return fromF32(d, sh); };
const fill = (sh: number[], val: number) => fromF32(new Float32Array(sh.reduce((a, b) => a * b, 1)).fill(val), sh);
const block = () => ({
  ln1w: fill([D], 1), ln1b: fill([D], 0),
  wq: mk([D, D], 0.02), bq: fill([D], 0), wk: mk([D, D], 0.02), bk: fill([D], 0),
  wv: mk([D, D], 0.02), bv: fill([D], 0), wo: mk([D, D], RSTD), bo: fill([D], 0),
  ln2w: fill([D], 1), ln2b: fill([D], 0),
  wfc: mk([D, FF], 0.02), bfc: fill([FF], 0), wproj: mk([FF, D], RSTD), bproj: fill([D], 0),
});
let params: Tree = { wte: mk([V, D], 0.02), wpe: mk([T, D], 0.02), blocks: Array.from({ length: NL }, block), lnfw: fill([D], 1), lnfb: fill([D], 0) };

function getBatch(src: Int32Array): [MX, MX] {
  const xb = new Int32Array(B * T), yb = new Int32Array(B * T);
  for (let b = 0; b < B; b++) { const ix = Math.floor(rand() * (src.length - T - 1)); for (let t = 0; t < T; t++) { xb[b * T + t] = src[ix + t]; yb[b * T + t] = src[ix + t + 1]; } }
  return [fromI32(xb, [B, T]), fromI32(yb, [B, T])];
}

const posMX = fromI32(Int32Array.from({ length: T }, (_, i) => i), [T]);
function forward(w: any, idx: MX): MX {
  const Bc = idx.shape[0];
  let x = w.wte.takeAxis(idx, 0).add(w.wpe.takeAxis(posMX, 0));
  const heads = (t: MX, wt: MX, b: MX) => t.matmul(wt).add(b).reshape([Bc, T, NH, DH]).transpose([0, 2, 1, 3]);
  for (const blk of w.blocks) {
    const n1 = x.layerNorm(blk.ln1w, blk.ln1b, EPS);
    const q = heads(n1, blk.wq, blk.bq), k = heads(n1, blk.wk, blk.bk), v = heads(n1, blk.wv, blk.bv);
    const att = MX.sdpa(q, k, v, ASCALE, true).transpose([0, 2, 1, 3]).reshape([Bc, T, D]);
    x = x.add(att.matmul(blk.wo).add(blk.bo));
    const n2 = x.layerNorm(blk.ln2w, blk.ln2b, EPS);
    x = x.add(n2.matmul(blk.wfc).add(blk.bfc).gelu().matmul(blk.wproj).add(blk.bproj));
  }
  return x.layerNorm(w.lnfw, w.lnfb, EPS).matmul(w.wte.transpose([1, 0]));
}
const lossMX = (w: Tree, idx: MX, tgt: MX): MX => crossEntropy(forward(w, idx).reshape([B * T, V]), tgt.reshape([B * T, 1]));
const lrAt = (it: number) => it < WARMUP ? LR0 * (it + 1) / WARMUP : MIN_LR + 0.5 * (1 + Math.cos(Math.PI * (it - WARMUP) / (ITERS - WARMUP))) * (LR0 - MIN_LR);

const vg = valueAndGrad(params, lossMX);
const sumsq = (g: MX) => g.mul(g).sumAxes(g.shape.map((_, i) => i), false);
const sB1 = scalar(B1), sB1m = scalar(1 - B1), sB2 = scalar(B2), sB2m = scalar(1 - B2);
let mS: MX[] | null = null, vS: MX[] | null = null;
const estimateVal = (p: Tree): number => tidy(() => { let s = 0; for (let e = 0; e < EVAL_ITERS; e++) { const [i, t] = getBatch(val); s += tidy(() => crossEntropy(forward(p, i).reshape([B * T, V]), t.reshape([B * T, 1])).itemF()); } return s / EVAL_ITERS; });

console.log(`=== base_train: ${(treeFlatten(params).reduce((a, p) => a + p.size, 0) / 1e6).toFixed(2)}M params (${NL}L/${NH}H/D${D}, vocab ${V}) ===`);
let loss = 0;
for (let it = 0; it < ITERS; it++) {
  const lr = lrAt(it), bc1 = 1 - B1 ** (it + 1), bc2 = 1 - B2 ** (it + 1);
  const fp = treeFlatten(params), oldM = mS, oldV = vS;
  const kept = tidy(() => {
    const [idx, tgt] = getBatch(train);
    const r = vg(params, idx, tgt);
    const fg = treeFlatten(r.grads);
    let total = sumsq(fg[0]); for (let i = 1; i < fg.length; i++) total = total.add(sumsq(fg[i]));
    const gnorm = total.sqrt().itemF(), cs = gnorm > CLIP ? CLIP / gnorm : 1, sCs = cs === 1 ? null : scalar(cs);
    const sqBc2 = Math.sqrt(bc2), sAlpha = scalar(lr * sqBc2 / bc1), sEpsHat = scalar(EPSA * sqBc2), sDecay = scalar(1 - lr * WD);
    const nP: MX[] = [], nM: MX[] = [], nV: MX[] = [];
    fp.forEach((pp, i) => {
      const g = sCs ? fg[i].mul(sCs) : fg[i];
      const mi = mS ? mS[i].mul(sB1).add(g.mul(sB1m)) : g.mul(sB1m);
      const vi = vS ? vS[i].mul(sB2).add(g.mul(g).mul(sB2m)) : g.mul(g).mul(sB2m);
      const core = mi.div(vi.sqrt().add(sEpsHat));
      nP.push(pp.shape.length >= 2 ? pp.mul(sDecay).sub(core.mul(sAlpha)) : pp.sub(core.mul(sAlpha))); nM.push(mi); nV.push(vi);
    });
    evalAll(...nP, ...nM, ...nV);
    return { p: treeUnflattenLike(params, nP), m: nM, v: nV, loss: r.loss };
  });
  loss = kept.loss; params = kept.p; mS = kept.m; vS = kept.v;
  fp.forEach((x) => x.free()); oldM?.forEach((x) => x.free()); oldV?.forEach((x) => x.free());
  if (it % 100 === 0) { console.log(`  iter ${String(it).padStart(4)}: train ${loss.toFixed(4)} val ${estimateVal(params).toFixed(4)} (lr ${lr.toExponential(1)})`); clearCache(); }
}
console.log(`FINAL train ${loss.toFixed(4)}  val ${estimateVal(params).toFixed(4)}`);

// --- save checkpoint (flatten the tree to dotted keys) + config ---
const rec: Record<string, MX> = { wte: (params as any).wte, wpe: (params as any).wpe, lnfw: (params as any).lnfw, lnfb: (params as any).lnfb };
(params as any).blocks.forEach((b: any, i: number) => { for (const k of Object.keys(b)) rec[`blocks.${i}.${k}`] = b[k]; });
saveSafetensors(CKPT, rec);
await Bun.write(CKPT.replace(/\.safetensors$/, "") + "-config.json", JSON.stringify({ vocab: V, n_layer: NL, n_head: NH, n_embd: D, block_size: T }, null, 2));
console.log(`saved checkpoint -> ${CKPT} (${Object.keys(rec).length} tensors)`);

// --- verify the checkpoint reloads (round-trip the writer) ---
const back = loadSafetensors(CKPT);
const a = (params as any).wte.copy().toF32(), b = new MX(get(back, "wte") as number).toF32();
const maxErr = a.reduce((m: number, x: number, i: number) => Math.max(m, Math.abs(x - b[i])), 0);
console.log(`CKPT roundtrip: ${maxErr === 0 ? "OK" : "maxErr " + maxErr.toExponential(2)}`);

// --- sample (greedy from BOS-ish) to show it learned BPE-level structure ---
seed(42);
const gen = tidy(() => {
  let ids = tok.encode("\n"); const out: number[] = [];
  for (let i = 0; i < 80 && ids.length < T; i++) {
    const L = ids.length, row = Int32Array.from([...ids, ...new Array(T - L).fill(0)]); // right-pad to T
    const lg = forward(params, fromI32(row, [1, T])).reshape([T, V]);
    const tk = sample(lg.slice([L - 1, 0], [L, V]), 0.8, 0, 0).itemU();            // logits at last real token
    ids.push(tk); out.push(tk);
  }
  return tok.decode(out);
});
console.log("\n--- sample ---\n" + gen + "\n");
