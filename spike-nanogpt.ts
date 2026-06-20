// SPIKE: nanoGPT (github.com/karpathy/nanoGPT) char-level Shakespeare — the next
// rung up from spike-microgpt.ts. A REAL multi-layer GPT (N blocks, mini-batched
// [B,T], AdamW + cosine LR w/ warmup + global grad clipping) trained from scratch
// on tiny-shakespeare, with the autograd being real MLX over FFI. Proves you can
// train a genuine small GPT — not a toy — in TypeScript driving mlx-c.
//
//   bun spike-nanogpt.ts                  # ~defaults below; fetches input.txt
//   ITERS=300 N_LAYER=4 bun spike-nanogpt.ts
//
// Validated vs reference-nanogpt.py: identical init + identical mini-batches
// (shared via /tmp/nanogpt-*.bin) -> matching step-0 loss; both converge to a
// comparable val loss. Training isn't bit-reproducible (FINDINGS §6 #8), so the
// bar is "same start, both converge" — as for lora-train / microGPT.
import { MX, fromI32, fromF32, scalar, evalAll, seed, sample, clearCache, tidy, dropout } from "./mx.ts";
import { crossEntropy } from "./loss.ts";
import { valueAndGrad } from "./train.ts";
import { treeFlatten, treeUnflattenLike, type Tree } from "./pytree.ts";

// --- config (nanoGPT's shakespeare-char, scaled to run on one Mac in a spike) ---
const NL = +(process.env.N_LAYER ?? 4), NH = +(process.env.N_HEAD ?? 4), D = +(process.env.N_EMBD ?? 128);
const T = +(process.env.BLOCK ?? 64), B = +(process.env.BATCH ?? 32), DH = D / NH, FF = 4 * D;
const ITERS = +(process.env.ITERS ?? 2000), WARMUP = 100, EVAL_ITERS = 40;
const EVAL_INTERVAL = +(process.env.EVAL_INTERVAL ?? 250);   // periodic val eval -> track best (cf. nanoGPT)
const LR0 = 1e-3, MIN_LR = 1e-4, WD = 0.1, CLIP = 1.0, B1 = 0.9, B2 = +(process.env.BETA2 ?? 0.95), EPSA = 1e-8;
const DROP = +(process.env.DROPOUT ?? 0);   // dropout prob (train only); 0 keeps the run exactly reproducible
const EPS = 1e-5, ASCALE = 1 / Math.sqrt(DH), SEED = 1337;

// --- data: tiny-shakespeare, char-level (fetched once) ---
if (!(await Bun.file("input.txt").exists())) {
  console.log("fetching input.txt (tiny-shakespeare)...");
  const r = await fetch("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt");
  await Bun.write("input.txt", await r.text());
}
const text = await Bun.file("input.txt").text();
const chars = [...new Set(text)].sort();                 // deterministic vocab (sorted)
const V = chars.length;
const stoi = new Map(chars.map((c, i) => [c, i] as const));
const data = Int32Array.from([...text].map((c) => stoi.get(c)!));
const nTrain = Math.floor(0.9 * data.length);
const train = data.subarray(0, nTrain), val = data.subarray(nTrain);

// --- deterministic init (seeded) -> flat blob written for the Python oracle ---
function rng(s0: number) { let s = s0 >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32; }
const rand = rng(SEED);
const randn = () => { const u = Math.max(rand(), 1e-12), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const blob: number[] = [];
const RSTD = 0.02 / Math.sqrt(2 * NL);                   // residual-proj init (nanoGPT)
const mk = (shape: number[], std: number): MX => {       // normal(0,std)
  const n = shape.reduce((a, b) => a * b, 1), d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = randn() * std;
  for (const x of d) blob.push(x); return fromF32(d, shape);
};
const fill = (shape: number[], val: number): MX => {     // const init (LN weight/bias, biases)
  const n = shape.reduce((a, b) => a * b, 1), d = new Float32Array(n).fill(val);
  for (const x of d) blob.push(x); return fromF32(d, shape);
};
const block = () => ({
  ln1w: fill([D], 1), ln1b: fill([D], 0),
  wq: mk([D, D], 0.02), bq: fill([D], 0), wk: mk([D, D], 0.02), bk: fill([D], 0),
  wv: mk([D, D], 0.02), bv: fill([D], 0), wo: mk([D, D], RSTD), bo: fill([D], 0),
  ln2w: fill([D], 1), ln2b: fill([D], 0),
  wfc: mk([D, FF], 0.02), bfc: fill([FF], 0), wproj: mk([FF, D], RSTD), bproj: fill([D], 0),
});
// build in treeFlatten order: wte, wpe, blocks..., lnfw, lnfb (Python reads the same order)
let params: Tree = {
  wte: mk([V, D], 0.02), wpe: mk([T, D], 0.02),
  blocks: Array.from({ length: NL }, block),
  lnfw: fill([D], 1), lnfb: fill([D], 0),
};
await Bun.write("/tmp/nanogpt-init.bin", new Uint8Array(Float32Array.from(blob).buffer));
const nParams = blob.length;

// --- precompute mini-batch start indices (seeded) -> file, so Python uses the
// SAME batches. train: [ITERS*B], val-eval: [EVAL_ITERS*B]. ---
const pickIdx = (src: Int32Array, count: number) => Int32Array.from({ length: count }, () => Math.floor(rand() * (src.length - T - 1)));
const trainIdx = pickIdx(train, ITERS * B), valIdx = pickIdx(val, EVAL_ITERS * B);
await Bun.write("/tmp/nanogpt-train-idx.bin", new Uint8Array(trainIdx.buffer.slice(0)));
await Bun.write("/tmp/nanogpt-val-idx.bin", new Uint8Array(valIdx.buffer.slice(0)));

function getBatch(src: Int32Array, idx: Int32Array, off: number): [MX, MX] {
  const xb = new Int32Array(B * T), yb = new Int32Array(B * T);
  for (let b = 0; b < B; b++) { const ix = idx[off + b]; for (let t = 0; t < T; t++) { xb[b * T + t] = src[ix + t]; yb[b * T + t] = src[ix + t + 1]; } }
  return [fromI32(xb, [B, T]), fromI32(yb, [B, T])];
}

// --- forward: N GPT-2 blocks, pre-LN, GELU MLP, tied head. idx:[B,T] -> [B,T,V] ---
const posMX = fromI32(Int32Array.from({ length: T }, (_, i) => i), [T]);
let DSEED = 0;                                                      // per-step base seed for dropout
function forward(w: any, idx: MX, training = false): MX {
  const Bc = idx.shape[0];                                          // batch dim (train: B, gen: 1)
  const drop = (x: MX, site: number) => training && DROP > 0 ? dropout(x, DROP, DSEED + site) : x;
  let x = drop(w.wte.takeAxis(idx, 0).add(w.wpe.takeAxis(posMX, 0)), 1);   // resid dropout on emb
  const heads = (t: MX, wt: MX, b: MX) => t.matmul(wt).add(b).reshape([Bc, T, NH, DH]).transpose([0, 2, 1, 3]);
  w.blocks.forEach((blk: any, i: number) => {
    const n1 = x.layerNorm(blk.ln1w, blk.ln1b, EPS);
    const q = heads(n1, blk.wq, blk.bq), k = heads(n1, blk.wk, blk.bk), v = heads(n1, blk.wv, blk.bv);
    const att = MX.sdpa(q, k, v, ASCALE, true).transpose([0, 2, 1, 3]).reshape([Bc, T, D]);
    x = x.add(drop(att.matmul(blk.wo).add(blk.bo), 10 + i * 2));    // resid dropout on attn out
    const n2 = x.layerNorm(blk.ln2w, blk.ln2b, EPS);
    x = x.add(drop(n2.matmul(blk.wfc).add(blk.bfc).gelu().matmul(blk.wproj).add(blk.bproj), 11 + i * 2));
  });
  return x.layerNorm(w.lnfw, w.lnfb, EPS).matmul(w.wte.transpose([1, 0]));  // tied -> [B,T,V]
}
const lossMX = (w: Tree, idx: MX, tgt: MX): MX => crossEntropy(forward(w, idx, true).reshape([B * T, V]), tgt.reshape([B * T, 1]));

// nanoGPT LR schedule: linear warmup -> cosine decay to MIN_LR
const lrAt = (it: number) => it < WARMUP ? LR0 * (it + 1) / WARMUP
  : MIN_LR + 0.5 * (1 + Math.cos(Math.PI * (it - WARMUP) / (ITERS - WARMUP))) * (LR0 - MIN_LR);

const vg = valueAndGrad(params, lossMX);
const sumsq = (g: MX) => g.mul(g).sumAxes(g.shape.map((_, i) => i), false);  // scalar
let mS: MX[] | null = null, vS: MX[] | null = null;

// Hoisted Adam constants: these never change, so allocate the scalar arrays ONCE
// instead of re-creating ~10 of them per parameter per step (the dominant per-step
// host overhead — hundreds of FFI mlx_array_new_float calls per iteration).
const sB1 = scalar(B1), sB1m = scalar(1 - B1), sB2 = scalar(B2), sB2m = scalar(1 - B2);
const sEPSA = scalar(EPSA), sWD = scalar(WD);

// val loss over the shared eval batches (no dropout); used periodically to track
// the BEST checkpoint — a big model on 1MB of text overfits, so the final-step
// loss overstates error (cf. nanoGPT, which reports its best eval).
const estimateVal = (p: Tree): number => tidy(() => {
  let s = 0;
  for (let e = 0; e < EVAL_ITERS; e++) {
    const [idx, tgt] = getBatch(val, valIdx, e * B);
    s += tidy(() => crossEntropy(forward(p, idx).reshape([B * T, V]), tgt.reshape([B * T, 1])).itemF());
  }
  return s / EVAL_ITERS;
});

console.log(`=== nanoGPT in mlx-ts: ${(nParams / 1e6).toFixed(2)}M params (${NL} layers, ${NH} heads, D=${D}, T=${T}, vocab=${V}) ===`);
let loss = 0, step0 = 0, bestVal = Infinity;
for (let it = 0; it < ITERS; it++) {
  const lr = lrAt(it);
  DSEED = it * 100;                                          // unique dropout seeds per step
  const bc1 = 1 - B1 ** (it + 1), bc2 = 1 - B2 ** (it + 1);
  const fp = treeFlatten(params), oldM = mS, oldV = vS;      // flatten once (reused to free below)
  const kept = tidy(() => {
    const [idx, tgt] = getBatch(train, trainIdx, it * B);
    const r = vg(params, idx, tgt);
    const fg = treeFlatten(r.grads);
    // global grad-norm clip
    let total = sumsq(fg[0]); for (let i = 1; i < fg.length; i++) total = total.add(sumsq(fg[i]));
    const gnorm = total.sqrt().itemF();
    const cs = gnorm > CLIP ? CLIP / gnorm : 1;
    // per-step scalars: created ONCE per step (not once per parameter)
    const sLr = scalar(lr), sBc1 = scalar(bc1), sBc2 = scalar(bc2), sCs = cs === 1 ? null : scalar(cs);
    const nP: MX[] = [], nM: MX[] = [], nV: MX[] = [];
    fp.forEach((pp, i) => {
      const g = sCs ? fg[i].mul(sCs) : fg[i];
      const mi = mS ? mS[i].mul(sB1).add(g.mul(sB1m)) : g.mul(sB1m);       // 0-init: m0=0 -> g*(1-b1)
      const vi = vS ? vS[i].mul(sB2).add(g.mul(g).mul(sB2m)) : g.mul(g).mul(sB2m);
      let upd = mi.div(sBc1).div(vi.div(sBc2).sqrt().add(sEPSA));
      if (pp.shape.length >= 2) upd = upd.add(pp.mul(sWD));   // decoupled weight decay (2D only)
      nP.push(pp.sub(upd.mul(sLr))); nM.push(mi); nV.push(vi);
    });
    evalAll(...nP, ...nM, ...nV);
    return { p: treeUnflattenLike(params, nP), m: nM, v: nV, loss: r.loss };
  });
  loss = kept.loss;
  if (it === 0) step0 = loss;
  params = kept.p; mS = kept.m; vS = kept.v;
  fp.forEach((x) => x.free()); oldM?.forEach((x) => x.free()); oldV?.forEach((x) => x.free());
  if (it % EVAL_INTERVAL === 0 && it > 0) {
    const v = estimateVal(params); bestVal = Math.min(bestVal, v);
    console.log(`  iter ${String(it).padStart(4)}: train ${loss.toFixed(4)} val ${v.toFixed(4)} (best ${bestVal.toFixed(4)}, lr ${lr.toExponential(1)})`);
    clearCache();
  } else if (it % 100 === 0) { console.log(`  iter ${String(it).padStart(4)}: loss ${loss.toFixed(4)} (lr ${lr.toExponential(1)})`); clearCache(); }
}

const valLoss = estimateVal(params);                        // final-step val (no dropout) — matches Python oracle
bestVal = Math.min(bestVal, valLoss);
console.log(`STEP0 loss=${step0.toFixed(4)}`);
console.log(`FINAL train loss=${loss.toFixed(4)}`);
console.log(`VAL loss=${valLoss.toFixed(4)}`);
console.log(`BEST VAL loss=${bestVal.toFixed(4)}`);

// --- sample Shakespeare (autoregressive, context cropped to last T) ---
seed(42);
const sampleText = tidy(() => {
  let ids = [stoi.get("\n") ?? 0];
  const out: number[] = [];
  for (let i = 0; i < 400; i++) {
    const ctx = ids.slice(-T), L = ctx.length;           // forward expects [B=1,T]: right-pad
    const row = Int32Array.from([...ctx, ...new Array(T - L).fill(0)]); // causal mask blocks trailing pads
    const logits = forward(params, fromI32(row, [1, T]));            // [1,T,V]
    const last = logits.reshape([T, V]).takeAxis(fromI32(Int32Array.from([L - 1]), [1]), 0); // pos of last real token
    const tok = sample(last, 0.8, 0, 0).itemU();
    ids.push(tok); out.push(tok);
  }
  return out.map((i) => chars[i]).join("");
});
console.log("\n--- sample ---\n" + sampleText + "\n");
