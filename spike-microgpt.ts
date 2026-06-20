// SPIKE: Karpathy's microGPT (karpathy.github.io/2026/02/12/microgpt) — but the
// philosophical mirror-image. His 200-line version hand-rolls a `Value` autograd
// engine to show "I cannot simplify this any further". Here the *same* tiny GPT
// is trained end-to-end (forward + backward + Adam + sampling) with the autograd
// being REAL MLX driven from TypeScript over mlx-c — i.e. the efficient path he
// calls "just efficiency", proven fully drivable from TS.
//
// Faithful to his config: a single GPT-2-style block, 4 heads, char-level names
// tokenizer (27 tokens: '\n' as BOS/EOS + a..z), tied lm_head, ~4k parameters.
// Trains document-by-document on the makemore names corpus, then samples names.
//
//   bun spike-microgpt.ts
//
// Validated against an MLX-Python mirror (reference-microgpt.py): identical init
// (written to /tmp/microgpt-init.f32) + identical data order -> matching loss
// curve (training isn't bit-reproducible — FINDINGS §6 gotcha #8 — so the bar is
// "same start, both converge", as for lora-train).
import { MX, fromI32, fromF32, scalar, evalAll, seed, sample, clearCache, tidy } from "./mx.ts";
import { crossEntropy } from "./loss.ts";
import { valueAndGrad } from "./train.ts";
import { treeFlatten, treeUnflattenLike, type Tree } from "./pytree.ts";

// --- config (his numbers: d=16, 4 heads, 1 layer, block 16 -> ~4k params) ---
const V = 27, D = 16, NH = 4, DH = D / NH, BLOCK = 16, FF = 4 * D;
const EPS = 1e-5, ASCALE = 1 / Math.sqrt(DH);
const STEPS = Number(process.env.STEPS ?? 1000), LR0 = 1e-2, SEED = 1337;

// --- deterministic init (seeded), written to a file so Python loads the SAME
// starting weights. Matrices ~ N(0, 0.02); biases 0; LN weight 1, bias 0. ---
function rng(s0: number) { let s = s0 >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32; }
const rand = rng(SEED);
const randn = () => { const u = Math.max(rand(), 1e-12), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
type Spec = [string, number[], "randn" | "zeros" | "ones"];
const SPEC: Spec[] = [
  ["wte", [V, D], "randn"], ["wpe", [BLOCK, D], "randn"],
  ["ln1w", [D], "ones"], ["ln1b", [D], "zeros"],
  ["wq", [D, D], "randn"], ["bq", [D], "zeros"], ["wk", [D, D], "randn"], ["bk", [D], "zeros"],
  ["wv", [D, D], "randn"], ["bv", [D], "zeros"], ["wo", [D, D], "randn"], ["bo", [D], "zeros"],
  ["ln2w", [D], "ones"], ["ln2b", [D], "zeros"],
  ["wfc", [D, FF], "randn"], ["bfc", [FF], "zeros"], ["wproj", [FF, D], "randn"], ["bproj", [D], "zeros"],
  ["lnfw", [D], "ones"], ["lnfb", [D], "zeros"],
];
const blob: number[] = [];
const p: Record<string, MX> = {};
for (const [name, shape, kind] of SPEC) {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = kind === "randn" ? randn() * 0.02 : kind === "ones" ? 1 : 0;
  blob.push(...data);
  p[name] = fromF32(data, shape);
}
await Bun.write("/tmp/microgpt-init.f32", new Uint8Array(Float32Array.from(blob).buffer));
let params: Tree = p;
const nParams = blob.length;

// --- data: makemore names (32k), fetched once if absent; '\n'=0, a..z=1..26 ---
if (!(await Bun.file("names.txt").exists())) {
  console.log("fetching names.txt (makemore corpus)...");
  const r = await fetch("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt");
  await Bun.write("names.txt", await r.text());
}
const names = (await Bun.file("names.txt").text()).split("\n").map((s) => s.trim().toLowerCase()).filter((s) => /^[a-z]+$/.test(s));
const enc = (name: string) => [0, ...[...name].map((c) => c.charCodeAt(0) - 96)]; // [BOS, chars]

// --- forward: single GPT-2 block, pre-LN, GELU MLP, tied lm_head. ids:[L] -> logits:[L,V] ---
function forward(w: any, ids: MX, L: number): MX {
  const pos = fromI32(Int32Array.from({ length: L }, (_, i) => i), [L]);
  let x = w.wte.takeAxis(ids, 0).add(w.wpe.takeAxis(pos, 0));         // [L,D]
  // attention
  const n1 = x.layerNorm(w.ln1w, w.ln1b, EPS);
  const heads = (t: MX, wt: MX, b: MX) => t.matmul(wt).add(b).reshape([1, L, NH, DH]).transpose([0, 2, 1, 3]); // [1,NH,L,DH]
  const q = heads(n1, w.wq, w.bq), k = heads(n1, w.wk, w.bk), vv = heads(n1, w.wv, w.bv);
  const att = MX.sdpa(q, k, vv, ASCALE, true).transpose([0, 2, 1, 3]).reshape([L, D]);
  x = x.add(att.matmul(w.wo).add(w.bo));
  // mlp
  const n2 = x.layerNorm(w.ln2w, w.ln2b, EPS);
  x = x.add(n2.matmul(w.wfc).add(w.bfc).gelu().matmul(w.wproj).add(w.bproj));
  // tied head
  return x.layerNorm(w.lnfw, w.lnfb, EPS).matmul(w.wte.transpose([1, 0])); // [L,V]
}

// loss over one name: predict next token at every position (targets = inputs<<1, EOS=0)
const lossFn = (w: Tree, ids: MX, tgt: MX, lenMX: MX): MX => {
  const L = lenMX.shape[0];
  return crossEntropy(forward(w, ids, L), tgt.reshape([L, 1]));
};
const vg = valueAndGrad(params, lossFn);

// Inlined Adam so all optimizer state lives in arrays we control and can carry
// across the per-step tidy() — keeping params + moments alive while every forward
// intermediate, gradient and Adam temp is freed deterministically. Without this
// the 1000-step sync loop blows MLX's buffer limit (FINDINGS §6 gotcha #5: the
// FinalizationRegistry never fires inside a tight synchronous loop).
const B1 = 0.9, B2 = 0.999, EPSA = 1e-8;
let mS: MX[] | null = null, vS: MX[] | null = null;

console.log(`=== microGPT in mlx-ts: ${nParams} params (D=${D}, ${NH} heads, 1 layer), real MLX autograd over FFI ===`);
let loss = 0, step0 = 0;
for (let step = 0; step < STEPS; step++) {
  const lr = LR0 * (1 - step / STEPS);                          // linear decay -> 0 (his schedule)
  const bc1 = 1 - B1 ** (step + 1), bc2 = 1 - B2 ** (step + 1);
  const oldP = treeFlatten(params), oldM = mS, oldV = vS;
  const kept = tidy(() => {
    const toks = enc(names[step % names.length]);               // [BOS, c1..cn]
    const L = toks.length;
    const idsMX = fromI32(Int32Array.from(toks), [L]);
    const tgtMX = fromI32(Int32Array.from([...toks.slice(1), 0]), [L]); // next-token targets, EOS=0
    const lenMX = fromI32(new Int32Array(L), [L]);              // shape carrier for L
    const r = vg(params, idsMX, tgtMX, lenMX);
    const fp = treeFlatten(params), fg = treeFlatten(r.grads), nP: MX[] = [], nM: MX[] = [], nV: MX[] = [];
    fp.forEach((pp, i) => {
      const g = fg[i];
      const mi = (mS ? mS[i].mul(scalar(B1)) : scalar(0)).add(g.mul(scalar(1 - B1)));
      const vi = (vS ? vS[i].mul(scalar(B2)) : scalar(0)).add(g.mul(g).mul(scalar(1 - B2)));
      const upd = mi.div(scalar(bc1)).div(vi.div(scalar(bc2)).sqrt().add(scalar(EPSA)));
      nP.push(pp.sub(upd.mul(scalar(lr)))); nM.push(mi); nV.push(vi);
    });
    evalAll(...nP, ...nM, ...nV);
    return { p: treeUnflattenLike(params, nP), m: nM, v: nV, loss: r.loss };
  });
  loss = kept.loss;
  if (step === 0) step0 = loss;
  params = kept.p; mS = kept.m; vS = kept.v;
  oldP.forEach((x) => x.free()); oldM?.forEach((x) => x.free()); oldV?.forEach((x) => x.free()); // drop prior gen
  if (step % 100 === 0) { console.log(`  step ${String(step).padStart(4)}: loss ${loss.toFixed(4)}`); clearCache(); }
}
console.log(`STEP0 loss=${step0.toFixed(4)}`);
console.log(`FINAL loss=${loss.toFixed(4)}`);

// --- sample some names (multinomial, recompute forward each step; seqs are tiny) ---
seed(42);
function gen(w: any, temp = 0.85): string {
  return tidy(() => {
    let ids = [0];
    for (let t = 0; t < BLOCK && ids.length <= BLOCK; t++) {
      const logits = forward(w, fromI32(Int32Array.from(ids), [ids.length]), ids.length);
      const last = logits.takeAxis(fromI32(Int32Array.from([ids.length - 1]), [1]), 0); // [1,V]
      const tok = sample(last, temp, 0, 0).itemU();
      if (tok === 0) break;
      ids.push(tok);
    }
    return ids.slice(1).map((i) => String.fromCharCode(96 + i)).join("");
  });
}
console.log("samples: " + Array.from({ length: 10 }, () => gen(params)).join(" "));
