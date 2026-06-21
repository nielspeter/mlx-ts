// chat_sft (nanochat stage): load OUR pretrained base checkpoint (base-train.ts)
// and supervised-fine-tune it into a chat model on chat-formatted examples with
// completion-only loss. Saves a chat checkpoint for chat-ckpt.ts. This is the
// pretrain -> SFT handoff, all on our own weights.
//   bun chat-sft.ts        (needs base-ckpt.safetensors + tokenizer-trained.json)
import { MX, fromI32, scalar, evalAll, clearCache, tidy } from "./mx.ts";
import { crossEntropy } from "./loss.ts";
import { valueAndGrad } from "./train.ts";
import { treeFlatten, treeUnflattenLike, type Tree } from "./pytree.ts";
import { Tokenizer, GPT2_SPLIT } from "./tokenizer.ts";
import { loadCkpt, saveCkpt, forward, generate, freeParams, type Cfg } from "./nanogpt-model.ts";

const BASE = process.env.BASE_CKPT ?? "base-ckpt.safetensors";
const OUT = process.env.CHAT_CKPT ?? "chat-ckpt.safetensors";
const ITERS = +(process.env.ITERS ?? 400), LR0 = +(process.env.LR ?? 1e-3), WARMUP = 20;
const B1 = 0.9, B2 = 0.95, EPSA = 1e-8, CLIP = 1.0;

const tok = await Tokenizer.fromFile("tokenizer-trained.json", GPT2_SPLIT);
const EOS = tok.encode("<|endoftext|>")[0];
let { params, cfg } = await loadCkpt(BASE);
const V = cfg.vocab, T = cfg.block_size;

// --- chat SFT data (instruction -> response); completion-only loss ---
const PROMPT = (q: string) => `User: ${q}\nAssistant:`;
const DATA = [
  { q: "What is the capital of France?", a: "The capital of France is Paris." },
  { q: "Who are you?", a: "I am a small language model trained with mlx-ts." },
  { q: "What color is the sky?", a: "The sky is blue." },
  { q: "Say hello.", a: "Hello! How can I help you today?" },
  { q: "What is two plus two?", a: "Two plus two is four." },
];
const examples = DATA.map(({ q, a }) => {
  const pIds = tok.encode(PROMPT(q));
  let ids = tok.encode(PROMPT(q) + " " + a); ids.push(EOS);
  if (ids.length > T) ids = ids.slice(0, T);                   // base wpe only covers block_size
  const c = pIds.length - 1, comp = Int32Array.from(ids.slice(c + 1));
  return { idsMX: fromI32(Int32Array.from(ids), [1, ids.length]), L: ids.length, tgt: fromI32(comp, [ids.length - 1 - c, 1]) };
});

const lossFn = (p: Tree, idsMX: MX, tgt: MX, lenMX: MX): MX => {
  const L = lenMX.shape[0], c = L - 1 - tgt.shape[0];
  const logits = forward(p, idsMX, cfg).reshape([L, V]).slice([c, 0], [L - 1, V]);
  return crossEntropy(logits, tgt);
};
const vg = valueAndGrad(params, lossFn);

const sB1 = scalar(B1), sB1m = scalar(1 - B1), sB2 = scalar(B2), sB2m = scalar(1 - B2);
const sumsq = (g: MX) => g.mul(g).sumAxes(g.shape.map((_, i) => i), false);
let mS: MX[] | null = null, vS: MX[] | null = null;
const lrAt = (it: number) => it < WARMUP ? LR0 * (it + 1) / WARMUP : LR0;
const reply = (q: string) => tok.decode(generate(params, tok.encode(PROMPT(q)), cfg, EOS, { maxNew: 32, temp: 0 })).trim();

console.log(`=== chat_sft: ${(treeFlatten(params).reduce((a, p) => a + p.size, 0) / 1e6).toFixed(2)}M params, base=${BASE} ===`);
console.log(`before: Q:${DATA[0].q} -> ${JSON.stringify(reply(DATA[0].q))}`);

let loss = 0, step0 = 0;
for (let it = 0; it < ITERS; it++) {
  const lr = lrAt(it), bc1 = 1 - B1 ** (it + 1), bc2 = 1 - B2 ** (it + 1);
  const ex = examples[it % examples.length], fp = treeFlatten(params), oldM = mS, oldV = vS;
  const lenMX = fromI32(new Int32Array(ex.L), [ex.L]);
  const kept = tidy(() => {
    const r = vg(params, ex.idsMX, ex.tgt, lenMX);
    const fg = treeFlatten(r.grads);
    let total = sumsq(fg[0]); for (let i = 1; i < fg.length; i++) total = total.add(sumsq(fg[i]));
    const gnorm = total.sqrt().itemF(), cs = gnorm > CLIP ? CLIP / gnorm : 1, sCs = cs === 1 ? null : scalar(cs);
    const sqBc2 = Math.sqrt(bc2), sAlpha = scalar(lr * sqBc2 / bc1), sEpsHat = scalar(EPSA * sqBc2);
    const nP: MX[] = [], nM: MX[] = [], nV: MX[] = [];
    fp.forEach((pp, i) => {
      const g = sCs ? fg[i].mul(sCs) : fg[i];
      const mi = mS ? mS[i].mul(sB1).add(g.mul(sB1m)) : g.mul(sB1m);
      const vi = vS ? vS[i].mul(sB2).add(g.mul(g).mul(sB2m)) : g.mul(g).mul(sB2m);
      nP.push(pp.sub(mi.div(vi.sqrt().add(sEpsHat)).mul(sAlpha))); nM.push(mi); nV.push(vi);
    });
    evalAll(...nP, ...nM, ...nV);
    return { p: treeUnflattenLike(params, nP), m: nM, v: nV, loss: r.loss };
  });
  loss = kept.loss; if (it === 0) step0 = loss;
  params = kept.p; mS = kept.m; vS = kept.v;
  fp.forEach((x) => x.free()); oldM?.forEach((x) => x.free()); oldV?.forEach((x) => x.free());
  if (it % 100 === 0) { console.log(`  iter ${String(it).padStart(3)}: loss ${loss.toFixed(4)}`); clearCache(); }
}
console.log(`STEP0 loss=${step0.toFixed(4)}  FINAL loss=${loss.toFixed(4)}`);

await saveCkpt(OUT, params, cfg);
console.log(`saved chat checkpoint -> ${OUT}`);
console.log("\n--- after chat_sft ---");
for (const { q } of DATA.slice(0, 3)) console.log(`Q: ${q}\nA: ${reply(q)}\n`);
