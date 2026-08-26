// chat_sft (nanochat stage): load OUR pretrained base checkpoint (base-train.ts)
// and supervised-fine-tune it into a chat model on chat-formatted examples with
// completion-only loss. Saves a chat checkpoint for chat-ckpt.ts. This is the
// pretrain -> SFT handoff, all on our own weights.
//   bun chat-sft.ts        (needs checkpoints/base-ckpt.safetensors + models/tokenizer-trained.json)
import { clearCache, evalAll, fromF32, fromI32, MX, scalar, tidy } from "../src/core/mx.ts";
import { type Tree, treeFlatten, treeUnflattenLike } from "../src/core/pytree.ts";
import { type Cfg, forward, freeParams, generate, loadCkpt, saveCkpt } from "../src/models/nanogpt-model.ts";
import { valueAndGrad } from "../src/nn/autograd.ts";
import { maskedCrossEntropy } from "../src/nn/loss.ts";
import { GPT2_SPLIT, Tokenizer } from "../src/text/tokenizer.ts";

const BASE = process.env.BASE_CKPT ?? "checkpoints/base-ckpt.safetensors";
const OUT = process.env.CHAT_CKPT ?? "checkpoints/chat-ckpt.safetensors";
const ITERS = +(process.env.ITERS ?? 400), LR0 = +(process.env.LR ?? 1e-3), WARMUP = 20;
const B1 = 0.9, B2 = 0.95, EPSA = 1e-8, CLIP = 1.0;

const tok = await Tokenizer.fromFile("models/tokenizer-trained.json", GPT2_SPLIT);
const EOS = tok.encode("<|endoftext|>")[0];
let { params, cfg } = await loadCkpt(BASE);
const V = cfg.vocab, T = cfg.block_size;

// --- chat SFT data (instruction -> response); completion-only loss ---
const PROMPT = (q: string) => `User: ${q}\nAssistant:`;
// build one example: prompt (incl. "Assistant:") + completion, loss on completion only
function build(prompt: string, completion: string) {
  const pIds = tok.encode(prompt);
  let ids = tok.encode(prompt + " " + completion); ids.push(EOS);
  if (ids.length > T) ids = ids.slice(0, T);                   // base wpe only covers block_size
  return { ids, cStart: pIds.length - 1 };                     // cStart = first completion target index
}

// STORIES=<corpus> -> story-aligned SFT: instruction-tune the base on its OWN
// competence ("Tell me a story about {topic}." -> a real story from the corpus),
// so a TinyStories base produces coherent stories on request. Otherwise a small
// QA set (the default; what the offline validate-all check exercises).
const STORIES = process.env.STORIES;
let examples: ReturnType<typeof build>[], demo: string[];
if (STORIES) {
  const STOP = new Set("the and was that with they them then there here have this what your just very into over about after said were they his her him she you are for not but his big had has one".split(" "));
  const topic = (s: string) => {
    const f = new Map<string, number>();
    for (const w of s.toLowerCase().match(/[a-z]{4,}/g) ?? []) if (!STOP.has(w)) f.set(w, (f.get(w) ?? 0) + 1);
    return [...f].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "something";
  };
  const raw = await Bun.file(STORIES).slice(0, +(process.env.SFT_BYTES ?? 8_000_000)).text();
  const stories = raw.split("<|endoftext|>").map((s) => s.trim()).filter((s) => s.length > 120).slice(0, +(process.env.SFT_N ?? 600));
  examples = stories.map((s) => build(`User: Tell me a story about ${topic(s)}.\nAssistant:`, s));
  demo = ["Tell me a story about a cat.", "Tell me a story about a robot.", "Tell me a story about the sea."];
  console.log(`story-aligned SFT: ${examples.length} examples from ${STORIES}`);
} else {
  const DATA = [
    { q: "What is the capital of France?", a: "The capital of France is Paris." },
    { q: "Who are you?", a: "I am a small language model trained with mlx-ts." },
    { q: "What color is the sky?", a: "The sky is blue." },
    { q: "Say hello.", a: "Hello! How can I help you today?" },
    { q: "What is two plus two?", a: "Two plus two is four." },
  ];
  examples = DATA.map(({ q, a }) => build(PROMPT(q), a));
  demo = DATA.slice(0, 3).map((d) => d.q);
}

// BATCHED SFT: BS examples/step, padded to SEQ, masked loss over completion tokens
// only — averaging over many tokens kills the batch-1 variance that made the
// scaled-up chat ramble. (Masked loss is NaN-safe now that crossEntropy uses
// stable log_softmax.)
const BS = +(process.env.DEVBATCH ?? 8);
const SEQ = Math.min(T, Math.max(...examples.map((e) => e.ids.length)));
const lossFn = (p: Tree, idsMX: MX, tgt: MX, mask: MX): MX =>
  maskedCrossEntropy(forward(p, idsMX, cfg).reshape([BS * SEQ, V]), tgt, mask);
const vg = valueAndGrad(params, lossFn);

let bseed = 1234; const rnd = () => (bseed = (bseed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
function makeBatch(): [MX, MX, MX] {                            // padded ids, shifted targets, completion mask
  const xb = new Int32Array(BS * SEQ), tb = new Int32Array(BS * SEQ), mb = new Float32Array(BS * SEQ);
  for (let b = 0; b < BS; b++) {
    const e = examples[Math.floor(rnd() * examples.length)], n = Math.min(e.ids.length, SEQ);
    for (let i = 0; i < n; i++) xb[b * SEQ + i] = e.ids[i];
    for (let i = 0; i < n - 1; i++) { tb[b * SEQ + i] = e.ids[i + 1]; if (i >= e.cStart) mb[b * SEQ + i] = 1; }
  }
  return [fromI32(xb, [BS, SEQ]), fromI32(tb, [BS * SEQ, 1]), fromF32(mb, [BS * SEQ, 1])];
}

const sB1 = scalar(B1), sB1m = scalar(1 - B1), sB2 = scalar(B2), sB2m = scalar(1 - B2);
const sumsq = (g: MX) => g.mul(g).sumAxes(g.shape.map((_, i) => i), false);
let mS: MX[] | null = null, vS: MX[] | null = null;
const lrAt = (it: number) => it < WARMUP ? LR0 * (it + 1) / WARMUP : LR0;
const RTEMP = STORIES ? 0.7 : 0, RMAX = STORIES ? 120 : 32;   // stories: sample; QA: greedy
const reply = (q: string) => tok.decode(generate(params, tok.encode(PROMPT(q)), cfg, EOS, { maxNew: RMAX, temp: RTEMP })).trim();

console.log(`=== chat_sft: ${(treeFlatten(params).reduce((a, p) => a + p.size, 0) / 1e6).toFixed(2)}M params, base=${BASE} ===`);
console.log(`before: ${JSON.stringify(reply(demo[0]))}`);

let loss = 0, step0 = 0;
for (let it = 0; it < ITERS; it++) {
  const lr = lrAt(it), bc1 = 1 - B1 ** (it + 1), bc2 = 1 - B2 ** (it + 1);
  const fp = treeFlatten(params), oldM = mS, oldV = vS;
  const kept = tidy(() => {
    const [idsMX, tgtMX, maskMX] = makeBatch();
    const r = vg(params, idsMX, tgtMX, maskMX);
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
for (const q of demo) console.log(`Q: ${q}\nA: ${reply(q)}\n`);
