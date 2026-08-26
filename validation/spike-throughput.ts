// SPIKE: is mlx-ts a serving library or a correctness toy?
// Tests the two throughput unknowns on the real 4-bit Qwen3:
//   1. async-eval overlap — feed the sampled token as a DEVICE array so the next
//      step's graph builds while the GPU runs the current one (mlx-lm's trick).
//   2. lifetime x async composition — does tidy()+eager-free stay correct and
//      bounded while async evals are in flight?
// Pass = async >= sync tok/s, identical tokens, bounded memory.

import { activeMemoryMB, asyncEval, fromI32, MX, sample, tidy } from "../src/core/mx.ts";
import { loadSafetensors } from "../src/io/loader.ts";
import { generate, type KV, Qwen3 } from "../src/models/qwen-nn.ts";
import { Tokenizer } from "../src/text/tokenizer.ts";

const cfg = await Bun.file("models/config-4bit.json").json();
const model = new Qwen3(cfg, loadSafetensors("models/model-q4.safetensors"));
const tok = await Tokenizer.fromFile("models/tokenizer.json");
const prompt = tok.encode("Tell me a story about a brave robot exploring the deep ocean.");
const N = 128;

generate(model, prompt, { max: 8, temp: 0, topP: 0, window: 0 }); // warmup

// --- SYNC baseline (eval + read token every step) ---
const tS = performance.now();
const sync = generate(model, prompt, { max: N, temp: 0, topP: 0, window: 0 });
const syncSecs = (performance.now() - tS) / 1000;

// --- ASYNC overlap ---
// build one step's graph (tidy: keep token + cache, free the rest) WITHOUT eval
function buildStep(idsMX: MX, B: number, L: number, offset: number, cache: KV[]): MX {
  const old = cache.slice();
  const t = tidy(() => {
    const logits = model.logitsLastMX(idsMX.reshape([B, L]), B, L, offset, cache, 0);
    return { t: sample(logits, 0, 0), keep: cache.flatMap((c) => (c ? [c.k, c.v] : [])) };
  }).t;
  for (const c of old) if (c) { c.k.free(); c.v.free(); } // safe: refcount keeps them for the pending eval
  return t; // [B] uint32, graph built, NOT evaluated
}

function genAsync() {
  const cache: KV[] = Array(model.NL).fill(null);
  const flat = () => cache.flatMap((c) => (c ? [c.k, c.v] : []));
  const t0 = performance.now();
  const ids = fromI32(Int32Array.from(prompt), [1, prompt.length]);
  let y = buildStep(ids, 1, prompt.length, 0, cache);
  asyncEval(y, ...flat()); ids.free();
  let pos = prompt.length;
  const gen: number[] = [];
  for (let i = 0; i < N; i++) {
    const yNext = buildStep(y, 1, 1, pos, cache); // builds while GPU runs y
    asyncEval(yNext, ...flat());
    gen.push(y.itemU());                          // sync y (queued last iter)
    y.free(); y = yNext; pos++;
  }
  const secs = (performance.now() - t0) / 1000;
  y.free();
  return { gen, secs, mem: activeMemoryMB() };
}
const a = genAsync();

const same = JSON.stringify(sync.gen) === JSON.stringify(a.gen);
console.log("=== throughput spike: 4-bit Qwen3-0.6B, 128 tokens ===");
console.log(`sync  : ${(N / syncSecs).toFixed(1)} tok/s`);
console.log(`async : ${(N / a.secs).toFixed(1)} tok/s   (overlap speedup ${(syncSecs / a.secs).toFixed(2)}x)`);
console.log(`tokens identical (sync == async): ${same}`);
console.log(`async memory after 128 tok: ${a.mem.toFixed(0)} MB (bounded)`);
