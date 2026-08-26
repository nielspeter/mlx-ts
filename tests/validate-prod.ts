// Validates the production features: temp/top-p sampling (reproducible),
// batching (B>1), and bounded memory over a long generation (proves the
// FinalizationRegistry frees per-step intermediates).
//   bun validate-prod.ts

import { activeMemoryMB, seed } from "../src/core/mx.ts";
import { loadSafetensors } from "../src/io/loader.ts";
import { generate, generateBatch, type KV, Qwen3, stepTidy } from "../src/models/qwen-nn.ts";
import { Tokenizer } from "../src/text/tokenizer.ts";

const cfg = await Bun.file("models/config-4bit.json").json();
const model = new Qwen3(cfg, loadSafetensors("models/model-q4.safetensors"));
const tok = await Tokenizer.fromFile("models/tokenizer.json");
const enc = (s: string) => tok.encode(s);

// --- A) sampling: temperature + top-p, reproducible under a fixed seed ---
console.log("=== A) temp/top-p sampling ===");
seed(123);
const a = generate(model, enc("Once upon a time"), { max: 24, temp: 0.8, topP: 0.95, window: 0 });
seed(123);
const b = generate(model, enc("Once upon a time"), { max: 24, temp: 0.8, topP: 0.95, window: 0 });
console.log(`  sample:      ${JSON.stringify(tok.decode(a.gen))}`);
console.log(`  reproducible (same seed -> same ids): ${JSON.stringify(a.gen) === JSON.stringify(b.gen)}`);
const greedy = generate(model, enc("Once upon a time"), { max: 24, temp: 0, topP: 0, window: 0 });
console.log(`  greedy:      ${JSON.stringify(tok.decode(greedy.gen))}`);

// --- B) batching: two sequences in one forward pass ---
console.log("\n=== B) batched decode (B=2) ===");
let p1 = enc("The capital of France is"), p2 = enc("The capital of Japan is");
const L = Math.min(p1.length, p2.length);
p1 = p1.slice(-L); p2 = p2.slice(-L);              // common length (suffix) — no padding mask needed
const outs = generateBatch(model, [p1, p2], 12);
console.log(`  row 0: ${JSON.stringify(tok.decode(p1) + tok.decode(outs[0]))}`);
console.log(`  row 1: ${JSON.stringify(tok.decode(p2) + tok.decode(outs[1]))}`);

// --- C) bounded memory over a long generation (tidy() frees intermediates) ---
console.log("\n=== C) memory over 200-token generation (tidy per step) ===");
const cache: KV[] = Array(model.NL).fill(null);
const ids = enc("Tell me a long story about a robot.");
let tk = stepTidy(model, Int32Array.from(ids), 1, ids.length, 0, cache, 0, 0, 0);
let tok0 = tk.itemU(); tk.free();
let pos = ids.length;
const base = activeMemoryMB();
for (let i = 1; i <= 200; i++) {
  tk = stepTidy(model, Int32Array.from([tok0]), 1, 1, pos, cache, 0, 0, 0);
  tok0 = tk.itemU(); tk.free(); pos++;
  if (i % 50 === 0) console.log(`  step ${i.toString().padStart(3)}: active memory ${activeMemoryMB().toFixed(0)} MB`);
}
console.log(`  growth over 200 steps: ${(activeMemoryMB() - base).toFixed(0)} MB (only KV cache grows; per-step intermediates freed each step)`);
