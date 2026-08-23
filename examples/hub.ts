// The short path: a repo id in, tokens out. No curl, no filenames, no knowing
// where weights are supposed to live.
//
//   bun examples/hub.ts "Why is the sky blue?"
//
// The first run downloads ~350 MB to ~/.cache/mlx-ts (MLXTS_CACHE overrides);
// every run after that is local.
import { load, streamText } from "../src/index.ts";

const REPO = process.env.MLXTS_REPO ?? "mlx-community/Qwen3-0.6B-4bit";
const prompt = process.argv.slice(2).join(" ") || "The capital of France is";

let shown = false;
const { model, tokenizer, config } = await load(REPO, {
  onProgress: (done, total) => {
    if (!total || shown) return;
    if (done === total) { console.log(`  downloaded ${(total / 1e6).toFixed(0)} MB`); shown = true; }
  },
});
console.log(`${REPO} — ${config.model_type}, ${config.quantization.bits}-bit, ${config.num_hidden_layers} layers\n`);

process.stdout.write(prompt);
const t0 = performance.now();
let n = 0;
for await (const piece of streamText(model, tokenizer, tokenizer.encode(prompt), { max: 48 })) {
  process.stdout.write(piece); n++;
}
console.log(`\n\n(${n} chunks in ${((performance.now() - t0) / 1000).toFixed(2)}s)`);
