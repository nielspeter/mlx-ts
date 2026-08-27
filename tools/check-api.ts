// Public-API completeness: a module export that never reaches src/index.ts is
// invisible to consumers, however correct it is.
//
// This exists because regex missed the same thing twice. `export { OLMoE }` and
// `export { Qwen3, ... }` on their own lines were both read as "this file
// exports nothing", so OLMoE was omitted and Qwen3 stayed unreachable while
// generateBatch — which takes a Qwen3 as its first argument — was exported.
// Comparing runtime keys cannot be fooled that way.
//
//   bun tools/check-api.ts
import * as idx from "../src/index.ts";

// Deliberately internal: consumers use node:fs directly, and these constants are
// implementation detail of the Whisper front-end. nanogpt.* is namespaced.
const INTERNAL = new Set([
  "readBytes", "readJson", "readText", "writeJson", "writeText",
  "EN", "EOT", "NO_TIMESTAMPS", "SOT", "TRANSCRIBE", "TRANSLATE",
  "HOP", "N_FFT", "N_MELS", "N_SAMPLES",
  "forward", "freeParams", "loadCkpt", "saveCkpt", "generate",
]);

// qwen.ts and gpt2.ts are CLI scripts, not modules — importing them runs them.
const MODULES = [
  "core/mx", "core/pytree", "nn/nn", "nn/optim", "nn/loss", "nn/autograd",
  "text/tokenizer", "text/chat-template", "text/lm", "text/whisper-tokenizer",
  "io/loader", "io/fs", "audio/mel",
  "models/whisper", "models/olmoe", "models/qwen-nn", "models/nanogpt-model",
  "models/vae", "models/clip", "models/unet", "models/diffusion", "models/stable-diffusion",
  "text/clip-tokenizer", "image/png",
  "ffi/index",
];

const pub = new Set(Object.keys(idx));
const stranded: string[] = [];
for (const mod of MODULES) {
  const m = await import(`../src/${mod}.ts`);
  for (const name of Object.keys(m)) {
    if (name !== "default" && !pub.has(name) && !INTERNAL.has(name)) stranded.push(`${mod}:${name}`);
  }
}

if (stranded.length) {
  console.error(`stranded (exported by a module, absent from src/index.ts):\n  ${stranded.join("\n  ")}`);
  process.exit(1);
}
console.log(`public API complete — ${pub.size} exports, nothing stranded`);
