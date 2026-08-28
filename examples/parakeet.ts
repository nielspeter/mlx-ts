// Speech to text with Parakeet TDT — a recording in, a transcript out.
//
//   bun examples/parakeet.ts audio.wav
//
// NVIDIA's FastConformer transducer. Where Whisper sees a fixed 30 s window and
// decodes autoregressively, Parakeet's decoder walks the encoder frames and
// predicts how far to skip at each step — so it does far less work, and streams
// by construction.
//
// First run downloads ~2.4 GB. Any format afconvert reads works.
import { Parakeet } from "../src/index.ts";

const path = process.argv[2];
if (!path) {
  console.error("usage: bun examples/parakeet.ts <audio-file>");
  process.exit(1);
}

const t0 = performance.now();
const model = await Parakeet.fromPretrained();
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const t1 = performance.now();
const text = await model.transcribeFile(path);
const took = (performance.now() - t1) / 1000;

console.log(`\n${text}\n`);
console.log(`transcribed in ${took.toFixed(2)}s`);
