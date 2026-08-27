// Text to music: MusicGen, end to end, in TypeScript.
//
//   bun examples/musicgen.ts "trance"
//   bun examples/musicgen.ts "happy rock" --seconds 10 --out rock.wav
//   bun examples/musicgen.ts "trance" --model jasonvassallo/mlx-musicgen-medium
//   bun examples/musicgen.ts "trance" --seed 1234        # reproducible take
//
// First run downloads ~2.6 GB (the model, plus EnCodec's decoder) into
// ~/.cache/mlx-ts. Every run after that is local.
//
// -small is the default because it is the only size Facebook ships as
// safetensors; -medium and -large are PyTorch pickles, unreadable from
// TypeScript. jasonvassallo/mlx-musicgen-{medium,large} are the same weights
// converted — bigger download (~7.4 GB for medium), noticeably better audio.
//
// The pipeline: the prompt is tokenized with SentencePiece Unigram, encoded by
// T5, and projected into the LM's width. The LM then emits four interleaved
// EnCodec codebooks under a delay pattern, guided by the gap between its
// conditional and unconditional predictions. EnCodec turns those codes back
// into a waveform.
import { MusicGen, saveAudio, seed } from "../src/index.ts";

const argv = process.argv.slice(2);
const flag = (name: string, dflt: number) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? Number(argv[i + 1]) : dflt;
};
const str = (name: string, dflt: string) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const prompt = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--"))).join(" ") || "trance";
const seconds = flag("seconds", 5);
const out = str("out", "out.wav");
// Sampling draws from MLX's RNG, so seeding it makes a prompt reproduce its
// take exactly — the difference between "that one was good" and "that one is
// gone". Unseeded runs stay random.
const seedArg = argv.includes("--seed") ? flag("seed", 0) : null;

console.log(`prompt: ${JSON.stringify(prompt)}  (${seconds}s)` +
            (seedArg === null ? "" : `  seed ${seedArg}`));
if (seedArg !== null) seed(seedArg);
const t0 = performance.now();
const model = await MusicGen.fromPretrained(str("model", "facebook/musicgen-small"));
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s\n`);

// EnCodec runs at 50 frames per second, so that is the step budget.
const maxSteps = Math.round(seconds * 50);
const t1 = performance.now();
const audio = model.generate(prompt, {
  maxSteps,
  temp: flag("temp", 1.0),
  topK: flag("topk", 250),
  guidance: flag("guidance", 3.0),
  onStep: (i, n) => {
    if (i % 10 === 0 || i === n) {
      const pct = ((i / n) * 100).toFixed(0).padStart(3);
      const rate = i / ((performance.now() - t1) / 1000);
      process.stdout.write(`\r  ${pct}%  ${i}/${n} frames  ${rate.toFixed(1)} frames/s   `);
    }
  },
});

const samples = audio.toF32();
await saveAudio(out, samples, model.samplingRate);
const secs = samples.length / model.samplingRate;
console.log(`\n\nwrote ${out} — ${secs.toFixed(1)}s of audio at ${model.samplingRate} Hz`);
console.log(`total ${((performance.now() - t1) / 1000).toFixed(1)}s`);
