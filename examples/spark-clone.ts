// Voice cloning with Spark-TTS: a recording in, the same voice saying something
// else out.
//
//   bun examples/spark-clone.ts ref.wav "Text to speak in that voice."
//
// The speaker is measured from the first 6 seconds of the reference, encoded as
// 32 tokens, and written into the prompt — so the LM generates only the words.
// Any format afconvert reads works; it is resampled to 16 kHz mono.
import { SPARK_SAMPLE_RATE, SparkTTS, playAudio, saveAudio, seed } from "../src/index.ts";

const argv = process.argv.slice(2);
const flag = (name: string, def?: string): string | undefined => {
  const i = argv.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = argv[i + 1];
  argv.splice(i, 2);
  return v;
};

const out = flag("out", "data/cloned.wav") as string;
const temp = Number(flag("temp", "0.8"));
const s = flag("seed");
const play = argv.includes("--play") && (argv.splice(argv.indexOf("--play"), 1), true);
if (s) seed(Number(s));

const rest = argv.filter((a) => !a.startsWith("--"));
const ref = rest[0];
if (!ref) {
  console.error("usage: bun examples/spark-clone.ts <reference.wav> [text]");
  process.exit(1);
}
const text = rest.slice(1).join(" ") ||
  "This sentence was never spoken by the person you are about to hear say it.";

console.log(`voice: ${ref}`);
console.log(`text:  ${text}\n`);

const t0 = performance.now();
const tts = await SparkTTS.fromPretrained();
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const tokens = await tts.speakerTokens(ref);
console.log(`speaker: ${tokens.length} tokens, first 8 [${tokens.slice(0, 8).join(", ")}]`);

const t1 = performance.now();
const audio = await tts.clone(text, ref, {
  temp,
  seed: s ? Number(s) : undefined,
  onToken: (n) => {
    if (n > 0 && n % 25 === 0) {
      process.stdout.write(`\r  ${n} tokens  ${(n * 0.02).toFixed(1)}s of audio   `);
    }
  },
});

const samples = audio.toF32();
audio.free();
await saveAudio(out, samples, SPARK_SAMPLE_RATE);

const secs = samples.length / SPARK_SAMPLE_RATE;
const took = (performance.now() - t1) / 1000;
console.log(`\n\nwrote ${out} — ${secs.toFixed(1)}s of audio in ${took.toFixed(1)}s ` +
            `(${(secs / took).toFixed(1)}x realtime)`);
if (play) await playAudio(out);
