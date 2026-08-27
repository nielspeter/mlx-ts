// Text to speech with Spark-TTS: a sentence in, a WAV out.
//
//   bun examples/spark-tts.ts "The quick brown fox jumps over the lazy dog."
//   bun examples/spark-tts.ts --gender male --pitch low --speed high "Hello there."
//
// First run downloads ~1.7 GB (a 0.5B Qwen2 plus BiCodec). The voice is
// invented from the gender/pitch/speed controls, so the same text with a
// different seed is a different speaker.
import { SPARK_SAMPLE_RATE, SparkTTS, type Gender, type Level, playAudio, saveAudio, seed } from "../src/index.ts";

const argv = process.argv.slice(2);
const flag = (name: string, def?: string): string | undefined => {
  const i = argv.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = argv[i + 1];
  argv.splice(i, 2);
  return v;
};

const out = flag("out", "data/speech.wav") as string;
const gender = flag("gender", "female") as Gender;
const pitch = flag("pitch", "moderate") as Level;
const speedLevel = flag("speed", "moderate") as Level;
const temp = Number(flag("temp", "0.8"));
const play = argv.includes("--play") && (argv.splice(argv.indexOf("--play"), 1), true);
const s = flag("seed");
if (s) seed(Number(s));

const text = argv.filter((a) => !a.startsWith("--")).join(" ") ||
  "MLX runs on the GPU of your Mac, and this sentence was spoken by TypeScript.";

console.log(`text:  ${text}`);
console.log(`voice: ${gender}, pitch ${pitch}, speed ${speedLevel}\n`);

const t0 = performance.now();
const tts = await SparkTTS.fromPretrained();
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const t1 = performance.now();
const audio = await tts.generate(text, {
  gender, pitch, speed: speedLevel, temp,
  seed: s ? Number(s) : undefined,
  onToken: (n) => {
    if (n > 0 && n % 25 === 0) {
      const rate = n / ((performance.now() - t1) / 1000);
      process.stdout.write(`\r  ${n} tokens  ${(n * 0.02).toFixed(1)}s of audio  ${rate.toFixed(1)} tok/s   `);
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
