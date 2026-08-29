// The end-to-end check for Parakeet: transcribe real speech and compare the
// words.
//
// The numeric check next door pins every stage against PyTorch, but it runs on
// synthetic noise — where the model is maximally uncertain and the transcript is
// meaningless. This is the one that says the decode path produces *language*.
//
// The audio comes from Spark-TTS rather than a committed fixture, which makes
// this a cross-model check: two independently verified models, one speaking and
// one listening. A failure in either shows up here.
//
// Needs no Python. Downloads ~1.7 GB for Spark and ~2.4 GB for Parakeet.
//   bun validation/parakeet-transcribe.ts
import { decodeAudio } from "../src/audio/mel.ts";
import { saveAudio } from "../src/audio/wav.ts";
import { Parakeet } from "../src/models/parakeet-model.ts";
import { SPARK_SAMPLE_RATE, SparkTTS } from "../src/models/spark-tts.ts";

const TEXT = process.argv[2] ?? "The quick brown fox jumps over the lazy dog.";
const OUT = "data/parakeet-check.wav";

const tts = await SparkTTS.fromPretrained();
const audio = await tts.generate(TEXT, { seed: 11 });
const samples = audio.toF32();
audio.free();
await saveAudio(OUT, samples, SPARK_SAMPLE_RATE);
console.log(`spoke ${(samples.length / SPARK_SAMPLE_RATE).toFixed(1)}s -> ${OUT}`);

const asr = await Parakeet.fromPretrained();
const t0 = performance.now();
const heard = (await asr.transcribeFile(OUT)).trim();
const secs = (performance.now() - t0) / 1000;

// Compared on letters and digits: ASR punctuation and casing vary without
// saying anything about whether the words were recognised.
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
const ok = norm(heard) === norm(TEXT);

console.log(`said:  ${JSON.stringify(TEXT)}`);
console.log(`heard: ${JSON.stringify(heard)}`);
console.log(`${(samples.length / SPARK_SAMPLE_RATE / secs).toFixed(0)}x realtime`);

// --- and again, streaming ---------------------------------------------------
// Fed in 100 ms pieces, as a microphone would. The encoder's attention is
// global, so each chunk is encoded with past context and a little future audio;
// the decoder is a transducer and genuinely incremental, emitting tokens that
// are never revised. Streaming therefore costs some accuracy against
// transcribing the whole clip at once, and this is what pins that cost.
const { ParakeetStream } = await import("../src/models/parakeet-stream.ts");
const { ParakeetTokenizer } = await import("../src/text/parakeet-tokenizer.ts");
const { readJson } = await import("../src/io/fs.ts");
const { hubFile } = await import("../src/io/hub.ts");
const { singleFileWeights } = await import("../src/io/loader.ts");
const REPO = "nvidia/parakeet-tdt-0.6b-v3";
const cfg = await readJson<import("../src/models/parakeet.ts").ParakeetConfig>(await hubFile(REPO, "config.json"));
const stream = new ParakeetStream(
  singleFileWeights(await hubFile(REPO, "model.safetensors")),
  cfg,
  await ParakeetTokenizer.fromFile(await hubFile(REPO, "tokenizer.json")),
);
const pcm = await decodeAudio(OUT);
for (let i = 0; i < pcm.length; i += 1600) stream.push(pcm.subarray(i, i + 1600));
stream.flush();
const streamed = stream.text.trim();
console.log(`streamed (${stream.latencySeconds.toFixed(2)}s lag): ${JSON.stringify(streamed)}`);

const streamOk = norm(streamed) === norm(TEXT);
console.log(ok && streamOk ? "transcribe: ok" : "transcribe: MISMATCH");
if (!ok || !streamOk) process.exit(1);
