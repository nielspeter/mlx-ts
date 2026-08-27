// The end-to-end check for Spark-TTS: speak a sentence, then transcribe it back
// with Whisper and compare.
//
// The parity checks either side of this one (spark-lm.ts, bicodec-decode.ts)
// compare against mlx-audio, which pins the numbers but cannot say whether the
// result is *speech*. A pipeline can be numerically faithful stage by stage and
// still produce noise if the stages are joined wrong — the d-vector added at
// the wrong point, say. Whisper is an independent judge of that.
//
// Needs no Python. Downloads ~1.7 GB for Spark and ~150 MB for Whisper.
//   bun validation/spark-roundtrip.ts
import { decodeAudio, loadMelFilters } from "../src/audio/mel.ts";
import { saveAudio } from "../src/audio/wav.ts";
import { SPARK_SAMPLE_RATE, SparkTTS } from "../src/models/spark-tts.ts";
import { loadWhisper } from "../src/models/whisper.ts";
import { WhisperTokenizer } from "../src/text/whisper-tokenizer.ts";

const TEXT = process.argv[2] ?? "MLX runs on the GPU of your Mac.";
const OUT = "data/spark-roundtrip.wav";

const tts = await SparkTTS.fromPretrained();
const audio = await tts.generate(TEXT, { seed: 42 });
const samples = audio.toF32();
audio.free();
await saveAudio(OUT, samples, SPARK_SAMPLE_RATE);
console.log(`spoke ${(samples.length / SPARK_SAMPLE_RATE).toFixed(1)}s -> ${OUT}`);

const whisper = await loadWhisper();
const tok = await WhisperTokenizer.fromFile();
const heard = tok.decode(whisper.transcribe(await decodeAudio(OUT), await loadMelFilters())).trim();

// Compared on letters and digits only: Whisper writes "1, 2, 3" for "one two
// three" and varies on punctuation, neither of which says anything about the
// audio.
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
const ok = norm(heard) === norm(TEXT);
console.log(`said:  ${JSON.stringify(TEXT)}`);
console.log(`heard: ${JSON.stringify(heard)}`);
console.log(ok ? "roundtrip: ok" : "roundtrip: MISMATCH");
if (!ok) process.exit(1);
