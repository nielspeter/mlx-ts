// The end-to-end check for voice cloning: clone a voice, then verify the output
// is that voice and says the right words.
//
// Two independent judges, neither of which is the quantity being copied:
//   - ECAPA's x-vector, the speaker-verification embedding. The 32 tokens come
//     from the perceiver/FSQ path, so scoring with the x-vector is not circular.
//   - Whisper, for the words.
//
// The reference is Spark's own output, so this needs no audio fixture: generate
// a described voice, clone *it*, and check the clone matches. A cloning bug that
// still produces speech — a scrambled token order, say — lands near the
// unrelated-voice floor rather than near 1.
//
// Needs no Python.
//   bun validation/spark-clone.ts
import { decodeAudio, melSpectrogram } from "../src/audio/mel.ts";
import { saveAudio } from "../src/audio/wav.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { referenceClip, SpeakerTokenizer, volumeNormalize } from "../src/models/speaker.ts";
import { SPARK_SAMPLE_RATE, SparkTTS } from "../src/models/spark-tts.ts";
import { loadWhisper } from "../src/models/whisper.ts";
import { WhisperTokenizer } from "../src/text/whisper-tokenizer.ts";

const REF = "data/clone-reference.wav";
const OUT = "data/clone-output.wav";
const TEXT = "The quick brown fox jumps over the lazy dog.";

const tts = await SparkTTS.fromPretrained();

// A reference voice, and a deliberately different one as the similarity floor.
for (const [path, gender, seed] of [[REF, "male", 11], ["data/clone-other.wav", "female", 22]] as const) {
  const a = await tts.generate("This recording is the reference voice for the cloning check.",
    { gender, pitch: gender === "male" ? "low" : "high", seed });
  await saveAudio(path, a.toF32(), SPARK_SAMPLE_RATE);
  a.free();
}

const tokens = await tts.speakerTokens(REF);
console.log(`speaker tokens: ${tokens.length} (first 8: ${tokens.slice(0, 8).join(", ")})`);
if (tokens.length !== 32) throw new Error(`expected 32 speaker tokens, got ${tokens.length}`);

const cloned = await tts.clone(TEXT, REF, { seed: 7 });
await saveAudio(OUT, cloned.toF32(), SPARK_SAMPLE_RATE);
cloned.free();

// --- is it the same voice? ---
const W = singleFileWeights(await hubFile("mlx-community/Spark-TTS-0.5B-bf16", "BiCodec/model.safetensors"));
const enc = new SpeakerTokenizer(W);
async function xvec(p: string): Promise<number[]> {
  const mel = melSpectrogram(referenceClip(volumeNormalize(await decodeAudio(p))));
  const v = enc.xVector(mel).toF32();
  mel.free();
  const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  return v.map((x) => x / n);
}
const cos = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

const ref = await xvec(REF);
const same = cos(ref, await xvec(OUT));
const other = cos(ref, await xvec("data/clone-other.wav"));
console.log(`similarity to the reference: ${same.toFixed(4)}`);
console.log(`similarity to another voice: ${other.toFixed(4)}   (the floor)`);

// --- does it say the right thing? ---
const whisper = await loadWhisper();
const wtok = await WhisperTokenizer.fromFile();
const heard = wtok.decode(whisper.transcribe(await decodeAudio(OUT), await import("../src/audio/mel.ts").then((m) => m.loadMelFilters()))).trim();
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
console.log(`heard: ${JSON.stringify(heard)}`);

// A clone should sit far above the floor and comfortably above chance. 0.6 is
// well below what a working clone scores (~0.9) and well above a different
// voice, so it separates "cloned" from "generated some other voice".
const ok = same > 0.6 && same > other + 0.2 && norm(heard) === norm(TEXT);
console.log(ok ? "clone: ok" : "clone: MISMATCH");
if (!ok) process.exit(1);
