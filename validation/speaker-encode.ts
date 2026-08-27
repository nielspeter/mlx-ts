// TS side of the speaker-encoder parity check, against mlx-audio.
//
// Fingerprints every boundary — mel, ECAPA features, x-vector, perceiver
// latents, and the 32 tokens — because a token mismatch alone says nothing
// about which stage produced it.
//
// Deterministic synthetic audio of exactly the 6 s window the model crops to,
// so this needs no audio file and no decoder.
//   /tmp/sdvenv/bin/python reference/reference-speaker.py && bun validation/speaker-encode.ts
import { melSpectrogram } from "../src/audio/mel.ts";
import type { MX } from "../src/index.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { ecapaTdnn, perceiverResample, SpeakerTokenizer } from "../src/models/speaker.ts";

const REPO = "mlx-community/Spark-TTS-0.5B-bf16";
const W = singleFileWeights(await hubFile(REPO, "BiCodec/model.safetensors"));

function fp(tag: string, a: MX) {
  const f = a.toF32();
  let sum = 0, abs = 0;
  for (const v of f) { sum += v; abs += Math.abs(v); }
  console.log(`${tag.padEnd(10)} shape=[${a.shape.join(", ")}] mean=${(sum / f.length).toFixed(6)} ` +
              `absmean=${(abs / f.length).toFixed(6)} ` +
              `first4=[${Array.from(f.slice(0, 4)).map((v) => +v.toFixed(5)).join(", ")}]`);
}

const N = (6 * 16000) / 320 * 320;                 // the model's own reference window
const wav = Float32Array.from({ length: N }, (_, i) => ((i * 131 + 7) % 1009) / 1009 - 0.5);

const mel = melSpectrogram(wav);
fp("mel", mel);

const { features, xVector } = ecapaTdnn(W, "speaker_encoder.speaker_encoder", mel);
fp("features", features);
fp("x_vector", xVector);

fp("latents", perceiverResample(W, "speaker_encoder.perceiver_sampler", features));

console.log("tokens:", JSON.stringify(new SpeakerTokenizer(W).tokenize(mel)));
