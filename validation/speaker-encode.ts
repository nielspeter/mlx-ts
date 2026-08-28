// The speaker encoder against the committed reference numbers: mel -> ECAPA ->
// perceiver -> FSQ -> 32 tokens.
//
// Every boundary is fingerprinted, because a token mismatch on its own says
// nothing about which of the four stages produced it. The numbers come from the
// original PyTorch Spark-TTS (see validation/golden.ts), so this needs no Python.
//
// Deterministic synthetic audio of exactly the 6 s window the model crops to —
// no audio file, no decoder.
//   bun validation/speaker-encode.ts
import { melSpectrogram } from "../src/audio/mel.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { ecapaTdnn, perceiverResample, SpeakerTokenizer } from "../src/models/speaker.ts";
import { check, checkIds, loadGolden, verdict } from "./golden.ts";

const g = loadGolden("spark-golden.json").speaker;
const W = singleFileWeights(await hubFile("mlx-community/Spark-TTS-0.5B-bf16", "BiCodec/model.safetensors"));

const wav = Float32Array.from({ length: g.samples }, (_, i) => ((i * 131 + 7) % 1009) / 1009 - 0.5);

const mel = melSpectrogram(wav);
check("mel", mel, g.mel, true);                     // [1, T, 128] both sides

const { features, xVector } = ecapaTdnn(W, "speaker_encoder.speaker_encoder", mel);
check("features", features, g.features);            // channels-last here, first here
check("x_vector", xVector, g.x_vector, true);       // [1, 1024] both sides

check("latents", perceiverResample(W, "speaker_encoder.perceiver_sampler", features), g.latents, true);
checkIds("tokens", new SpeakerTokenizer(W).tokenize(mel), g.tokens);

verdict("speaker encoder");
