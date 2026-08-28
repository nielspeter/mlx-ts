// BiCodec's decode path against the committed reference numbers:
// quantizer -> speaker d-vector -> prenet -> wave generator.
//
// All four stages, because a mismatch in the waveform says nothing about which
// of the four caused it. The numbers come from the original PyTorch Spark-TTS
// (see validation/golden.ts), so this needs no Python.
//
// Worth knowing when diffing against mlx-audio: its WNConvTranspose1d passes
// `groups` positionally into conv_transpose1d's `output_padding` slot, so every
// upsampling stage emits one extra sample and 16 frames come out as 5171 rather
// than 320*16 = 5120. PyTorch gives 5120, and so do we.
//   bun validation/bicodec-decode.ts
import { fromI32 } from "../src/index.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { BiCodecPrenet, BiCodecQuantizer, SpeakerDetokenizer, WaveGenerator } from "../src/models/bicodec.ts";
import { check, loadGolden, verdict } from "./golden.ts";

const g = loadGolden("spark-golden.json").bicodec;
const W = singleFileWeights(await hubFile("mlx-community/Spark-TTS-0.5B-bf16", "BiCodec/model.safetensors"));

const zq = new BiCodecQuantizer(W).detokenize(fromI32(Int32Array.from(g.semantic), [1, g.semantic.length]));
check("z_q", zq, g.z_q);                            // channels-last here, first there

const d = new SpeakerDetokenizer(W).detokenize([g.global]);
check("d_vector", d, g.d_vector, true);             // [1, 1024] both sides

const pre = new BiCodecPrenet(W).forward(zq, d);
check("prenet", pre, g.prenet);

// The d-vector is added back after the prenet — the generator sees the speaker
// twice, once through AdaLayerNorm and once as a plain bias.
const wav = new WaveGenerator(W).forward(pre.add(d.reshape([1, 1, d.shape[1]])));
// [1, N, 1] and [1, 1, N] flatten identically, so these are the samples
// themselves rather than a summary of them.
check("wav", wav, g.wav, true);

verdict("BiCodec decode");
