// TS side of the BiCodec parity check, against mlx-audio's own port.
//
// Fingerprints all four stages, not just the waveform: quantizer -> speaker
// d-vector -> prenet -> wave generator, and a mismatch at the end says nothing
// about which of the four is wrong. Our tensors are channels-last [B, T, C]
// where mlx-audio's are [B, C, T], so `first4` differs by layout — mean and
// absmean do not.
//
//   /tmp/sdvenv/bin/python reference/reference-bicodec.py && bun validation/bicodec-decode.ts
import { fromI32, type MX } from "../src/index.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { BiCodecPrenet, BiCodecQuantizer, SpeakerDetokenizer, WaveGenerator } from "../src/models/bicodec.ts";

const REPO = "mlx-community/Spark-TTS-0.5B-bf16";
const W = singleFileWeights(await hubFile(REPO, "BiCodec/model.safetensors"));

function fp(tag: string, a: MX) {
  const f = a.toF32();
  let sum = 0, abs = 0;
  for (const v of f) { sum += v; abs += Math.abs(v); }
  console.log(`${tag.padEnd(9)} shape=[${a.shape.join(", ")}] mean=${(sum / f.length).toFixed(6)} ` +
              `absmean=${(abs / f.length).toFixed(6)} ` +
              `first4=[${Array.from(f.slice(0, 4)).map((v) => +v.toFixed(5)).join(", ")}]`);
}

const T = 16;
const semantic = Int32Array.from({ length: T }, (_, i) => (i * 137 + 11) % 8192);
const glob = Array.from({ length: 32 }, (_, i) => (i * 91 + 7) % 4096);

const zq = new BiCodecQuantizer(W).detokenize(fromI32(semantic, [1, T]));
fp("z_q", zq);

const d = new SpeakerDetokenizer(W).detokenize([glob]);
fp("d_vector", d);

const pre = new BiCodecPrenet(W).forward(zq, d);
fp("prenet", pre);

// The d-vector is added back after the prenet — the generator sees the speaker
// twice, once through AdaLayerNorm and once as a plain bias.
const wav = new WaveGenerator(W).forward(pre.add(d.reshape([1, 1, d.shape[1]])));
fp("wav", wav);
