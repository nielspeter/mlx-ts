// EnCodec decoder in TypeScript vs MLX Python (reference/reference-encodec.py).
// Same weights, same deterministic codes. This is the half of MusicGen that
// turns tokens into audio, and it exercises the custom Metal LSTM kernel.
//   bun validation/encodec-decode.ts

import { fromU32 } from "../src/core/mx.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { type EncodecConfig, EncodecDecoder } from "../src/models/encodec.ts";

const REPO = "mlx-community/encodec-32khz-float32";
const cfg = await readJson<EncodecConfig>(await hubFile(REPO, "config.json"));
const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));
const model = new EncodecDecoder(cfg, W);

const B = 1, K = 4, T = 16;
const data = new Uint32Array(B * K * T);
for (let k = 0; k < K; k++) for (let t = 0; t < T; t++) data[k * T + t] = (t * 131 + k * 977 + 7) % 2048;
const codes = fromU32(data, [B, K, T]);

const audio = model.decode(codes);
const a = audio.toF32();
console.log(`  samples: ${a.length}`);
console.log(`  first8 : ${Array.from(a.slice(0, 8)).map((v) => v.toFixed(6)).join(", ")}`);
console.log(`  sum    : ${Array.from(a).reduce((s, v) => s + v, 0).toFixed(6)}`);
