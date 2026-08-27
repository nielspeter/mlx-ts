// TS side of the CLIP text-encoder parity check.
//
// Fixed token ids, so this tests the transformer alone — the tokenizer is a
// separate concern and gets its own check. The oracle is mlx-examples' own
// CLIP port loading the same checkpoint.
//   /tmp/sdvenv/bin/python reference/reference-clip.py && bun validation/clip-encode.ts
import { fromU32 } from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { ClipTextEncoder, type ClipConfig } from "../src/models/clip.ts";

const REPO = "openai/clip-vit-large-patch14";

const cfg = (await readJson<{ text_config: ClipConfig }>(await hubFile(REPO, "config.json"))).text_config;
const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));
const clip = new ClipTextEncoder(cfg, W);

// BOS, "a photo of a cat", EOS, then padding — the shape a real prompt takes.
const ids = [49406, 320, 1125, 539, 320, 2368, 49407, 49407, 49407, 49407];
const h = clip.encode(fromU32(Uint32Array.from(ids), [1, ids.length]));

const f = h.toF32();
let sum = 0, abs = 0;
for (const v of f) { sum += v; abs += Math.abs(v); }
console.log(`clip_text shape=[${h.shape.join(", ")}] mean=${(sum / f.length).toFixed(6)} ` +
            `absmean=${(abs / f.length).toFixed(6)} ` +
            `first4=[${Array.from(f.slice(0, 4)).map((v) => +v.toFixed(4)).join(", ")}]`);
