// T5 encoder vs Hugging Face (reference/reference-t5.py), same local weights.
//   bun validation/t5-encode.ts

import { fromU32 } from "../src/core/mx.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { type T5Config, T5Encoder } from "../src/models/t5.ts";

const REPO = "facebook/musicgen-small";
const cfg = (await readJson<any>(await hubFile(REPO, "config.json"))).text_encoder as T5Config;
const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));
const t5 = new T5Encoder(cfg, W);

const ids = fromU32(Uint32Array.from([3, 17, 1029, 55, 1]), [1, 5]);
const h = t5.encode(ids);
const a = h.toF32();
console.log(`  shape  : (${h.shape.join(", ")})`);
console.log(`  first8 : ${Array.from(a.slice(0, 8)).map((v) => v.toFixed(5)).join(", ")}`);
