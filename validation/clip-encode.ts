// CLIP's text encoder on fixed ids, so this is the transformer alone — the
// tokenizer is checked separately.
//
// Causal masking is the detail that matters: without it the conditioning is
// plausible but wrong, and an image still comes out.
//
// Reference: transformers' CLIPTextModel (see validation/golden.ts).
//   bun validation/clip-encode.ts
import { fromU32 } from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { type ClipConfig, ClipTextEncoder } from "../src/models/clip.ts";
import { check, loadGolden, verdict } from "./golden.ts";

const REPO = "openai/clip-vit-large-patch14";
const g = loadGolden("sd-golden.json").clip_text;

const cfg = (await readJson<{ text_config: ClipConfig }>(await hubFile(REPO, "config.json"))).text_config;
const clip = new ClipTextEncoder(cfg, singleFileWeights(await hubFile(REPO, "model.safetensors")));

check("clip_text", clip.encode(fromU32(Uint32Array.from(g.ids), [1, g.ids.length])), g, true);
verdict("CLIP text encoder");
