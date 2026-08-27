// TS side of the CLIP tokenizer parity check, against mlx-examples' own port.
//
// The cases are chosen for the places this can quietly differ: repeated
// whitespace and casing (both normalised away), punctuation runs, digits split
// one at a time, contractions, and a word that must fall back to several merges.
//   /tmp/sdvenv/bin/python reference/reference-clip-tokenizer.py && bun validation/clip-tokenizer.ts
import { hubFile } from "../src/io/hub.ts";
import { ClipTokenizer } from "../src/text/clip-tokenizer.ts";

const REPO = "openai/clip-vit-large-patch14";
const tok = await ClipTokenizer.fromFiles(await hubFile(REPO, "vocab.json"), await hubFile(REPO, "merges.txt"));

const CASES = [
  "a photo of a cat",
  "A  PHOTO   of a CAT",
  "uplifting trance, 138 bpm",
  "an astronaut riding a horse on mars, highly detailed, 4k",
  "hello-world (test) 42",
  "it's a dog's life",
  "cafe naive",
];
for (const c of CASES) console.log(`${JSON.stringify(c)} -> [${tok.encode(c).join(", ")}]`);
