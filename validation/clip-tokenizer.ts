// CLIP's tokenizer against the committed reference ids.
//
// Character-level BPE with a `</w>` word-end mark, not the byte-level kind in
// src/text/tokenizer.ts. The cases cover where it can quietly differ: repeated
// whitespace, casing, punctuation runs, per-digit splitting, contractions, and
// words that need several merges.
//
// Reference: Hugging Face's own CLIPTokenizer (see validation/golden.ts).
//   bun validation/clip-tokenizer.ts
import { hubFile } from "../src/io/hub.ts";
import { ClipTokenizer } from "../src/text/clip-tokenizer.ts";
import { checkIds, loadGolden, verdict } from "./golden.ts";

const g = loadGolden("sd-golden.json").clip_tokenizer;
const tok = await ClipTokenizer.fromFiles(
  await hubFile("openai/clip-vit-large-patch14", "vocab.json"),
  await hubFile("openai/clip-vit-large-patch14", "merges.txt"),
);

for (const [i, text] of (g.cases as string[]).entries()) {
  checkIds(JSON.stringify(text).slice(0, 24), tok.encode(text), g.ids[i]);
}
verdict("CLIP tokenizer");
