// Parakeet's vocabulary — decode only.
//
// A transcriber never encodes: the model emits ids and we turn them into text.
// That makes this far smaller than tokenizer.ts, which needs the merge table and
// the byte-level alphabet to go the other way.
//
// SentencePiece-style, so word boundaries are carried *inside* the tokens as
// U+2581 ("▁") rather than by separate space tokens. Decoding is a lookup plus
// that substitution — the "Metaspace" decoder in tokenizer.json.
import { readJson } from "../io/fs.ts";

const METASPACE = "▁";

export class ParakeetTokenizer {
  private idToTok: string[];

  constructor(json: { model: { vocab: [string, number][] | Record<string, number> } }) {
    const v = json.model.vocab;
    const pairs: [string, number][] = Array.isArray(v) ? v : Object.entries(v);
    this.idToTok = [];
    for (const [tok, id] of pairs) this.idToTok[id] = tok;
  }

  static async fromFile(path: string): Promise<ParakeetTokenizer> {
    return new ParakeetTokenizer(await readJson(path));
  }

  /** Token ids -> text. Blanks and other specials are the caller's to filter. */
  decode(ids: number[]): string {
    let s = "";
    for (const id of ids) s += this.idToTok[id] ?? "";
    // prepend_scheme "always": every utterance starts with a metaspace, which
    // would otherwise come back as a leading space.
    return s.replaceAll(METASPACE, " ").replace(/^ /, "");
  }
}
