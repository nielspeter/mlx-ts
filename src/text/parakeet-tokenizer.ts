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

/** A word and the span of audio it covers, in seconds. */
export type Word = { text: string; start: number; end: number };

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

  /**
   * Whether this token begins a new word.
   *
   * Words are emitted in pieces — "country" arrives as ▁co + un + tr + y — so a
   * caller showing text incrementally needs this to avoid cutting a word in
   * half at a chunk boundary.
   */
  startsWord(id: number): boolean {
    return (this.idToTok[id] ?? "").startsWith(METASPACE);
  }

  /**
   * Timed tokens -> words with start and end times.
   *
   * A word begins at a token carrying the metaspace and runs until the next one
   * does, so grouping is the same rule `startsWord` uses — only accumulating
   * instead of cutting.
   *
   * The span comes from the model's own duration head: a token emitted at frame
   * f covering d frames occupies [f, f+d). A duration of 0 means another token
   * follows at the same frame, which would give a zero-width word, so every
   * piece is credited at least one frame.
   */
  words(toks: { id: number; frame: number; duration: number }[], secondsPerFrame = 0.08): Word[] {
    const out: Word[] = [];
    for (const t of toks) {
      const piece = this.idToTok[t.id] ?? "";
      const start = t.frame * secondsPerFrame;
      const end = (t.frame + Math.max(t.duration, 1)) * secondsPerFrame;
      if (out.length === 0 || piece.startsWith(METASPACE)) {
        out.push({ text: piece.replaceAll(METASPACE, ""), start, end });
      } else {
        const w = out[out.length - 1];
        w.text += piece;
        w.end = Math.max(w.end, end);
      }
    }
    return out;
  }

  /**
   * Token ids -> text. Blanks and other specials are the caller's to filter.
   *
   * `continuation` keeps a leading space. An utterance starts with a metaspace
   * (prepend_scheme "always") which should not become a leading space — but a
   * *chunk* in the middle of a stream must keep it, or its first word runs into
   * the previous chunk's last one.
   */
  decode(ids: number[], continuation = false): string {
    let s = "";
    for (const id of ids) s += this.idToTok[id] ?? "";
    s = s.replaceAll(METASPACE, " ");
    return continuation ? s : s.replace(/^ /, "");
  }
}
