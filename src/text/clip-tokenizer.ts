// CLIP's tokenizer — what turns a Stable Diffusion prompt into token ids.
//
// BPE, but not the byte-level kind in src/text/tokenizer.ts: CLIP merges
// characters directly and marks word ends with a "</w>" suffix on the last
// character. The text is lowercased and its whitespace collapsed first, so
// "A  PHOTO" and "a photo" are the same prompt.
//
// Reads vocab.json + merges.txt, as published alongside every CLIP checkpoint.
import { readText } from "../io/fs.ts";

const BOS = "<|startoftext|>";
const EOS = "<|endoftext|>";

// The split CLIP uses: its two specials, English contractions, then runs of
// letters, single digits, and runs of everything else that is not whitespace.
const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+/giu;

export type ClipEncodeOptions = {
  prependBos?: boolean;
  appendEos?: boolean;
  /** Pad with EOS to this length — 77 for Stable Diffusion. */
  padTo?: number;
};

export class ClipTokenizer {
  private vocab: Record<string, number>;
  private ranks: Map<string, number>;
  private cache = new Map<string, string[]>();

  constructor(vocab: Record<string, number>, merges: string[]) {
    this.vocab = vocab;
    this.ranks = new Map();
    // merges.txt starts with a version comment; rank is line order.
    const lines = merges[0]?.startsWith("#version") ? merges.slice(1) : merges;
    lines.forEach((line, i) => {
      const t = line.trim();
      if (t) this.ranks.set(t, i);
    });
  }

  static async fromFiles(vocabPath: string, mergesPath: string): Promise<ClipTokenizer> {
    const vocab = JSON.parse(await readText(vocabPath)) as Record<string, number>;
    return new ClipTokenizer(vocab, (await readText(mergesPath)).split("\n"));
  }

  get bosId(): number { return this.vocab[BOS]; }
  get eosId(): number { return this.vocab[EOS]; }

  /** Merge one pre-token's characters until no ranked pair remains. */
  private bpe(text: string): string[] {
    const hit = this.cache.get(text);
    if (hit) return hit;
    if (text === BOS || text === EOS) return [text];

    // Every character stands alone, and the last one carries the word-end mark.
    const chars = [...text];
    let parts = [...chars.slice(0, -1), `${chars[chars.length - 1]}</w>`];

    while (parts.length > 1) {
      let best: [number, number] | null = null;   // [rank, index]
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== undefined && (best === null || rank < best[0])) best = [rank, i];
      }
      if (best === null) break;

      // Merge *every* occurrence of the winning pair, not just the first.
      const a = parts[best[1]], b = parts[best[1] + 1];
      const merged: string[] = [];
      for (let i = 0; i < parts.length;) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          merged.push(a + b); i += 2;
        } else {
          merged.push(parts[i]); i += 1;
        }
      }
      parts = merged;
    }

    this.cache.set(text, parts);
    return parts;
  }

  encode(text: string, opts: ClipEncodeOptions = {}): number[] {
    const { prependBos = true, appendEos = true, padTo } = opts;
    const clean = text.toLowerCase().replace(/\s+/g, " ").trim();

    const ids: number[] = [];
    if (prependBos) ids.push(this.bosId);
    for (const m of clean.matchAll(PAT)) {
      for (const piece of this.bpe(m[0])) {
        const id = this.vocab[piece];
        if (id === undefined) throw new Error(`clip tokenizer: no id for ${JSON.stringify(piece)}`);
        ids.push(id);
      }
    }
    if (appendEos) ids.push(this.eosId);

    // SD always feeds a fixed 77-token window, padded with EOS.
    if (padTo) {
      while (ids.length < padTo) ids.push(this.eosId);
      if (ids.length > padTo) ids.length = padTo;
    }
    return ids;
  }
}
