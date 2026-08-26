// SentencePiece Unigram tokenizer — what T5 uses, and a different algorithm
// from the byte-level BPE in tokenizer.ts.
//
// BPE merges greedily from characters upward. Unigram goes the other way: the
// vocabulary carries a log-probability per token, and encoding is a Viterbi
// search for the segmentation with the highest total. Two segmentations of the
// same string can both be valid; the scores decide.
//
// Pipeline, matching the tokenizer.json this reads:
//   normalize -> WhitespaceSplit -> Metaspace ("▁", prefix) -> Viterbi -> </s>
import { readJson } from "../io/fs.ts";

const SPACE = "▁";           // ▁, SentencePiece's visible space

type Entry = { id: number; score: number };

export class UnigramTokenizer {
  private vocab = new Map<string, Entry>();
  private byId: string[] = [];
  private maxLen = 1;
  private unkId: number;
  private eosId: number;

  private constructor(pieces: [string, number][], unkId: number, eosId: number) {
    pieces.forEach(([tok, score], id) => {
      this.vocab.set(tok, { id, score });
      this.byId[id] = tok;
      if (tok.length > this.maxLen) this.maxLen = tok.length;
    });
    this.unkId = unkId;
    this.eosId = eosId;
  }

  static async fromFile(path: string): Promise<UnigramTokenizer> {
    const j = await readJson<any>(path);
    if (j.model?.type !== "Unigram") throw new Error(`not a Unigram tokenizer: ${j.model?.type}`);
    const eos = (j.added_tokens ?? []).find((t: any) => t.content === "</s>")?.id ?? 1;
    return new UnigramTokenizer(j.model.vocab, j.model.unk_id ?? 0, eos);
  }

  /**
   * Text -> token ids, with `</s>` appended (the TemplateProcessing this
   * checkpoint declares).
   *
   * Normalization is NFKC. The checkpoint declares a `Precompiled` charsmap,
   * which is SentencePiece's own table; NFKC agrees with it on ordinary text
   * and may differ on exotic input. Prompts here are short English phrases.
   */
  encode(text: string, addEos = true): number[] {
    const out: number[] = [];
    for (const word of text.normalize("NFKC").trim().split(/\s+/)) {
      if (word) out.push(...this.viterbi(SPACE + word));
    }
    if (addEos) out.push(this.eosId);
    return out;
  }

  /** Highest-scoring segmentation of one pre-token. */
  private viterbi(piece: string): number[] {
    const n = piece.length;
    // best[i] = best score for piece[0..i), with the token that ends there.
    const best = new Array<{ score: number; from: number; id: number } | null>(n + 1).fill(null);
    best[0] = { score: 0, from: -1, id: -1 };

    for (let i = 0; i < n; i++) {
      const cur = best[i];
      if (!cur) continue;
      const limit = Math.min(n, i + this.maxLen);
      let matched = false;
      for (let j = i + 1; j <= limit; j++) {
        const e = this.vocab.get(piece.slice(i, j));
        if (!e) continue;
        matched = true;
        const score = cur.score + e.score;
        if (!best[j] || score > best[j]!.score) best[j] = { score, from: i, id: e.id };
      }
      // No vocabulary entry starts here: emit <unk> for one character so the
      // search cannot dead-end on an unseen symbol.
      if (!matched) {
        const j = i + 1;
        const score = cur.score - 10;
        if (!best[j] || score > best[j]!.score) best[j] = { score, from: i, id: this.unkId };
      }
    }

    const ids: number[] = [];
    for (let i = n; i > 0; ) {
      const b = best[i];
      if (!b) break;
      ids.push(b.id);
      i = b.from;
    }
    return ids.reverse();
  }

  /** Token ids -> text, undoing the Metaspace substitution. */
  decode(ids: number[]): string {
    return ids.map((i) => this.byId[i] ?? "").join("").replaceAll(SPACE, " ").trim();
  }
}
