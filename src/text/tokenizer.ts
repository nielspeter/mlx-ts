// Pure-TS byte-level BPE tokenizer for Qwen3 (loads the real models/tokenizer.json).
// The one piece genuinely outside MLX — HF-style GPT-2 byte-level BPE.
// Validated bit-exactly against Python `tokenizers` in tok-test.ts.

// GPT-2 bytes<->unicode map: keep printable bytes as themselves, escape the rest
// to U+0100.. so every byte becomes a single safe unicode char.
import { readJson } from "../io/fs.ts";

function byteUnicodeMaps(): [Map<number, string>, Map<string, number>] {
  const bs: number[] = [];
  const add = (lo: number, hi: number) => { for (let i = lo; i <= hi; i++) bs.push(i); };
  add(0x21, 0x7e); add(0xa1, 0xac); add(0xae, 0xff);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const b2c = new Map<number, string>(), c2b = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) { const ch = String.fromCodePoint(cs[i]); b2c.set(bs[i], ch); c2b.set(ch, bs[i]); }
  return [b2c, c2b];
}

// Qwen pretokenization split (cl100k-style). `(?i:...)` -> explicit case classes.
const SPLIT =
  /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

// Original GPT-2 (r50k) split — lowercase contractions, digit runs grouped. This
// is what HF's ByteLevel pretokenizer applies for the `gpt2` tokenizer, so it's
// required for token-exact GPT-2 encoding. Pass it to Tokenizer for GPT-2 models.
export const GPT2_SPLIT =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export class Tokenizer {
  private b2c: Map<number, string>;
  private c2b: Map<string, number>;
  private vocab = new Map<string, number>();
  private idToTok: string[] = [];
  private ranks = new Map<string, number>();
  private specialById = new Map<number, string>();
  private specialRe: RegExp;
  private enc = new TextEncoder();
  private dec = new TextDecoder();
  private split: RegExp;
  private nfc: boolean;

  constructor(json: any, split: RegExp = SPLIT) {
    this.split = split;
    this.nfc = json.normalizer != null;   // GPT-2 has no normalizer; Qwen uses NFC
    [this.b2c, this.c2b] = byteUnicodeMaps();
    for (const [tok, id] of Object.entries<number>(json.model.vocab)) {
      this.vocab.set(tok, id); this.idToTok[id] = tok;
    }
    json.model.merges.forEach((m: [string, string] | string, i: number) => {
      const [a, b] = Array.isArray(m) ? m : m.split(" ");
      this.ranks.set(a + " " + b, i);
    });
    const specials: string[] = [];
    for (const a of json.added_tokens ?? []) {
      this.specialById.set(a.id, a.content);
      this.idToTok[a.id] = a.content;
      specials.push(a.content);
    }
    specials.sort((x, y) => y.length - x.length);
    this.specialRe = new RegExp("(" + specials.map(escapeRe).join("|") + ")");
  }

  static async fromFile(path: string, split: RegExp = SPLIT): Promise<Tokenizer> {
    return new Tokenizer(await readJson(path), split);
  }

  vocabSize(): number { return this.idToTok.length; }   // highest id + 1 (embedding rows needed)

  private bpe(token: string): string[] {
    const word = Array.from(token);
    if (word.length < 2) return word;
    for (;;) {
      let best = Infinity, at = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      word.splice(at, 2, word[at] + word[at + 1]);
    }
    return word;
  }

  private encodeChunk(text: string, out: number[]): void {
    for (const m of text.matchAll(this.split)) {
      let mapped = "";
      for (const byte of this.enc.encode(m[0])) mapped += this.b2c.get(byte);
      for (const piece of this.bpe(mapped)) {
        const id = this.vocab.get(piece);
        if (id === undefined) throw new Error(`OOV piece ${JSON.stringify(piece)}`);
        out.push(id);
      }
    }
  }

  encode(text: string): number[] {
    if (this.nfc) text = text.normalize("NFC");   // Qwen normalizer = NFC; GPT-2 = none
    const out: number[] = [];
    // split off special tokens, which map straight to their id
    for (const part of text.split(this.specialRe)) {
      if (part === "") continue;
      const sid = [...this.specialById].find(([, c]) => c === part)?.[0];
      if (sid !== undefined) out.push(sid);
      else this.encodeChunk(part, out);
    }
    return out;
  }

  decode(ids: number[], skipSpecial = true): string {
    let s = "";
    for (const id of ids) {
      if (this.specialById.has(id)) { if (!skipSpecial) s += this.specialById.get(id); continue; }
      s += this.idToTok[id] ?? "";
    }
    const bytes: number[] = [];
    for (const ch of s) { const b = this.c2b.get(ch); if (b !== undefined) bytes.push(b); }
    return this.dec.decode(Uint8Array.from(bytes));
  }

  // Incremental detokenizer for streaming: a token can split a multi-byte UTF-8
  // char across token boundaries, so per-token decode would emit U+FFFD. A
  // streaming TextDecoder buffers the incomplete byte sequence until the next
  // token completes it. `add(id)` returns the newly-decodable text (may be "");
  // `flush()` drains any trailing buffered bytes at end of stream.
  detokenizer(skipSpecial = true): { add(id: number): string; flush(): string } {
    const dec = new TextDecoder("utf-8");
    const { idToTok, c2b, specialById } = this;
    return {
      add(id: number): string {
        if (specialById.has(id)) { // flush buffered bytes first, then the literal special (boundary)
          return dec.decode() + (skipSpecial ? "" : (specialById.get(id) ?? ""));
        }
        const tok = idToTok[id] ?? "";
        const bytes: number[] = [];
        for (const ch of tok) { const b = c2b.get(ch); if (b !== undefined) bytes.push(b); }
        return dec.decode(Uint8Array.from(bytes), { stream: true });
      },
      flush: (): string => dec.decode(),
    };
  }
}
