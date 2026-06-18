// Pure-TS byte-level BPE tokenizer for Qwen3 (loads the real tokenizer.json).
// The one piece genuinely outside MLX — HF-style GPT-2 byte-level BPE.
// Validated bit-exactly against Python `tokenizers` in tok-test.ts.

// GPT-2 bytes<->unicode map: keep printable bytes as themselves, escape the rest
// to U+0100.. so every byte becomes a single safe unicode char.
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

// GPT-2 / Qwen pretokenization split. `(?i:...)` -> explicit case classes; `gu` flags.
const SPLIT =
  /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

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

  constructor(json: any) {
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

  static async fromFile(path: string): Promise<Tokenizer> {
    return new Tokenizer(await Bun.file(path).json());
  }

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
    for (const m of text.matchAll(SPLIT)) {
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
    text = text.normalize("NFC");            // tokenizer.json normalizer = NFC
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
}
