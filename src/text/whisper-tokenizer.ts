// Whisper tiktoken tokenizer — decode only (token ids -> text), which is all
// transcription needs: the prompt is fixed special-token ids and we never encode
// text. The .tiktoken file is `base64(bytes) rank` per line; tiktoken stores raw
// bytes (no GPT-2 byte<->unicode remap), so decode is concat-bytes + UTF-8.
//   vocab: models/whisper-multilingual.tiktoken (from mlx_whisper assets)

// Special-token ids for the multilingual vocab (base ranks = 50257):
import { readText } from "../io/fs.ts";

export const EOT = 50257;            // <|endoftext|>
export const SOT = 50258;            // <|startoftranscript|>
export const TRANSCRIBE = 50359;
export const TRANSLATE = 50358;
export const NO_TIMESTAMPS = 50363;
export const langToken = (i: number) => SOT + 1 + i; // SOT+1 = en, then by LANGUAGES order
export const EN = langToken(0);

export class WhisperTokenizer {
  private idToBytes: Uint8Array[] = [];
  private dec = new TextDecoder();

  constructor(tiktoken: string) {
    for (const line of tiktoken.split("\n")) {
      if (!line) continue;
      const sp = line.indexOf(" ");
      if (sp < 0) continue;
      const bytesB64 = line.slice(0, sp), rank = Number(line.slice(sp + 1));
      this.idToBytes[rank] = new Uint8Array(Buffer.from(bytesB64, "base64"));
    }
  }

  static async fromFile(path = "models/whisper-multilingual.tiktoken"): Promise<WhisperTokenizer> {
    return new WhisperTokenizer(await readText(path));
  }

  // Concatenate the bytes of all non-special ids, then UTF-8 decode.
  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) { const b = this.idToBytes[id]; if (b) for (const x of b) bytes.push(x); } // specials have no bytes -> skipped
    return this.dec.decode(Uint8Array.from(bytes));
  }
}
