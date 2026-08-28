// Parakeet, assembled: a file path in, a transcript out.
//
// The pieces are checked separately against transformers.ParakeetForTDT
// (validation/parakeet-encode.ts), so this is only the join plus the loading.
import { decodeAudio, parakeetMel } from "../audio/mel.ts";
import { setCacheLimit } from "../core/mx.ts";
import { readJson } from "../io/fs.ts";
import { hubFile } from "../io/hub.ts";
import { singleFileWeights, type Weights } from "../io/loader.ts";
import { ParakeetTokenizer } from "../text/parakeet-tokenizer.ts";
import { decodeGreedy, encode, type ParakeetConfig } from "./parakeet.ts";

const DEFAULT_REPO = "nvidia/parakeet-tdt-0.6b-v3";

export class Parakeet {
  private W: Weights;
  private cfg: ParakeetConfig;
  private tok: ParakeetTokenizer;

  private constructor(W: Weights, cfg: ParakeetConfig, tok: ParakeetTokenizer) {
    this.W = W;
    this.cfg = cfg;
    this.tok = tok;
  }

  static async fromPretrained(repo = DEFAULT_REPO): Promise<Parakeet> {
    const cfg = await readJson<ParakeetConfig>(await hubFile(repo, "config.json"));
    const tok = await ParakeetTokenizer.fromFile(await hubFile(repo, "tokenizer.json"));
    return new Parakeet(singleFileWeights(await hubFile(repo, "model.safetensors")), cfg, tok);
  }

  /** 16 kHz mono float samples -> token ids, blanks already dropped. */
  tokens(pcm: Float32Array, cacheLimitMB = 4096): number[] {
    const prev = setCacheLimit(cacheLimitMB);
    try {
      return decodeGreedy(this.W, this.cfg, encode(this.W, this.cfg.encoder_config, parakeetMel(pcm)));
    } finally {
      setCacheLimit(prev);
    }
  }

  /** 16 kHz mono float samples -> text. */
  transcribe(pcm: Float32Array, cacheLimitMB = 4096): string {
    return this.tok.decode(this.tokens(pcm, cacheLimitMB));
  }

  /** Any format afconvert reads -> text. Resampled to 16 kHz mono. */
  async transcribeFile(path: string, cacheLimitMB = 4096): Promise<string> {
    return this.transcribe(await decodeAudio(path), cacheLimitMB);
  }
}
