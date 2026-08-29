// Parakeet, assembled: a file path in, a transcript out.
//
// The pieces are checked separately against transformers.ParakeetForTDT
// (validation/parakeet-encode.ts), so this is only the join plus the loading.
import { decodeAudio, parakeetMel } from "../audio/mel.ts";
import { type MX, setCacheLimit, tidy } from "../core/mx.ts";
import { readJson } from "../io/fs.ts";
import { hubFile } from "../io/hub.ts";
import { singleFileWeights, type Weights } from "../io/loader.ts";
import { ParakeetTokenizer, type Word } from "../text/parakeet-tokenizer.ts";
import { decodeGreedy, decodeGreedyTimed, encode, type ParakeetConfig } from "./parakeet.ts";

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
    return this.run(pcm, cacheLimitMB, decodeGreedy);
  }

  /**
   * 16 kHz mono float samples -> words with start and end times in seconds.
   *
   * Costs nothing over `transcribe`: the decode loop walks encoder frames
   * anyway, so the timing is already in hand and is simply kept.
   */
  words(pcm: Float32Array, cacheLimitMB = 4096): Word[] {
    return this.tok.words(this.run(pcm, cacheLimitMB, decodeGreedyTimed));
  }

  /**
   * Mel, encode, decode — under one cache limit and one arena.
   *
   * tidy() is what frees the mel and the encoder states. Both are plain scratch
   * and neither decode path owns them, so without it every call leaked an
   * utterance worth of activations.
   */
  private run<T>(
    pcm: Float32Array,
    cacheLimitMB: number,
    decode: (W: Weights, c: ParakeetConfig, e: MX) => T,
  ): T {
    const prev = setCacheLimit(cacheLimitMB);
    try {
      return tidy(() => {
        const mel = parakeetMel(pcm, { nMels: this.cfg.encoder_config.num_mel_bins });
        return decode(this.W, this.cfg, encode(this.W, this.cfg.encoder_config, mel));
      });
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

  /** Any format afconvert reads -> timed words. Resampled to 16 kHz mono. */
  async wordsFromFile(path: string, cacheLimitMB = 4096): Promise<Word[]> {
    return this.words(await decodeAudio(path), cacheLimitMB);
  }
}
