// Streaming transcription with Parakeet.
//
// The decoder here is genuinely incremental: it is a transducer, so its state
// carries across chunks and a token, once emitted, is never revised. That is the
// difference from re-transcribing a sliding window, where words can change
// retroactively and every segment boundary is a seam.
//
// The *encoder* is the constraint. Its attention is global — measured, not
// assumed: encoding 3 s of a clip and 6 s of the same clip gives different
// values for the same early frames (up to 1.8e-1 near the prefix edge). So each
// chunk is encoded with a window of past audio for context and a little future
// audio for accuracy:
//
//     [ left context .... | chunk | lookahead ]
//                           ^^^^^ only these frames are decoded
//
// The lookahead is what costs latency, and it earns it. Word error against
// transcribing the whole clip at once, averaged over three recordings:
//
//     chunk   look    avg / worst latency   WER
//     1.04s   0.32s   0.84 / 1.36 s         6.7%
//     1.04s   0.64s   1.16 / 1.68 s         5.3%
//     1.04s   1.04s   1.56 / 2.08 s         2.2%   <- default
//     1.52s   0.64s   1.40 / 2.16 s         2.2%
//     2.00s   0.64s   1.64 / 2.64 s         3.0%
//
// Two settings tie at 2.2%; this one updates every 1.04 s rather than every
// 1.52 s, and a live transcript that moves more often reads as more live.
//
// "avg" is what a speaker mostly feels: a word at the end of a chunk waits only
// the lookahead, one at the start waits the whole chunk as well. Both exclude
// compute, which is ~0.12 s per chunk here.
//
// Left context matters far less than either — 5 s and 10 s score the same — and
// past ~5 s it stops helping at all.
import { parakeetMel } from "../audio/mel.ts";
import { type MX, setCacheLimit, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";
import type { ParakeetTokenizer } from "../text/parakeet-tokenizer.ts";
import {
  encode,
  initialState,
  joint,
  type ParakeetConfig,
  type PredictorState,
  predict,
  projectEncoder,
} from "./parakeet.ts";

/** Mel hop (160) times the encoder's 8x subsampling: one frame is 80 ms. */
export const SAMPLES_PER_FRAME = 1280;

export type StreamOptions = {
  /** Frames decoded per step. Larger is more efficient, and adds latency. */
  chunkFrames?: number;
  /** Past frames the encoder may attend to. Beyond ~5 s this stops helping. */
  leftFrames?: number;
  /** Future frames before decoding a chunk — the accuracy/latency dial. */
  lookaheadFrames?: number;
  /**
   * Minimum frames in the *first* window.
   *
   * At the start there is no left context, so a bare chunk gives the encoder
   * about a second of audio and it hallucinates — one clip returned an empty
   * transcript entirely. Growing only the first window costs latency once, at
   * the beginning of an utterance, and nothing after.
   */
  warmupFrames?: number;
  /**
   * Hold back a trailing partial word until the next chunk completes it.
   *
   * The model emits subwords, so a chunk can end mid-word — "country" arrives as
   * ▁co + un + tr + y. Concatenated the text is right either way, so this
   * matters only if you render each emission separately rather than appending to
   * a growing transcript. It costs the last word of each chunk one extra chunk
   * of delay, which is why it is off by default.
   */
  wholeWords?: boolean;
  cacheLimitMB?: number;
};

export class ParakeetStream {
  private W: Weights;
  private cfg: ParakeetConfig;
  private tok: ParakeetTokenizer;
  private chunk: number;
  private left: number;
  private look: number;
  private warmup: number;
  private wholeWords: boolean;
  private cacheLimitMB: number;
  /** Tokens decoded but not yet returned, because they may be a partial word. */
  private held = 0;

  /** Audio kept for context, and the absolute frame it starts at. */
  private buf = new Float32Array(0);
  private bufFrame = 0;
  /** First frame not yet decoded, absolute. */
  private next = 0;

  private state: PredictorState;
  private prev: number;
  private cached: MX | null = null;
  private ids: number[] = [];
  /**
   * Frames the model asked to skip past the last chunk edge.
   *
   * A duration is the model saying "nothing happens for the next N frames".
   * Dropping that at a boundary makes the next chunk re-examine frames it had
   * already dismissed, and with short chunks it happens constantly — one clip
   * transcribed to nothing at all until this was carried across.
   */
  private carry = 0;

  constructor(W: Weights, cfg: ParakeetConfig, tok: ParakeetTokenizer, o: StreamOptions = {}) {
    this.W = W;
    this.cfg = cfg;
    this.tok = tok;
    this.chunk = o.chunkFrames ?? 13; // 1.04 s
    this.left = o.leftFrames ?? 62; // 5 s
    this.look = o.lookaheadFrames ?? 13; // 1.04 s
    this.warmup = o.warmupFrames ?? 38; // 3 s
    this.wholeWords = o.wholeWords ?? false;
    this.cacheLimitMB = o.cacheLimitMB ?? 2048;
    this.state = initialState(cfg);
    this.prev = cfg.blank_token_id;
  }

  /** The delay between speaking and seeing a word, ignoring compute. */
  get latencySeconds(): number {
    return ((this.chunk + this.look) * SAMPLES_PER_FRAME) / 16000;
  }

  /** Everything transcribed so far. */
  get text(): string {
    return this.tok.decode(this.ids);
  }

  /**
   * Feed 16 kHz mono samples. Returns whatever text became final — often "",
   * since a chunk only decodes once enough lookahead has arrived.
   */
  push(pcm: Float32Array): string {
    const merged = new Float32Array(this.buf.length + pcm.length);
    merged.set(this.buf);
    merged.set(pcm, this.buf.length);
    this.buf = merged;

    const before = this.ids.length;
    const prevLimit = setCacheLimit(this.cacheLimitMB);
    try {
      // Decode only while a full chunk *and* its lookahead have arrived.
      while (this.available() >= this.next + this.chunkAt() + this.look) {
        this.step(this.chunkAt());
      }
    } finally {
      setCacheLimit(prevLimit);
    }
    this.trim();
    return this.take(before);
  }

  /**
   * Text for everything decoded since `from`, minus a trailing partial word when
   * `wholeWords` is set.
   */
  private take(from: number): string {
    if (!this.wholeWords) return this.tok.decode(this.ids.slice(from), from > 0);
    const start = from - this.held;
    // From the last word start onward, the word may still be unfinished.
    let cut = this.ids.length;
    while (cut > start && !this.tok.startsWord(this.ids[cut - 1])) cut--;
    if (cut > start) cut--;
    cut = Math.max(start, cut);
    this.held = this.ids.length - cut;
    return this.tok.decode(this.ids.slice(start, cut), start > 0);
  }

  /**
   * Finish the utterance: decode what is left with whatever lookahead exists.
   *
   * The tail is the least accurate part of any stream — it is the one chunk that
   * never gets its future context.
   */
  flush(): string {
    const before = this.ids.length;
    const prevLimit = setCacheLimit(this.cacheLimitMB);
    try {
      while (this.next < this.available()) {
        this.step(Math.min(this.chunkAt(), this.available() - this.next));
      }
    } finally {
      setCacheLimit(prevLimit);
    }
    // Nothing is held back at the end — the utterance is over.
    const start = before - this.held;
    this.held = 0;
    return this.tok.decode(this.ids.slice(start), start > 0);
  }

  /**
   * Release the predictor state. A stream owns tensors that outlive every
   * step by design — the LSTM state is the whole point of it — so they are
   * not covered by the per-step tidy() and have to be handed back here.
   */
  close(): void {
    for (const layer of this.state) {
      layer.h.free();
      layer.c.free();
    }
    this.state = [];
    this.cached?.free();
    this.cached = null;
  }

  private available(): number {
    return this.bufFrame + Math.floor(this.buf.length / SAMPLES_PER_FRAME);
  }

  /** The first window is grown to `warmup`; every later one is a plain chunk. */
  private chunkAt(): number {
    return this.next === 0 ? Math.max(this.chunk, this.warmup) : this.chunk;
  }

  /** Encode a window around the next chunk and decode that chunk's frames. */
  private step(chunk: number): void {
    const from = Math.max(this.bufFrame, this.next - this.left);
    const to = Math.min(this.available(), this.next + chunk + this.look);
    const win = this.buf.subarray(
      (from - this.bufFrame) * SAMPLES_PER_FRAME,
      (to - this.bufFrame) * SAMPLES_PER_FRAME,
    );
    // The subsampling stem needs a few frames to produce anything at all.
    if (win.length < SAMPLES_PER_FRAME * 4) {
      this.next = to;
      return;
    }

    // Everything the encoder makes is scratch except its projected output: the
    // mel, and 24 blocks of intermediates that were previously left to the GC.
    // Measured before this: 17.5 MB per 30 s of audio, growing linearly with no
    // plateau — about 2 GB an hour, which is the one thing a stream cannot do.
    //
    // nMels comes from the config, not parakeetMel's default: the class should
    // work for any Parakeet checkpoint, not just the 128-bin one.
    const ep = tidy(() => {
      const mel = parakeetMel(win, { nMels: this.cfg.encoder_config.num_mel_bins });
      return projectEncoder(this.W, encode(this.W, this.cfg.encoder_config, mel));
    });
    const lo = this.next - from;
    const hi = Math.min(ep.shape[1], lo + chunk);
    // Deliberately outside the tidy() above: decodeFrames advances the predictor
    // state, which must outlive this chunk. Inside, the arena would adopt it and
    // free the stream's own memory out from under it.
    this.decodeFrames(ep, lo + this.carry, hi);
    ep.free();
    this.next += chunk;
  }

  /** The TDT loop over one chunk, carrying predictor state across calls. */
  private decodeFrames(ep: MX, lo: number, hi: number): void {
    const { vocab_size: V, decoder_hidden_size: H, blank_token_id: BLANK, durations } = this.cfg;
    let t = lo;
    let guard = 0;
    while (t < hi && guard++ < this.cfg.max_symbols_per_step * (hi - lo) + 64) {
      if (this.cached === null) {
        // Same reason as decodeGreedy: without tidy() the LSTM's intermediates
        // accumulate, and a stream is exactly where that matters — it runs for
        // as long as someone keeps talking.
        const r = tidy(() => predict(this.W, this.cfg, this.prev, this.state));
        for (const layer of this.state) {
          layer.h.free();
          layer.c.free();
        }
        this.cached = r.out;
        this.state = r.state;
      }
      const logits = tidy(() =>
        joint(this.W, ep.slice([0, t, 0], [1, t + 1, H]).reshape([1, H]), this.cached as MX).toF32(),
      );

      let token = 0,
        best = -Infinity;
      for (let i = 0; i < V; i++)
        if (logits[i] > best) {
          best = logits[i];
          token = i;
        }
      let dur = 0,
        bestD = -Infinity;
      for (let i = 0; i < durations.length; i++) {
        if (logits[V + i] > bestD) {
          bestD = logits[V + i];
          dur = durations[i];
        }
      }

      if (token === BLANK) {
        if (dur === 0) dur = 1; // never stall on a blank
      } else {
        this.ids.push(token);
        this.prev = token;
        this.cached.free();
        this.cached = null;
      }
      t += dur;
    }
    this.carry = Math.max(0, t - hi);
  }

  /** Drop audio older than the context window, so a long session stays bounded. */
  private trim(): void {
    const keepFrom = Math.max(this.bufFrame, this.next - this.left);
    if (keepFrom <= this.bufFrame) return;
    this.buf = this.buf.slice((keepFrom - this.bufFrame) * SAMPLES_PER_FRAME);
    this.bufFrame = keepFrom;
  }
}
