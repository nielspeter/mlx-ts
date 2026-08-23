// Whisper (speech-to-text) over mlx-c — encoder + decoder, mirroring
// mlx_whisper/whisper.py exactly. Conv1d stem -> sinusoidal pos -> bidirectional
// encoder; token+learned-pos -> causal decoder with cross-attention to the audio
// features -> tied-embedding logits. Runs in fp16 like the reference.
//   weights: models/whisper-tiny.safetensors (converted from mlx-community/whisper-tiny)

import { MX, fromF32, fromI32, scalar, sample, tidy, evalAll } from "../core/mx.ts";
import { loadSafetensors, get } from "../io/loader.ts";
import { decodeAudio, padOrTrim, loadMelFilters, logMel } from "../../examples/audio.ts";
import { WhisperTokenizer } from "../text/whisper-tokenizer.ts";
import { readJson } from "../io/fs.ts";

const FP16 = 9, EPS = 1e-5;

// Whisper special-token ids depend on the language count (99 for v1/v2 -> 51865
// vocab, 100 for v3 -> 51866). Byte-BPE base is 50257; specials follow in order:
// eot, sot, [nLang langs], translate, transcribe, +3, notimestamps, [1501 ts].
const BASE = 50257;
function specials(nVocab: number) {
  const nLang = nVocab - BASE - 1509; // 8 scalar specials + 1501 timestamps
  const sot = BASE + 1;
  return { eot: BASE, sot, langStart: sot + 1, translate: BASE + 2 + nLang, transcribe: BASE + 3 + nLang, noTimestamps: BASE + 7 + nLang, nLang };
}
type Specials = ReturnType<typeof specials>;

type Lin = { wt: MX; b: MX | null };
// per-layer decode cache: growing self-attn k/v + fixed cross-attn k/v (from audio).
type WCache = { sk: MX | null; sv: MX | null; ck: MX | null; cv: MX | null };

// sinusoids(length, channels) — Whisper's fixed encoder positional embedding.
function sinusoids(length: number, channels: number): MX {
  const half = channels / 2;
  const inc = Math.log(10000) / (half - 1);
  const out = new Float32Array(length * channels);
  for (let t = 0; t < length; t++)
    for (let i = 0; i < half; i++) {
      const v = t * Math.exp(-inc * i);
      out[t * channels + i] = Math.sin(v);
      out[t * channels + half + i] = Math.cos(v);
    }
  return fromF32(out, [length, channels]).astype(FP16);
}

export class Whisper {
  nAudioHead: number; nTextHead: number; nTextLayer: number; D: number; sp: Specials;
  conv1w: MX; conv1b: MX; conv2w: MX; conv2b: MX; encPos: MX; lnPostW: MX; lnPostB: MX;
  enc: any[]; tokEmb: MX; tokEmbT: MX; textPos: MX; dec: any[]; lnW: MX; lnB: MX;

  constructor(cfg: any, w: number) {
    this.nAudioHead = cfg.n_audio_head; this.nTextHead = cfg.n_text_head;
    this.nTextLayer = cfg.n_text_layer; this.D = cfg.n_audio_state; this.sp = specials(cfg.n_vocab);
    const W = (n: string) => new MX(get(w, n));
    const lin = (n: string, bias = true): Lin => ({ wt: W(`${n}.weight`).transpose([1, 0]), b: bias ? W(`${n}.bias`) : null });
    const ln = (n: string) => ({ w: W(`${n}.weight`), b: W(`${n}.bias`) });

    // encoder
    this.conv1w = W("encoder.conv1.weight"); this.conv1b = W("encoder.conv1.bias");
    this.conv2w = W("encoder.conv2.weight"); this.conv2b = W("encoder.conv2.bias");
    this.encPos = sinusoids(cfg.n_audio_ctx, this.D);
    const lp = ln("encoder.ln_post"); this.lnPostW = lp.w; this.lnPostB = lp.b;
    this.enc = Array.from({ length: cfg.n_audio_layer }, (_, i) => {
      const p = `encoder.blocks.${i}`;
      return {
        attnLn: ln(`${p}.attn_ln`), q: lin(`${p}.attn.query`), k: lin(`${p}.attn.key`, false), v: lin(`${p}.attn.value`), o: lin(`${p}.attn.out`),
        mlpLn: ln(`${p}.mlp_ln`), mlp1: lin(`${p}.mlp1`), mlp2: lin(`${p}.mlp2`),
      };
    });

    // decoder
    this.tokEmb = W("decoder.token_embedding.weight"); this.tokEmbT = this.tokEmb.transpose([1, 0]);
    this.textPos = W("decoder.positional_embedding");
    const dl = ln("decoder.ln"); this.lnW = dl.w; this.lnB = dl.b;
    this.dec = Array.from({ length: cfg.n_text_layer }, (_, i) => {
      const p = `decoder.blocks.${i}`;
      return {
        attnLn: ln(`${p}.attn_ln`), q: lin(`${p}.attn.query`), k: lin(`${p}.attn.key`, false), v: lin(`${p}.attn.value`), o: lin(`${p}.attn.out`),
        crossLn: ln(`${p}.cross_attn_ln`), cq: lin(`${p}.cross_attn.query`), ck: lin(`${p}.cross_attn.key`, false), cv: lin(`${p}.cross_attn.value`), co: lin(`${p}.cross_attn.out`),
        mlpLn: ln(`${p}.mlp_ln`), mlp1: lin(`${p}.mlp1`), mlp2: lin(`${p}.mlp2`),
      };
    });
  }

  private lf(x: MX, p: Lin): MX { const y = x.matmul(p.wt); return p.b ? y.add(p.b) : y; }

  // q from x, k/v from src (= x for self-attn, = audio features for cross).
  private mha(x: MX, src: MX, q: Lin, k: Lin, v: Lin, o: Lin, nHead: number, causal: boolean): MX {
    const B = x.shape[0], Lq = x.shape[1], D = x.shape[2], Dh = D / nHead, Lk = src.shape[1];
    const qh = this.lf(x, q).reshape([B, Lq, nHead, Dh]).transpose([0, 2, 1, 3]);
    const kh = this.lf(src, k).reshape([B, Lk, nHead, Dh]).transpose([0, 2, 1, 3]);
    const vh = this.lf(src, v).reshape([B, Lk, nHead, Dh]).transpose([0, 2, 1, 3]);
    const a = MX.sdpa(qh, kh, vh, Dh ** -0.5, causal).transpose([0, 2, 1, 3]).reshape([B, Lq, D]);
    return this.lf(a, o);
  }

  // mel: [1, n_audio_ctx*2, n_mels] (fp16) -> audio features [1, n_audio_ctx, D]
  encoder(mel: MX): MX {
    let x = mel.conv1d(this.conv1w, 1, 1).add(this.conv1b).gelu();   // [1, 3000, D]
    x = x.conv1d(this.conv2w, 2, 1).add(this.conv2b).gelu();         // [1, 1500, D]
    x = x.add(this.encPos);
    for (const b of this.enc) {
      const n = x.layerNorm(b.attnLn.w, b.attnLn.b, EPS);
      x = x.add(this.mha(n, n, b.q, b.k, b.v, b.o, this.nAudioHead, false));   // self-attn: q,k,v all from n
      x = x.add(this.lf(this.lf(x.layerNorm(b.mlpLn.w, b.mlpLn.b, EPS), b.mlp1).gelu(), b.mlp2));
    }
    return x.layerNorm(this.lnPostW, this.lnPostB, EPS);
  }

  // tokens: [1, T] int32, audio: encoder output -> logits [1, T, n_vocab]
  decoder(tokens: MX, audio: MX): MX {
    const T = tokens.shape[1];
    const pos = this.textPos.takeAxis(fromI32(Int32Array.from({ length: T }, (_, i) => i), [T]), 0);
    let x = this.tokEmb.takeAxis(tokens, 0).add(pos);               // [1, T, D]
    for (const b of this.dec) {
      const n = x.layerNorm(b.attnLn.w, b.attnLn.b, EPS);
      x = x.add(this.mha(n, n, b.q, b.k, b.v, b.o, this.nTextHead, true));                                  // causal self-attn
      x = x.add(this.mha(x.layerNorm(b.crossLn.w, b.crossLn.b, EPS), audio, b.cq, b.ck, b.cv, b.co, this.nTextHead, false)); // cross-attn to audio
      x = x.add(this.lf(this.lf(x.layerNorm(b.mlpLn.w, b.mlpLn.b, EPS), b.mlp1).gelu(), b.mlp2));
    }
    x = x.layerNorm(this.lnW, this.lnB, EPS);
    return x.matmul(this.tokEmbT);                                  // tied-embedding logits
  }

  private heads(t: MX, H: number, Dh: number): MX { const B = t.shape[0], L = t.shape[1]; return t.reshape([B, L, H, Dh]).transpose([0, 2, 1, 3]); }

  // One cached decode call: process `tokens` ([1,T]) at sequence `offset`, updating
  // `cache` in place. Self-attn k/v are concatenated onto the cache; cross-attn k/v
  // are computed once from `audio` and reused. Returns logits [1, T, n_vocab].
  decoderStep(tokens: Int32Array, T: number, offset: number, audio: MX, cache: WCache[]): MX {
    const B = 1, D = this.D, H = this.nTextHead, Dh = D / H;
    const posIdx = fromI32(Int32Array.from({ length: T }, (_, i) => offset + i), [T]);
    let x = this.tokEmb.takeAxis(fromI32(tokens, [1, T]), 0).add(this.textPos.takeAxis(posIdx, 0));
    for (let i = 0; i < this.dec.length; i++) {
      const b = this.dec[i], c = cache[i];
      // self-attention (causal); append new k/v to the cache
      const n = x.layerNorm(b.attnLn.w, b.attnLn.b, EPS);
      const q = this.heads(this.lf(n, b.q), H, Dh);
      let k = this.heads(this.lf(n, b.k), H, Dh), v = this.heads(this.lf(n, b.v), H, Dh);
      if (c.sk) { k = c.sk.concat(k, 2); v = c.sv!.concat(v, 2); }
      c.sk = k; c.sv = v;
      const sa = MX.sdpa(q, k, v, Dh ** -0.5, T > 1).transpose([0, 2, 1, 3]).reshape([B, T, D]);
      x = x.add(this.lf(sa, b.o));
      // cross-attention to audio; k/v computed once and cached
      const cn = x.layerNorm(b.crossLn.w, b.crossLn.b, EPS);
      const cq = this.heads(this.lf(cn, b.cq), H, Dh);
      if (!c.ck) { c.ck = this.heads(this.lf(audio, b.ck), H, Dh); c.cv = this.heads(this.lf(audio, b.cv), H, Dh); }
      const ca = MX.sdpa(cq, c.ck, c.cv!, Dh ** -0.5, false).transpose([0, 2, 1, 3]).reshape([B, T, D]);
      x = x.add(this.lf(ca, b.co));
      x = x.add(this.lf(this.lf(x.layerNorm(b.mlpLn.w, b.mlpLn.b, EPS), b.mlp1).gelu(), b.mlp2));
    }
    return x.layerNorm(this.lnW, this.lnB, EPS).matmul(this.tokEmbT);
  }

  // Detect the spoken language: one decode step on [sot] over the audio, argmax
  // over the language-token range. Returns the language token id.
  private detectLanguage(audio: MX): number {
    const cache: WCache[] = Array.from({ length: this.dec.length }, () => ({ sk: null, sv: null, ck: null, cv: null }));
    const lg = tidy(() => this.decoderStep(Int32Array.from([this.sp.sot]), 1, 0, audio, cache).reshape([this.tokEmb.shape[0]]));
    const v = lg.copy().toF32(); lg.free();
    let best = this.sp.langStart, bv = -Infinity;
    for (let i = 0; i < this.sp.nLang; i++) { const x = v[this.sp.langStart + i]; if (x > bv) { bv = x; best = this.sp.langStart + i; } }
    return best;
  }

  // Greedy transcription of a 16 kHz mono PCM clip (<= 30 s) -> token ids, with a
  // KV cache (cross-attn k/v built once, self-attn k/v grow by one per step).
  // Auto-detects the language unless opts.langToken / opts.prompt is given.
  transcribe(pcm: Float32Array, filtersT: MX, opts: { max?: number; prompt?: number[]; langToken?: number } = {}): number[] {
    const sp = this.sp;
    const { mel, F, nMels } = logMel(padOrTrim(pcm), { filtersT, dropLast: true });
    const audio = this.encoder(fromF32(mel, [1, F, nMels]).astype(9)); // [1, 1500, D]
    const lang = opts.langToken ?? this.detectLanguage(audio);
    const cache: WCache[] = Array.from({ length: this.dec.length }, () => ({ sk: null, sv: null, ck: null, cv: null }));
    const flatSelf = () => cache.flatMap((c) => (c.sk ? [c.sk, c.sv!] : []));
    const flatCross = () => cache.flatMap((c) => (c.ck ? [c.ck, c.cv!] : []));
    const tokens = [...(opts.prompt ?? [sp.sot, lang, sp.transcribe, sp.noTimestamps])];
    const start = tokens.length;
    let input = Int32Array.from(tokens), T = tokens.length, offset = 0;
    for (let step = 0; step < (opts.max ?? 224); step++) {
      const oldSelf = flatSelf(); // superseded by the concat this step; free after eval
      const tokMX = tidy(() => {
        const logits = this.decoderStep(input, T, offset, audio, cache);
        const last = logits.takeAxis(fromI32(Int32Array.from([T - 1]), [1]), 1).reshape([1, logits.shape[2]]);
        return { t: sample(last, 0, 0), keep: [...flatSelf(), ...flatCross()] };
      }).t;
      evalAll(tokMX, ...flatSelf(), ...flatCross());
      for (const m of oldSelf) m.free(); // safe: mlx refcounts keep them for the pending concat/eval
      const next = tokMX.itemU(); tokMX.free();
      offset += T;
      if (next === sp.eot) break;
      tokens.push(next);
      input = Int32Array.from([next]); T = 1;
    }
    for (const m of [...flatSelf(), ...flatCross()]) m.free();
    return tokens.slice(start);
  }
}

// ---- CLI: bun whisper.ts <audio-file> ----
if (import.meta.main) {
  const file = process.argv[2];
  if (!file) { console.error("usage: bun whisper.ts <audio-file>"); process.exit(1); }
  const model = await loadWhisper("models/config-turbo.json", "models/whisper-turbo.safetensors");
  const tok = await WhisperTokenizer.fromFile();
  const filtersT = await loadMelFilters("models/whisper-mel-filters-128.f32", 128);
  const pcm = await decodeAudio(file);
  const t0 = performance.now();
  const ids = model.transcribe(pcm, filtersT);
  const secs = (performance.now() - t0) / 1000;
  console.log(`text: ${tok.decode(ids).trim()}`);
  console.log(`(${ids.length} tokens in ${secs.toFixed(2)}s)`);
}

export function loadWhisper(cfgPath = "models/config-whisper.json", weightsPath = "models/whisper-tiny.safetensors") {
  return readJson(cfgPath).then((cfg: any) => new Whisper(cfg, loadSafetensors(weightsPath)));
}
