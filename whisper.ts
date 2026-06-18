// Whisper (speech-to-text) over mlx-c — encoder + decoder, mirroring
// mlx_whisper/whisper.py exactly. Conv1d stem -> sinusoidal pos -> bidirectional
// encoder; token+learned-pos -> causal decoder with cross-attention to the audio
// features -> tied-embedding logits. Runs in fp16 like the reference.
//   weights: whisper-tiny.safetensors (converted from mlx-community/whisper-tiny)

import { MX, fromF32, fromI32, scalar, sample, tidy } from "./mx.ts";
import { loadSafetensors, get } from "./loader.ts";
import { decodeAudio, padOrTrim, loadMelFilters, logMel } from "./audio.ts";
import { WhisperTokenizer, SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, EOT } from "./whisper-tokenizer.ts";

const FP16 = 9, EPS = 1e-5;

type Lin = { wt: MX; b: MX | null };

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
  nAudioHead: number; nTextHead: number; nTextLayer: number; D: number;
  conv1w: MX; conv1b: MX; conv2w: MX; conv2b: MX; encPos: MX; lnPostW: MX; lnPostB: MX;
  enc: any[]; tokEmb: MX; tokEmbT: MX; textPos: MX; dec: any[]; lnW: MX; lnB: MX;

  constructor(cfg: any, w: number) {
    this.nAudioHead = cfg.n_audio_head; this.nTextHead = cfg.n_text_head;
    this.nTextLayer = cfg.n_text_layer; this.D = cfg.n_audio_state;
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

  // Greedy transcription of a 16 kHz mono PCM clip (<= 30 s) -> token ids.
  // filtersT: Whisper's mel filterbank ([bins, n_mels]); prompt: SOT/lang/task.
  transcribe(pcm: Float32Array, filtersT: MX, opts: { max?: number; prompt?: number[] } = {}): number[] {
    const { mel, F } = logMel(padOrTrim(pcm), { filtersT, dropLast: true });
    const audio = this.encoder(fromF32(mel, [1, F, mel.length / F]).astype(9)); // [1, 1500, D]
    const tokens = [...(opts.prompt ?? [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS])];
    const start = tokens.length;
    for (let i = 0; i < (opts.max ?? 224); i++) {
      const T = tokens.length;
      const nextMX = tidy(() => {
        const logits = this.decoder(fromI32(Int32Array.from(tokens), [1, T]), audio);
        const last = logits.takeAxis(fromI32(Int32Array.from([T - 1]), [1]), 1).reshape([1, logits.shape[2]]);
        return sample(last, 0, 0); // greedy argmax -> [1]
      });
      const next = nextMX.itemU(); nextMX.free();
      if (next === EOT) break;
      tokens.push(next);
    }
    return tokens.slice(start);
  }
}

// ---- CLI: bun whisper.ts <audio-file> ----
if (import.meta.main) {
  const file = process.argv[2];
  if (!file) { console.error("usage: bun whisper.ts <audio-file>"); process.exit(1); }
  const model = await loadWhisper();
  const tok = await WhisperTokenizer.fromFile();
  const filtersT = await loadMelFilters();
  const pcm = await decodeAudio(file);
  const t0 = performance.now();
  const ids = model.transcribe(pcm, filtersT);
  const secs = (performance.now() - t0) / 1000;
  console.log(`text: ${tok.decode(ids).trim()}`);
  console.log(`(${ids.length} tokens in ${secs.toFixed(2)}s)`);
}

export function loadWhisper(cfgPath = "config-whisper.json", weightsPath = "whisper-tiny.safetensors") {
  return Bun.file(cfgPath).json().then((cfg: any) => new Whisper(cfg, loadSafetensors(weightsPath)));
}
