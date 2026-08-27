// MusicGen's language model: a 24-layer decoder emitting EnCodec codebook
// tokens, conditioned on T5 text embeddings via cross-attention.
//
// Weights follow Hugging Face's layout (facebook/musicgen-small), not the
// audiocraft state_dict.bin that mlx-examples maps.
//
// Memory: generation is a long loop over a KV cache that must outlive each
// step — the tidy()/escape() split exactly. Every step runs inside tidy(), and
// the cache escapes it. Without that the cache is freed as a scope-local
// intermediate and the next step reads freed handles (see FINDINGS §6.6).
import { escape, evalAll, fromI32, fromU32, MX, Owned, sample, scalar, setCacheLimit, stack, tidy } from "../core/mx.ts";
import { readJson } from "../io/fs.ts";
import { hubFile } from "../io/hub.ts";
import type { Weights } from "../io/loader.ts";
import { singleFileWeights } from "../io/loader.ts";
import { UnigramTokenizer } from "../text/unigram.ts";
import { type EncodecConfig, EncodecDecoder } from "./encodec.ts";
import { type T5Config, T5Encoder } from "./t5.ts";
export type MusicGenConfig = {
  hidden_size: number; num_hidden_layers: number; num_attention_heads: number;
  ffn_dim: number; num_codebooks: number; bos_token_id: number;
  vocab_size: number; max_position_embeddings: number;
};

/** One layer's key/value cache; `null` until the first step fills it. */
export type LayerKV = { k: MX; v: MX } | null;

const EPS = 1e-5;

export class MusicGenLM {
  cfg: MusicGenConfig;
  W: Weights;
  private nH: number;
  private headDim: number;
  private scale: number;

  constructor(cfg: MusicGenConfig, W: Weights) {
    this.cfg = cfg; this.W = W;
    this.nH = cfg.num_attention_heads;
    this.headDim = cfg.hidden_size / this.nH;
    this.scale = this.headDim ** -0.5;
  }

  // HF stores Linear weights as [out, in] and applies x @ W.T, so transpose.
  private lin(name: string, x: MX): MX {
    const w = this.W.mx(`${name}.weight`);
    const y = x.matmul(w.transpose([1, 0]));
    try { return y.add(this.W.mx(`${name}.bias`)); } catch { return y; }
  }
  private norm(name: string, x: MX): MX {
    return x.layerNorm(this.W.mx(`${name}.weight`), this.W.mx(`${name}.bias`), EPS);
  }

  /** [B, L, D] -> [B, nH, L, headDim] */
  private split(x: MX, B: number, L: number): MX {
    return x.reshape([B, L, this.nH, this.headDim]).transpose([0, 2, 1, 3]);
  }

  private attention(prefix: string, q: MX, kv: MX, cache: LayerKV, causal: boolean):
      { out: MX; kv: LayerKV } {
    const [B, Lq] = q.shape;
    const Lk = kv.shape[1];
    let keys = this.split(this.lin(`${prefix}.k_proj`, kv), B, Lk);
    let values = this.split(this.lin(`${prefix}.v_proj`, kv), B, Lk);
    const queries = this.split(this.lin(`${prefix}.q_proj`, q), B, Lq);

    if (cache) {                       // self-attention: append this step's k/v
      keys = cache.k.concat(keys, 2);
      values = cache.v.concat(values, 2);
    }
    const o = MX.sdpa(queries, keys, values, this.scale, causal && Lq > 1)
      .transpose([0, 2, 1, 3]).reshape([B, Lq, this.cfg.hidden_size]);
    return { out: this.lin(`${prefix}.out_proj`, o), kv: { k: keys, v: values } };
  }

  /**
   * One decoding step.
   * @param tokens [B, 1, num_codebooks] codebook indices for this position
   * @param cond   [B, Lt, D] projected T5 conditioning
   * @param cache  per-layer KV, mutated in place; entries escape the caller's arena
   * @param offset absolute position, for the positional embedding
   */
  step(tokens: MX, cond: MX, cache: Owned<LayerKV>, offset: number): MX {
    const { num_codebooks: K, num_hidden_layers: NL, hidden_size: D } = this.cfg;
    const B = tokens.shape[0];

    // Sum the per-codebook embeddings.
    let x: MX | null = null;
    for (let k = 0; k < K; k++) {
      const idx = tokens.slice([0, 0, k], [B, 1, k + 1]).reshape([B]);
      const e = this.W.mx(`decoder.model.decoder.embed_tokens.${k}.weight`).takeAxis(idx, 0).reshape([B, 1, D]);
      x = x ? x.add(e) : e;
    }
    // Learned sinusoidal table, as stored in this checkpoint.
    const pos = this.W.mx("decoder.model.decoder.embed_positions.weights")
      .takeAxis(fromI32(Int32Array.from([offset]), [1]), 0).reshape([1, 1, D]);
    x = x!.add(pos);

    for (let l = 0; l < NL; l++) {
      const p = `decoder.model.decoder.layers.${l}`;
      const sa = this.attention(`${p}.self_attn`, this.norm(`${p}.self_attn_layer_norm`, x), 
                                this.norm(`${p}.self_attn_layer_norm`, x), cache.get(l), true);
      // Owned.set() escapes the pair out of this step's arena and frees the one
      // it replaces. The cache grows by one token per step, so leaking it costs
      // ~393 KB x step x layer — about 49 GB over a 500-step generation, which
      // is exactly what it did before the free was there.
      cache.set(l, { k: sa.kv!.k, v: sa.kv!.v });
      x = x.add(sa.out);

      const xc = this.norm(`${p}.encoder_attn_layer_norm`, x);
      x = x.add(this.attention(`${p}.encoder_attn`, xc, cond, null, false).out);

      const xf = this.norm(`${p}.final_layer_norm`, x);
      x = x.add(this.lin(`${p}.fc2`, this.lin(`${p}.fc1`, xf).gelu()));
    }

    x = this.norm("decoder.model.decoder.layer_norm", x);
    // One head per codebook -> [B, 1, vocab, K]
    return stack(Array.from({ length: K }, (_, k) => this.lin(`decoder.lm_heads.${k}`, x!)), -1);
  }
}

// ---------------------------------------------------------------------------
// The whole pipeline: text -> T5 -> LM (delay pattern + CFG) -> EnCodec -> audio
// ---------------------------------------------------------------------------


export type GenerateOptions = {
  /** Frames to generate. 50 frames ≈ 1 second at 32 kHz. */
  maxSteps?: number;
  topK?: number;
  temp?: number;
  /** Classifier-free guidance: how far to push from the unconditional logits. */
  guidance?: number;
  /**
   * Ceiling on MLX's buffer-reuse cache while generating, in MB. Defaults to
   * 512, which costs nothing measurable and saves many GB — see generate().
   * Pass Infinity to leave MLX's own default alone.
   */
  cacheLimitMB?: number;
  onStep?: (step: number, total: number) => void;
};

// ---------------------------------------------------------------------------
// Repo layouts
//
// Two of them ship MusicGen, and the model code above knows only the Hugging
// Face names:
//
//   HF    one model.safetensors, keys like decoder.model.decoder.layers.0.*
//         Only musicgen-small ships it — -medium and -large are PyTorch
//         pickles, which TypeScript cannot read.
//   MLX   decoder.safetensors + t5.safetensors under mlx-audiogen's shorter
//         names. jasonvassallo/mlx-musicgen-* publishes medium, large, melody
//         and stereo this way, so it is the only route to anything above
//         small that does not involve converting a 6 GB pickle.
//
// Neither carries EnCodec in a form we can use, so the audio half always comes
// from its own repo (see fromPretrained). A Weights is just { mx, done }, so a
// layout costs a name rewrite and, when split, a route to the right file.

/**
 * A Hugging Face tensor name -> the same tensor in an mlx-audiogen checkpoint.
 * Exported for validation/musicgen-mlx-layout.ts, which checks every name the
 * model asks for against the real headers. Not part of the package API.
 */
export function mlxName(n: string): string {
  if (n.startsWith("text_encoder.")) {
    return n.slice("text_encoder.".length)
      .replace(".layer.0.SelfAttention.", ".self_attn.")
      .replace(".layer.0.layer_norm.", ".self_attn_norm.")
      .replace(".layer.1.DenseReluDense.", ".ff.")
      .replace(".layer.1.layer_norm.", ".ff_norm.");
  }
  // decoder.model.decoder.X -> X and decoder.lm_heads.X -> lm_heads.X;
  // enc_to_dec_proj.X is spelled the same in both.
  return n.replace(/^decoder\.(model\.decoder\.)?/, "");
}

/** Missing file -> null, so a layout can be probed. Anything else still throws. */
async function optionalHubFile(repo: string, file: string): Promise<string | null> {
  try {
    return await hubFile(repo, file);
  } catch (e) {
    if (String(e).includes(" 404 ")) return null;
    throw e;                       // a network or auth failure is not a layout answer
  }
}

/** Whichever layout `repo` uses, presented under the Hugging Face names. */
async function openWeights(repo: string): Promise<Weights> {
  const single = await optionalHubFile(repo, "model.safetensors");
  if (single) return singleFileWeights(single);

  const lmPath = await optionalHubFile(repo, "decoder.safetensors");
  if (lmPath) {
    const lm = singleFileWeights(lmPath);
    const t5 = singleFileWeights(await hubFile(repo, "t5.safetensors"));
    return {
      mx: (n) => (n.startsWith("text_encoder.") ? t5 : lm).mx(mlxName(n)),
      done: () => { lm.done(); t5.done(); },
    };
  }

  const size = repo.match(/(small|medium|large|melody)/)?.[1];
  throw new Error(
    `${repo} has no safetensors — only PyTorch checkpoints, which cannot be read ` +
    `from TypeScript.` +
    (size && size !== "small"
      ? ` Use jasonvassallo/mlx-musicgen-${size} instead: same weights, converted.`
      : ``),
  );
}

export class MusicGen {
  readonly samplingRate: number;
  private lm: MusicGenLM;
  private t5: T5Encoder;
  private codec: EncodecDecoder;
  private tok: UnigramTokenizer;
  private W: Weights;

  private constructor(lm: MusicGenLM, t5: T5Encoder, codec: EncodecDecoder,
                      tok: UnigramTokenizer, W: Weights, samplingRate: number) {
    this.lm = lm; this.t5 = t5; this.codec = codec; this.tok = tok; this.W = W;
    this.samplingRate = samplingRate;
  }

  static async fromPretrained(repo = "facebook/musicgen-small"): Promise<MusicGen> {
    const cfg = await readJson<any>(await hubFile(repo, "config.json"));
    const W = await openWeights(repo);
    const tok = await UnigramTokenizer.fromFile(await hubFile(repo, "tokenizer.json"));

    // EnCodec always comes from its own repo: the HF checkpoint bundles a copy
    // that still has weight_g/weight_v (weight-norm unfused), and the MLX
    // checkpoints leave it out entirely. It is the same 32 kHz codec at every
    // model size, so one repo serves all of them.
    const codecRepo = "mlx-community/encodec-32khz-float32";
    const codecCfg = await readJson<EncodecConfig>(await hubFile(codecRepo, "config.json"));
    const codecW = singleFileWeights(await hubFile(codecRepo, "model.safetensors"));

    return new MusicGen(
      new MusicGenLM(cfg.decoder as MusicGenConfig, W),
      new T5Encoder(cfg.text_encoder as T5Config, W),
      new EncodecDecoder(codecCfg, codecW),
      tok, W, cfg.audio_encoder.sampling_rate,
    );
  }

  /** Text prompt -> mono waveform. */
  generate(text: string, opts: GenerateOptions = {}): MX {
    const { maxSteps = 250, topK = 250, temp = 1.0, guidance = 3.0, onStep,
            cacheLimitMB = 512 } = opts;

    // MLX keeps freed Metal buffers in a reuse pool whose default ceiling is
    // the machine's RAM (36 GB here), and a generation loop fills it: 300 steps
    // of -small reached 16.7 GB total, and -medium reached 28 GB, which is
    // enough to drive a 39 GB machine into swap. Capping it costs nothing
    // measurable — 61.9 steps/s uncapped vs 61.4 at 256 MB — and this is the
    // only knob that bounds it, since active memory was never the problem.
    // Restored on the way out so the cap stays a local decision.
    const prevCacheLimit = Number.isFinite(cacheLimitMB) ? setCacheLimit(cacheLimitMB) : 0;
    try {
      const out = this.generateInner(text, maxSteps, topK, temp, guidance, onStep);
      // MLX is lazy, so without this the EnCodec decode would run at the
      // caller's first read — after the cap is restored, which cost 3.5 GB.
      evalAll(out);
      return out;
    } finally {
      if (Number.isFinite(cacheLimitMB)) setCacheLimit(prevCacheLimit);
    }
  }

  private generateInner(text: string, maxSteps: number, topK: number, temp: number,
                        guidance: number, onStep: GenerateOptions["onStep"]): MX {
    const K = this.lm.cfg.num_codebooks;

    // The delay pattern eats K frames, and EnCodec's decoder needs a few more
    // than that: its conv stack cannot run on an input shorter than the kernel
    // is wide. Below this it dies inside MLX with "[conv] Spatial dimensions of
    // input after padding cannot be smaller than weight spatial dimensions",
    // which tells a caller nothing. Measured: K + 3 is the shortest that
    // decodes.
    if (maxSteps < K + 3) {
      throw new Error(
        `maxSteps must be at least ${K + 3} (got ${maxSteps}): the delay pattern ` +
        `consumes ${K} frames and EnCodec cannot decode what is left. ` +
        `That is ${((K + 3) / 50).toFixed(2)}s of audio — use --seconds instead.`,
      );
    }
    const BOS = this.lm.cfg.bos_token_id;
    const V = this.lm.cfg.vocab_size;

    // Conditioning: T5 states projected into the LM's width, then batched with
    // an all-zero copy so one forward pass yields both the conditional and
    // unconditional logits.
    const cond = tidy(() => {
      const tokens = this.tok.encode(text);
      const ids = fromI32(Int32Array.from(tokens), [1, tokens.length]);
      const h = this.t5.encode(ids);
      const proj = h.matmul(this.W.mx("enc_to_dec_proj.weight").transpose([1, 0]))
        .add(this.W.mx("enc_to_dec_proj.bias"));
      return escape(proj.concat(proj.mul(scalar(0)), 0));            // [2, L, D]
    });

    // The delay pattern: codebook k does not start until step k, so the model
    // predicts each codebook conditioned on the coarser ones above it.
    const seq: number[][] = [new Array(K).fill(BOS)];
    using cache = new Owned<LayerKV>(this.lm.cfg.num_hidden_layers);

    for (let off = 0; off < maxSteps; off++) {
      const cur = seq[off];
      const next = tidy(() => {
        const inp = fromU32(Uint32Array.from([...cur, ...cur]), [2, 1, K]);   // CFG batch
        const logits = this.lm.step(inp, cond, cache, off);                    // [2,1,V,K]
        const c = logits.slice([0, 0, 0, 0], [1, 1, V, K]);
        const u = logits.slice([1, 0, 0, 0], [2, 1, V, K]);
        const guided = u.add(c.sub(u).mul(scalar(guidance)));
        // Sample each codebook independently from the guided logits.
        return Array.from({ length: K }, (_, k) =>
          sample(guided.slice([0, 0, 0, k], [1, 1, V, k + 1]).reshape([1, V]), temp, 0, topK).toU32()[0]);
      });
      for (let k = 0; k < K; k++) if (k > off) next[k] = BOS;                  // not started yet
      seq.push(next);
      onStep?.(off + 1, maxSteps);
    }
    // The conditioning escaped every tidy() above, so nothing else will release
    // it; the cache frees itself at scope exit. EnCodec needs the memory next.
    cond.free();
    cache.free();

    // Undo the delay: codebook i is i steps late, so shift it back.
    const rows = seq.length - K;
    const codes = new Uint32Array(K * (rows - 1));
    for (let t = 1; t < rows; t++) {
      for (let i = 0; i < K; i++) codes[i * (rows - 1) + (t - 1)] = seq[t + i][i];
    }
    return this.codec.decode(fromU32(codes, [1, K, rows - 1]));
  }
}
