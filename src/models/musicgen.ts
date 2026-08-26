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
import { escape, fromI32, MX, stack } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";

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
  step(tokens: MX, cond: MX, cache: LayerKV[], offset: number): MX {
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
                                this.norm(`${p}.self_attn_layer_norm`, x), cache[l], true);
      cache[l] = { k: escape(sa.kv!.k), v: escape(sa.kv!.v) };
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
