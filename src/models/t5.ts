// T5 encoder — the text conditioner for MusicGen.
//
// Three things T5 does differently from every other encoder in this repo, and
// each is silent if you get it wrong:
//   - RMSNorm with NO bias and no mean subtraction.
//   - NO 1/sqrt(head_dim) attention scaling. T5 folds it into the weights.
//   - Relative position bias instead of positional embeddings: a [buckets,
//     heads] table on block 0 only, added to the scores of EVERY layer.
import { escape, fromI32, type MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";

export type T5Config = {
  d_model: number; d_kv: number; d_ff: number; num_layers: number; num_heads: number;
  relative_attention_num_buckets: number; relative_attention_max_distance: number;
  layer_norm_epsilon: number;
};

/**
 * Map (memory_pos - query_pos) to a bucket: half the buckets count exact
 * offsets, the other half are logarithmic out to max_distance. Bidirectional,
 * so the sign picks which half of the table applies.
 */
function relativePositionBucket(q: number, k: number, numBuckets: number, maxDistance: number): Int32Array {
  const out = new Int32Array(q * k);
  const half = numBuckets >> 1;                 // bidirectional: table split in two
  const maxExact = half >> 1;
  const scale = (half - maxExact) / Math.log(maxDistance / maxExact);
  for (let i = 0; i < q; i++) {
    for (let j = 0; j < k; j++) {
      const rel = j - i;
      let bucket = rel > 0 ? half : 0;
      const n = Math.abs(rel);
      bucket += n < maxExact
        ? n
        : Math.min(maxExact + Math.floor(Math.log(n / maxExact) * scale), half - 1);
      out[i * k + j] = bucket;
    }
  }
  return out;
}

export class T5Encoder {
  cfg: T5Config;
  W: Weights;
  private prefix: string;
  constructor(cfg: T5Config, W: Weights, prefix = "text_encoder") { this.cfg = cfg; this.W = W; this.prefix = prefix; }

  private w(name: string): MX { return this.W.mx(`${this.prefix}.${name}`); }
  // T5 Linear layers have no bias; HF stores [out, in].
  private lin(name: string, x: MX): MX { return x.matmul(this.w(`${name}.weight`).transpose([1, 0])); }
  // RMSNorm: x / sqrt(mean(x^2) + eps) * weight. No bias, no centring.
  private rms(name: string, x: MX): MX { return x.rmsNorm(this.w(`${name}.weight`), this.cfg.layer_norm_epsilon); }

  /** token ids [B, L] -> encoder states [B, L, d_model] */
  encode(ids: MX): MX {
    return tidy(() => {
      const { num_heads: H, d_kv: dk, num_layers: NL, d_model: D } = this.cfg;
      const [B, L] = ids.shape;
      let x = this.w("shared.weight").takeAxis(ids.reshape([B * L]), 0).reshape([B, L, D]);

      // The bias table lives on block 0 and is reused by every layer.
      const buckets = fromI32(relativePositionBucket(L, L, this.cfg.relative_attention_num_buckets, this.cfg.relative_attention_max_distance), [L * L]);
      const bias = this.w("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
        .takeAxis(buckets, 0)                    // [L*L, H]
        .reshape([L, L, H]).transpose([2, 0, 1]).reshape([1, H, L, L]);

      for (let i = 0; i < NL; i++) {
        const b = `encoder.block.${i}`;
        const h = this.rms(`${b}.layer.0.layer_norm`, x);
        const split = (t: MX) => t.reshape([B, L, H, dk]).transpose([0, 2, 1, 3]);
        const q = split(this.lin(`${b}.layer.0.SelfAttention.q`, h));
        const k = split(this.lin(`${b}.layer.0.SelfAttention.k`, h));
        const v = split(this.lin(`${b}.layer.0.SelfAttention.v`, h));

        // Attention by hand rather than through fast SDPA: T5 needs an
        // ADDITIVE bias, and no 1/sqrt(dk) scaling.
        const scores = q.matmul(k.transpose([0, 1, 3, 2])).add(bias).softmax(-1);
        const o = scores.matmul(v).transpose([0, 2, 1, 3]).reshape([B, L, H * dk]);
        x = x.add(this.lin(`${b}.layer.0.SelfAttention.o`, o));

        const f = this.rms(`${b}.layer.1.layer_norm`, x);
        // feed_forward_proj is "relu" for t5-base (not gated): wi -> relu -> wo.
        const relu = (t: MX) => t.greater(this.zero()).where(t, this.zero());
        x = x.add(this.lin(`${b}.layer.1.DenseReluDense.wo`, relu(this.lin(`${b}.layer.1.DenseReluDense.wi`, f))));
      }
      return this.rms("encoder.final_layer_norm", x);
    });
  }

  private _zero: MX | null = null;
  private zero(): MX {
    // escape()d because it is cached on the instance and so must outlive the
    // tidy() that happens to create it. Without that, encode()'s own arena
    // frees it at scope exit while this field still points at the handle, and
    // the next generate() reads freed memory — "expected a non-empty
    // mlx_array". Same defect as FINDINGS §6.6, one level less obvious: the
    // value is created lazily, so the first call always works.
    if (!this._zero) this._zero = escape(fromI32(Int32Array.from([0]), [1]).astype(10));
    return this._zero;
  }
}
