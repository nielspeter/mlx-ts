// Qwen2 / Qwen2.5 decoder — the backbone under a great many checkpoints,
// including Spark-TTS.
//
// Close to src/models/qwen-nn.ts but not the same model, and the differences
// are the kind that produce plausible-but-wrong output rather than an error:
//
//   - Qwen2 puts a bias on q, k and v; Qwen3 has none.
//   - Qwen3 normalises q and k per head; Qwen2 does not.
//   - head_dim is hidden_size / heads here, not a separate config field.
//
// Weights are read at full precision by name, in Hugging Face's layout, so
// this loads any bf16/fp16 qwen2 checkpoint. The quantized path stays in
// qwen-nn.ts.
import { fromI32, MX } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";
import { Embedding, Linear, RMSNorm } from "../nn/nn.ts";
import type { Decoder, KV } from "../text/lm.ts";

export type Qwen2Config = {
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  rms_norm_eps: number;
  rope_theta: number;
  vocab_size: number;
  eos_token_id: number;
  tie_word_embeddings?: boolean;
  head_dim?: number;
};

type Layer = {
  inNorm: RMSNorm; postNorm: RMSNorm;
  q: Linear; k: Linear; v: Linear; o: Linear;
  gate: Linear; up: Linear; down: Linear;
};

export class Qwen2 implements Decoder {
  readonly numLayers: number;
  readonly eos: number;
  cfg: Qwen2Config;

  private D: number; private nH: number; private nKV: number; private Dh: number;
  private scale: number; private theta: number; private eps: number;
  private embed: Embedding;
  private finalNorm: RMSNorm;
  private head: Linear | null;
  private layers: Layer[];

  constructor(cfg: Qwen2Config, W: Weights) {
    this.cfg = cfg;
    this.D = cfg.hidden_size;
    this.numLayers = cfg.num_hidden_layers;
    this.nH = cfg.num_attention_heads;
    this.nKV = cfg.num_key_value_heads;
    this.Dh = cfg.head_dim ?? cfg.hidden_size / cfg.num_attention_heads;
    this.scale = this.Dh ** -0.5;
    this.theta = cfg.rope_theta;
    this.eps = cfg.rms_norm_eps;
    this.eos = cfg.eos_token_id;

    // nn.Linear multiplies x by its weight directly, so HF's [out, in] is
    // transposed once here rather than on every forward pass.
    const lin = (n: string, bias: boolean) =>
      new Linear(W.mx(`${n}.weight`).transpose([1, 0]), bias ? W.mx(`${n}.bias`) : undefined);
    const rn = (n: string) => new RMSNorm(W.mx(`${n}.weight`), this.eps);

    this.embed = new Embedding(W.mx("model.embed_tokens.weight"));
    this.finalNorm = rn("model.norm");
    // Most qwen2 checkpoints tie the head to the embedding; a few ship it.
    let head: Linear | null = null;
    if (!cfg.tie_word_embeddings) {
      try { head = lin("lm_head", false); } catch { head = null; }
    }
    this.head = head;

    this.layers = Array.from({ length: this.numLayers }, (_, i) => {
      const p = `model.layers.${i}`;
      return {
        inNorm: rn(`${p}.input_layernorm`), postNorm: rn(`${p}.post_attention_layernorm`),
        q: lin(`${p}.self_attn.q_proj`, true),      // the biases Qwen3 does not have
        k: lin(`${p}.self_attn.k_proj`, true),
        v: lin(`${p}.self_attn.v_proj`, true),
        o: lin(`${p}.self_attn.o_proj`, false),
        gate: lin(`${p}.mlp.gate_proj`, false),
        up: lin(`${p}.mlp.up_proj`, false),
        down: lin(`${p}.mlp.down_proj`, false),
      };
    });
  }

  private block(li: number, h: MX, B: number, L: number, offset: number, cache: KV[]): MX {
    const W = this.layers[li];
    const { nH, nKV, Dh } = this;

    const y = W.inNorm.forward(h);
    // No per-head q/k norm here — that is Qwen3's addition.
    let q = W.q.forward(y).reshape([B, L, nH, Dh]).transpose([0, 2, 1, 3]);
    let k = W.k.forward(y).reshape([B, L, nKV, Dh]).transpose([0, 2, 1, 3]);
    const v = W.v.forward(y).reshape([B, L, nKV, Dh]).transpose([0, 2, 1, 3]);

    q = q.rope(Dh, this.theta, offset);
    k = k.rope(Dh, this.theta, offset);

    const prev = cache[li];
    let keys = k, values = v;
    if (prev) { keys = prev.k.concat(k, 2); values = prev.v.concat(v, 2); }
    cache[li] = { k: keys, v: values };

    const o = MX.sdpa(q, keys, values, this.scale, L > 1)
      .transpose([0, 2, 1, 3]).reshape([B, L, nH * Dh]);
    h = h.add(W.o.forward(o));

    const y2 = W.postNorm.forward(h);
    return h.add(W.down.forward(W.gate.forward(y2).silu().mul(W.up.forward(y2))));
  }

  /** ids [B, L] -> logits at the last position [B, vocab]. Graph only, not evaluated. */
  logitsLastMX(idsMX: MX, B: number, L: number, offset: number, cache: KV[], _window: number): MX {
    let h = this.embed.forward(idsMX.reshape([B * L])).reshape([B, L, this.D]);
    for (let i = 0; i < this.numLayers; i++) h = this.block(i, h, B, L, offset, cache);
    const last = this.finalNorm.forward(h.slice([0, L - 1, 0], [B, L, this.D]).reshape([B, this.D]));
    return this.head ? this.head.forward(last) : this.embed.asLinear(last);
  }

  /** ids on the host -> logits at the last position. */
  logitsLast(ids: Int32Array, B: number, L: number, offset: number, cache: KV[], window: number): MX {
    return this.logitsLastMX(fromI32(ids, [B, L]), B, L, offset, cache, window);
  }
}
