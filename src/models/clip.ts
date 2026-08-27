// CLIP's text encoder — the conditioning half of Stable Diffusion.
//
// A pre-norm transformer with a causal mask, which is the detail worth knowing:
// CLIP is trained with one, so the text embedding at position i never sees
// later tokens, and dropping the mask produces plausible-but-wrong conditioning
// rather than an obvious failure.
//
// Weights follow Hugging Face's layout (openai/clip-vit-large-patch14 and the
// text_encoder/ of any Stable Diffusion repo), like the rest of this repo.
import { MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";

export type ClipConfig = {
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  max_position_embeddings: number;
  vocab_size: number;
  layer_norm_eps: number;
  /** "quick_gelu" for SD 1.x, plain "gelu" for some later encoders. */
  hidden_act?: string;
};

export class ClipTextEncoder {
  cfg: ClipConfig;
  private W: Weights;
  private prefix: string;

  constructor(cfg: ClipConfig, W: Weights, prefix = "text_model") {
    this.cfg = cfg;
    this.W = W;
    this.prefix = prefix;
  }

  private w(name: string): MX { return this.W.mx(`${this.prefix}.${name}`); }

  private lin(name: string, x: MX): MX {
    return x.matmul(this.w(`${name}.weight`).transpose([1, 0])).add(this.w(`${name}.bias`));
  }

  private ln(name: string, x: MX): MX {
    return x.layerNorm(this.w(`${name}.weight`), this.w(`${name}.bias`), this.cfg.layer_norm_eps);
  }

  /** CLIP's activation: x * sigmoid(1.702x), not the erf gelu used elsewhere. */
  private act(x: MX): MX {
    return this.cfg.hidden_act === "gelu" ? x.gelu() : x.mul(x.mulScalar(1.702).sigmoid());
  }

  /** Token ids [B, L] -> the final hidden states [B, L, D] SD conditions on. */
  encode(ids: MX): MX {
    return tidy(() => {
      const { hidden_size: D, num_attention_heads: H, num_hidden_layers: NL } = this.cfg;
      const [B, L] = ids.shape;
      const Dh = D / H;
      const scale = 1 / Math.sqrt(Dh);

      let x = this.w("embeddings.token_embedding.weight")
        .takeAxis(ids.reshape([B * L]), 0).reshape([B, L, D]);
      // Positions are a learned table, truncated to the sequence length.
      x = x.add(this.w("embeddings.position_embedding.weight").slice([0, 0], [L, D]));

      const heads = (t: MX) => t.reshape([B, L, H, Dh]).transpose([0, 2, 1, 3]);

      for (let l = 0; l < NL; l++) {
        const p = `encoder.layers.${l}`;
        const y = this.ln(`${p}.layer_norm1`, x);
        const o = MX.sdpa(
          heads(this.lin(`${p}.self_attn.q_proj`, y)),
          heads(this.lin(`${p}.self_attn.k_proj`, y)),
          heads(this.lin(`${p}.self_attn.v_proj`, y)),
          scale, true,                                   // causal: CLIP is trained with it
        ).transpose([0, 2, 1, 3]).reshape([B, L, D]);
        x = x.add(this.lin(`${p}.self_attn.out_proj`, o));

        const y2 = this.ln(`${p}.layer_norm2`, x);
        x = x.add(this.lin(`${p}.mlp.fc2`, this.act(this.lin(`${p}.mlp.fc1`, y2))));
      }

      return this.ln("final_layer_norm", x);
    });
  }
}
