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
import { clipBlocks } from "./clip-layers.ts";

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


  private ln(name: string, x: MX): MX {
    return x.layerNorm(this.w(`${name}.weight`), this.w(`${name}.bias`), this.cfg.layer_norm_eps);
  }

  /** Token ids [B, L] -> the final hidden states [B, L, D] SD conditions on. */
  encode(ids: MX): MX {
    return tidy(() => {
      const { hidden_size: D, num_hidden_layers: NL, num_attention_heads: H } = this.cfg;
      const [B, L] = ids.shape;

      let x = this.w("embeddings.token_embedding.weight")
        .takeAxis(ids.reshape([B * L]), 0).reshape([B, L, D]);
      // Positions are a learned table, truncated to the sequence length.
      x = x.add(this.w("embeddings.position_embedding.weight").slice([0, 0], [L, D]));

      x = clipBlocks(x, this.prefix, {
        w: (n) => this.W.mx(n), layers: NL, heads: H,
        eps: this.cfg.layer_norm_eps,
        quickGelu: this.cfg.hidden_act !== "gelu",
        causal: true,                        // CLIP's text tower is trained with a mask
      });

      return this.ln("final_layer_norm", x);
    });
  }
}
