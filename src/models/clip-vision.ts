// CLIP's vision tower — images into the same space as the text encoder.
//
// The transformer is identical to the text side (src/models/clip-layers.ts);
// what differs is the front and the back. The front turns an image into a
// sequence: a 14x14 stride-14 convolution cuts it into patches, a learned
// class token is prepended, and a position table is added. The back takes that
// class token, normalises it and projects it — into the same 768 dimensions
// text_projection maps prompts to, which is what makes similarity meaningful.
//
// Attention is NOT causal here: a patch may see the whole image. The text tower
// is masked, and sharing the layer code without sharing that flag is the bug
// this file most wants to avoid.
import { MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";
import { clipBlocks } from "./clip-layers.ts";

export type ClipVisionConfig = {
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  image_size: number;
  patch_size: number;
  layer_norm_eps: number;
  hidden_act?: string;
};

export class ClipVisionEncoder {
  cfg: ClipVisionConfig;
  private W: Weights;
  private prefix: string;

  constructor(cfg: ClipVisionConfig, W: Weights, prefix = "vision_model") {
    this.cfg = cfg;
    this.W = W;
    this.prefix = prefix;
  }

  private w(name: string): MX { return this.W.mx(`${this.prefix}.${name}`); }

  private ln(name: string, x: MX): MX {
    return x.layerNorm(this.w(`${name}.weight`), this.w(`${name}.bias`), this.cfg.layer_norm_eps);
  }

  /** The number of patches a square image of `image_size` produces. */
  get patches(): number {
    const side = this.cfg.image_size / this.cfg.patch_size;
    return side * side;
  }

  /**
   * Pixels `[B, H, W, 3]`, normalised the way CLIP expects, -> the pooled class
   * embedding `[B, D]` *before* the visual projection. Use `embed()` for the
   * projected vector that compares against text.
   */
  encode(pixels: MX): MX {
    return tidy(() => {
      const { hidden_size: D, num_hidden_layers: NL, num_attention_heads: H, patch_size: P } = this.cfg;
      const B = pixels.shape[0];

      // A stride-P convolution with a PxP kernel *is* the patch split: each
      // output position is one patch, projected to D. Stored [O, C, KH, KW].
      const kernel = this.w("embeddings.patch_embedding.weight").transpose([0, 2, 3, 1]);
      const grid = pixels.conv2d(kernel, [P, P], [0, 0]);          // [B, s, s, D]
      let x = grid.reshape([B, this.patches, D]);

      // The class token leads the sequence and is what gets pooled at the end.
      const cls = this.w("embeddings.class_embedding").reshape([1, 1, D]).broadcastTo([B, 1, D]);
      x = cls.concat(x, 1);                                         // [B, 1 + patches, D]
      x = x.add(this.w("embeddings.position_embedding.weight")
        .slice([0, 0], [this.patches + 1, D]));

      // Note the upstream spelling: HF's checkpoint really does say "layrnorm".
      x = this.ln("pre_layrnorm", x);
      x = clipBlocks(x, this.prefix, {
        w: (n) => this.W.mx(n), layers: NL, heads: H,
        eps: this.cfg.layer_norm_eps,
        quickGelu: this.cfg.hidden_act !== "gelu",
        causal: false,                       // an image has no reading order
      });

      const pooled = x.slice([0, 0, 0], [B, 1, D]).reshape([B, D]);
      return this.ln("post_layernorm", pooled);
    });
  }

  /** The projected image vector, in the space text prompts map into. */
  embed(pixels: MX, projection: MX): MX {
    return tidy(() => this.encode(pixels).matmul(projection.transpose([1, 0])));
  }
}
