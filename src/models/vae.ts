// Stable Diffusion's VAE decoder: latents in, pixels out.
//
// Built first, before the UNet, because it is the smallest piece that produces
// something you can look at — and it can be checked on its own, rather than
// fifty diffusion steps deep where a small drift has already ruined the image.
//
// Weights follow Hugging Face's diffusers layout (stabilityai/sd-vae-ft-mse),
// not mlx-examples' remapped names, which is the convention the rest of this
// repo uses. Two consequences:
//
//   - conv weights are stored [C_out, C_in, KH, KW] and MLX wants channels
//     last, so every one is transposed on load;
//   - the residual shortcut is a 1x1 conv rather than a Linear.

import { type MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";
import { GroupNorm } from "../nn/nn.ts";

export type VaeConfig = {
  block_out_channels: number[];
  layers_per_block: number;
  norm_num_groups: number;
  latent_channels: number;
  out_channels: number;
  /** Latents are stored scaled; the decoder divides it back out. */
  scaling_factor?: number;
};

/** Repeat each pixel `scale` times in H and W — the upsample SD's decoder uses. */
export function upsampleNearest(x: MX, scale = 2): MX {
  const [B, H, W, C] = x.shape;
  return x.reshape([B, H, 1, W, 1, C])
    .broadcastTo([B, H, scale, W, scale, C])
    .reshape([B, H * scale, W * scale, C]);
}

export class VaeDecoder {
  cfg: VaeConfig;
  private W: Weights;
  private gn: GroupNorm;

  constructor(cfg: VaeConfig, W: Weights) {
    this.cfg = cfg;
    this.W = W;
    this.gn = new GroupNorm(cfg.norm_num_groups, 1e-6);
  }

  /** A conv weight, transposed from PyTorch's [O,I,KH,KW] to MLX's [O,KH,KW,I]. */
  private convW(name: string): MX {
    return this.W.mx(`${name}.weight`).transpose([0, 2, 3, 1]);
  }

  private conv(name: string, x: MX, padding = 1): MX {
    return x.conv2d(this.convW(name), [1, 1], [padding, padding])
      .add(this.W.mx(`${name}.bias`));
  }

  private norm(name: string, x: MX): MX {
    return this.gn.forward(x, this.W.mx(`${name}.weight`), this.W.mx(`${name}.bias`));
  }

  /** GroupNorm -> SiLU -> conv, twice, plus the residual. */
  private resnet(p: string, x: MX): MX {
    let y = this.conv(`${p}.conv1`, this.norm(`${p}.norm1`, x).silu());
    y = this.conv(`${p}.conv2`, this.norm(`${p}.norm2`, y).silu());
    // Channel counts change at the first resnet of a block; the checkpoint
    // carries a 1x1 conv for exactly that case and omits it otherwise.
    let skip = x;
    try {
      skip = this.conv(`${p}.conv_shortcut`, x, 0);
    } catch { /* same width, identity shortcut */ }
    return y.add(skip);
  }

  // diffusers renamed the attention projections part-way through: older
  // checkpoints (sd-vae-ft-mse) spell them query/key/value/proj_attn, newer ones
  // to_q/to_k/to_v/to_out.0. Both are in the wild, so both are accepted.
  private static ATTN: Record<string, string[]> = {
    q: ["query", "to_q"], k: ["key", "to_k"], v: ["value", "to_v"], o: ["proj_attn", "to_out.0"],
  };

  /** The first spelling that exists in this checkpoint. */
  private attnW(p: string, which: string, part: "weight" | "bias"): MX {
    const names = VaeDecoder.ATTN[which];
    for (const n of names) {
      try { return this.W.mx(`${p}.${n}.${part}`); } catch { /* try the other spelling */ }
    }
    throw new Error(`${p}: no attention ${which} weight — tried ${names.join(", ")}`);
  }

  /** Single-head attention over the flattened spatial grid, plus a residual. */
  private attention(p: string, x: MX): MX {
    const [B, H, W, C] = x.shape;
    const y = this.norm(`${p}.group_norm`, x);
    const proj = (which: string) =>
      y.matmul(this.attnW(p, which, "weight").transpose([1, 0]))
        .add(this.attnW(p, which, "bias")).reshape([B, H * W, C]);

    const q = proj("q"), k = proj("k"), v = proj("v");
    const scores = q.divScalar(Math.sqrt(C)).matmul(k.transpose([0, 2, 1]));
    const o = scores.softmax(-1).matmul(v).reshape([B, H, W, C]);

    return x.add(o.matmul(this.attnW(p, "o", "weight").transpose([1, 0]))
      .add(this.attnW(p, "o", "bias")));
  }

  /** Latents [B, h, w, latent_channels] -> image [B, H, W, 3] in roughly [-1, 1]. */
  decode(latents: MX): MX {
    return tidy(() => {
      const { block_out_channels: ch, layers_per_block: L, scaling_factor } = this.cfg;

      // Each stage runs in its own tidy() and the previous output is freed as
      // soon as it is consumed. One tidy around the whole decoder would hold
      // every intermediate at every resolution until the end — which is several
      // GB by the time it reaches full size, and how this first fell over.
      let x = latents;
      const stage = (fn: (h: MX) => MX) => {
        const prev = x;
        x = tidy(() => fn(prev));
        if (prev !== latents) prev.free();
      };

      stage((h) => {
        const z = scaling_factor ? h.divScalar(scaling_factor) : h;
        return this.conv("decoder.conv_in", this.conv("post_quant_conv", z, 0));
      });

      stage((h) => this.resnet("decoder.mid_block.resnets.0", h));
      stage((h) => this.attention("decoder.mid_block.attentions.0", h));
      stage((h) => this.resnet("decoder.mid_block.resnets.1", h));

      // Up blocks run coarse to fine, and every one but the last upsamples.
      for (let i = 0; i < ch.length; i++) {
        for (let j = 0; j < L + 1; j++) {
          stage((h) => this.resnet(`decoder.up_blocks.${i}.resnets.${j}`, h));
        }
        if (i < ch.length - 1) {
          stage((h) => this.conv(`decoder.up_blocks.${i}.upsamplers.0.conv`, upsampleNearest(h)));
        }
      }

      stage((h) => this.conv("decoder.conv_out", this.norm("decoder.conv_norm_out", h).silu()));
      return x;
    });
  }
}
