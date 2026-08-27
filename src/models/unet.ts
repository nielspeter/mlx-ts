// Stable Diffusion's UNet — the model that actually does the denoising.
//
// A down/mid/up stack over latents, conditioned two ways: a timestep embedding
// added inside every resnet, and the CLIP text states cross-attended to inside
// every transformer block. The skip connections are the fiddly part — each up
// block concatenates residuals the down blocks pushed, in reverse order, and
// getting the stack depth wrong is a shape error rather than a silent one.
//
// Weights follow Hugging Face's diffusers layout, like the rest of the repo,
// which costs three conversions mlx-examples avoids by remapping:
//   - conv weights arrive [C_out, C_in, KH, KW] and MLX wants channels last;
//   - proj_in / proj_out are stored as 1x1 convs but used as linears;
//   - the feed-forward is GEGLU, one [2*hidden, dim] matrix that splits into
//     a value half and a gate half.
import { fromF32, MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";
import { GroupNorm } from "../nn/nn.ts";
import { upsampleNearest } from "./vae.ts";

export type UnetConfig = {
  in_channels: number;
  out_channels: number;
  block_out_channels: number[];
  layers_per_block: number;
  norm_num_groups: number;
  cross_attention_dim: number;
  /** For SD 1.x this is the head *count*, despite the name. */
  attention_head_dim: number | number[];
  down_block_types: string[];
  up_block_types: string[];
};

/**
 * The sinusoidal timestep embedding, matching MLX's SinusoidalPositionalEncoding
 * as mlx-examples configures it: max_freq 1, min_freq exp(-log(1e4) +
 * 2 log(1e4)/dims), cosines first, scale 1. Small enough to build directly.
 */
export function timestepEmbedding(t: number, dims: number): Float32Array {
  const half = dims / 2;
  const logMin = -Math.log(10000) + (2 * Math.log(10000)) / dims;
  const out = new Float32Array(dims);
  for (let i = 0; i < half; i++) {
    const oneZero = 1 - i / (half - 1);          // 1 down to 0
    const y = t * Math.exp(oneZero * (0 - logMin) + logMin);
    out[i] = Math.cos(y);                        // cos_first
    out[half + i] = Math.sin(y);
  }
  return out;
}

export class Unet {
  cfg: UnetConfig;
  private W: Weights;
  private gn: GroupNorm;

  constructor(cfg: UnetConfig, W: Weights) {
    this.cfg = cfg;
    this.W = W;
    this.gn = new GroupNorm(cfg.norm_num_groups, 1e-5);
  }

  private heads(i: number): number {
    const h = this.cfg.attention_head_dim;
    return Array.isArray(h) ? h[i] : h;
  }

  private conv(name: string, x: MX, stride = 1, padding = 1): MX {
    return x.conv2d(this.W.mx(`${name}.weight`).transpose([0, 2, 3, 1]), [stride, stride], [padding, padding])
      .add(this.W.mx(`${name}.bias`));
  }

  /** A 1x1 conv stored in the checkpoint, applied as a linear over channels. */
  private conv1x1AsLinear(name: string, x: MX): MX {
    const w = this.W.mx(`${name}.weight`);
    const [O, I] = w.shape;
    return x.matmul(w.reshape([O, I]).transpose([1, 0])).add(this.W.mx(`${name}.bias`));
  }

  private lin(name: string, x: MX, bias = true): MX {
    const y = x.matmul(this.W.mx(`${name}.weight`).transpose([1, 0]));
    return bias ? y.add(this.W.mx(`${name}.bias`)) : y;
  }

  private norm(name: string, x: MX): MX {
    return this.gn.forward(x, this.W.mx(`${name}.weight`), this.W.mx(`${name}.bias`));
  }

  private ln(name: string, x: MX): MX {
    return x.layerNorm(this.W.mx(`${name}.weight`), this.W.mx(`${name}.bias`), 1e-5);
  }

  /** GroupNorm -> SiLU -> conv, twice, with the timestep added in between. */
  private resnet(p: string, x: MX, temb: MX): MX {
    let y = this.conv(`${p}.conv1`, this.norm(`${p}.norm1`, x).silu());
    // [B, C] -> [B, 1, 1, C] so it broadcasts across the spatial grid.
    const t = this.lin(`${p}.time_emb_proj`, temb.silu());
    y = y.add(t.reshape([t.shape[0], 1, 1, t.shape[1]]));
    y = this.conv(`${p}.conv2`, this.norm(`${p}.norm2`, y).silu());

    let skip = x;
    try { skip = this.conv(`${p}.conv_shortcut`, x, 1, 0); } catch { /* widths match */ }
    return y.add(skip);
  }

  /** Self- or cross-attention; `ctx` null means self. Projections carry no bias. */
  private attn(p: string, x: MX, ctx: MX | null, nHeads: number): MX {
    const [B, L, D] = x.shape;
    const kv = ctx ?? x;
    const Dh = D / nHeads;
    const split = (t: MX, len: number) => t.reshape([B, len, nHeads, Dh]).transpose([0, 2, 1, 3]);

    const q = split(this.lin(`${p}.to_q`, x, false), L);
    const k = split(this.lin(`${p}.to_k`, kv, false), kv.shape[1]);
    const v = split(this.lin(`${p}.to_v`, kv, false), kv.shape[1]);

    const o = MX.sdpa(q, k, v, 1 / Math.sqrt(Dh), false)
      .transpose([0, 2, 1, 3]).reshape([B, L, D]);
    return this.lin(`${p}.to_out.0`, o);
  }

  /** The spatial transformer: norm, flatten, N blocks of self/cross/FFN, restore. */
  private transformer2d(p: string, x: MX, cond: MX, nHeads: number, nBlocks: number): MX {
    const [B, H, W, C] = x.shape;
    const input = x;

    let h = this.conv1x1AsLinear(`${p}.proj_in`, this.norm(`${p}.norm`, x).reshape([B, H * W, C]));

    for (let i = 0; i < nBlocks; i++) {
      const b = `${p}.transformer_blocks.${i}`;
      h = h.add(this.attn(`${b}.attn1`, this.ln(`${b}.norm1`, h), null, nHeads));
      h = h.add(this.attn(`${b}.attn2`, this.ln(`${b}.norm2`, h), cond, nHeads));

      // GEGLU: one projection, split into a value half and a gate half.
      const y = this.ln(`${b}.norm3`, h);
      const projW = this.W.mx(`${b}.ff.net.0.proj.weight`);
      const projB = this.W.mx(`${b}.ff.net.0.proj.bias`);
      const hid = projW.shape[0] / 2, dim = projW.shape[1];
      const half = (lo: number, hi: number) =>
        y.matmul(projW.slice([lo, 0], [hi, dim]).transpose([1, 0])).add(projB.slice([lo], [hi]));
      h = h.add(this.lin(`${b}.ff.net.2`, half(0, hid).mul(half(hid, 2 * hid).gelu())));
    }

    return this.conv1x1AsLinear(`${p}.proj_out`, h).reshape([B, H, W, C]).add(input);
  }

  /** Noisy latents + timestep + text states -> predicted noise, same shape as x. */
  forward(x: MX, timestep: number, cond: MX): MX {
    return tidy(() => {
      const { block_out_channels: ch, layers_per_block: L, cross_attention_dim: _ } = this.cfg;
      const B = x.shape[0];

      // Timestep: sinusoidal, then two linears with a SiLU between.
      const te = fromF32(timestepEmbedding(timestep, ch[0]), [1, ch[0]]);
      let temb = this.lin("time_embedding.linear_2",
        this.lin("time_embedding.linear_1", te).silu());
      if (B > 1) temb = temb.broadcastTo([B, temb.shape[1]]);

      let h = this.conv("conv_in", x);
      const residuals: MX[] = [h];

      for (let i = 0; i < ch.length; i++) {
        const cross = this.cfg.down_block_types[i].includes("CrossAttn");
        for (let j = 0; j < L; j++) {
          h = this.resnet(`down_blocks.${i}.resnets.${j}`, h, temb);
          if (cross) h = this.transformer2d(`down_blocks.${i}.attentions.${j}`, h, cond, this.heads(i), 1);
          residuals.push(h);
        }
        if (i < ch.length - 1) {
          h = this.conv(`down_blocks.${i}.downsamplers.0.conv`, h, 2, 1);
          residuals.push(h);
        }
      }

      h = this.resnet("mid_block.resnets.0", h, temb);
      h = this.transformer2d("mid_block.attentions.0", h, cond, this.heads(ch.length - 1), 1);
      h = this.resnet("mid_block.resnets.1", h, temb);

      for (let i = 0; i < ch.length; i++) {
        const cross = this.cfg.up_block_types[i].includes("CrossAttn");
        for (let j = 0; j < L + 1; j++) {
          // Each up-block layer eats one residual, most recent first.
          h = h.concat(residuals.pop()!, -1);
          h = this.resnet(`up_blocks.${i}.resnets.${j}`, h, temb);
          if (cross) h = this.transformer2d(`up_blocks.${i}.attentions.${j}`, h, cond, this.heads(ch.length - 1 - i), 1);
        }
        if (i < ch.length - 1) {
          h = this.conv(`up_blocks.${i}.upsamplers.0.conv`, upsampleNearest(h));
        }
      }

      return this.conv("conv_out", this.norm("conv_norm_out", h).silu());
    });
  }
}
