// BiCodec's decode path — the audio half of Spark-TTS.
//
// Tokens in, waveform out, in four stages:
//
//   semantic tokens -> quantizer   -> z_q       [B, T, 1024]
//   global tokens   -> speaker FSQ -> d_vector  [B, 1024]
//   z_q + d_vector  -> prenet      -> features  [B, T, 1024]
//   features        -> generator   -> waveform  [B, T*320]
//
// Only the decode direction is implemented: encoding needs the ECAPA speaker
// encoder and the perceiver sampler, some 400 tensors of the checkpoint that
// text-to-speech never touches.
//
// Weights are the published BiCodec checkpoint, whose weight-norm is *not*
// fused — unlike the EnCodec weights in encodec.ts, which arrive pre-fused. So
// convolutions here reconstruct w = g * v / ||v|| on load.
import { fromF32, MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";

/**
 * PyTorch weight-norm, undone: `w = g * v / ||v||`, with the norm taken over
 * every dimension but the output channel.
 *
 * Stored [C_out, C_in, K] and returned [C_out, K, C_in], which is the layout
 * MLX's conv1d wants.
 */
export function weightNormConv(W: Weights, name: string): MX {
  const v = W.mx(`${name}.weight_v`); // [C_out, C_in, K]
  const g = W.mx(`${name}.weight_g`); // [C_out, 1, 1]
  const norm = v.mul(v).sumAxes([1, 2], true).sqrt();
  return v.mul(g).div(norm).transpose([0, 2, 1]);
}

export class BiCodecQuantizer {
  private W: Weights;
  constructor(W: Weights) {
    this.W = W;
  }

  /**
   * Semantic token ids `[B, T]` -> latents `[B, T, 1024]`.
   *
   * A plain codebook lookup followed by a 1x1 convolution. The codebook is
   * eight-dimensional; out_project widens it to the feature size the prenet
   * expects.
   */
  detokenize(ids: MX): MX {
    return tidy(() => {
      const [B, T] = ids.shape;
      const codes = this.W.mx("quantizer.codebook.weight")
        .takeAxis(ids.reshape([B * T]), 0)
        .reshape([B, T, 8]);
      return codes
        .conv1d(weightNormConv(this.W, "quantizer.out_project"), 1, 0)
        .add(this.W.mx("quantizer.out_project.bias"));
    });
  }
}

/**
 * Global tokens -> the speaker embedding the whole utterance is conditioned on.
 *
 * These are finite-scalar-quantised rather than looked up in a codebook: each
 * index packs six digits in mixed radix (levels 4^6 = 4096), and decoding is
 * arithmetic, not a table. Done on the host — there are 32 tokens, and the
 * integer division and modulo would need ops MX does not expose.
 */
export class SpeakerDetokenizer {
  private W: Weights;
  private levels: number[];

  constructor(W: Weights, levels = [4, 4, 4, 4, 4, 4]) {
    this.W = W;
    this.levels = levels;
  }

  /** Global token ids `[B, N]` -> d-vector `[B, 1024]`. */
  detokenize(ids: number[][]): MX {
    return tidy(() => {
      const L = this.levels,
        D = L.length;
      // basis is the exclusive cumulative product: [1, 4, 16, 64, 256, 1024].
      const basis: number[] = [];
      let acc = 1;
      for (let i = 0; i < D; i++) {
        basis.push(acc);
        acc *= L[i];
      }

      const B = ids.length,
        N = ids[0].length;
      const codes = new Float32Array(B * N * D);
      for (let b = 0; b < B; b++) {
        for (let n = 0; n < N; n++) {
          for (let d = 0; d < D; d++) {
            const level = Math.floor(ids[b][n] / basis[d]) % L[d];
            const half = Math.floor(L[d] / 2);
            codes[(b * N + n) * D + d] = (level - half) / half; // back to [-1, 1]
          }
        }
      }

      const q = "speaker_encoder.quantizer";
      const zq = fromF32(codes, [B, N, D])
        .matmul(this.W.mx(`${q}.project_out.weight`).transpose([1, 0]))
        .add(this.W.mx(`${q}.project_out.bias`)); // [B, N, 128]

      // The reference swaps to [B, 128, N] before flattening, so the 4096
      // values are channel-major. Flattening token-major instead gives the
      // right shape and the wrong vector.
      return zq
        .transpose([0, 2, 1])
        .reshape([B, N * zq.shape[2]])
        .matmul(this.W.mx("speaker_encoder.project.weight").transpose([1, 0]))
        .add(this.W.mx("speaker_encoder.project.bias")); // [B, 1024]
    });
  }
}

/**
 * Adaptive layer norm: normalise, then scale and shift by projections of a
 * condition vector.
 *
 * This is where the speaker embedding actually enters the prenet. The
 * checkpoint gives it away — `norm.scale.weight [384, 1024]` takes the 1024-wide
 * d-vector — but a plain LayerNorm has the right shape for `norm.weight [384]`
 * and would load without complaint on the *other* backbones, so the two are
 * easy to conflate.
 */
function adaLayerNorm(W: Weights, p: string, x: MX, cond: MX): MX {
  const scale = cond.matmul(W.mx(`${p}.scale.weight`).transpose([1, 0])).add(W.mx(`${p}.scale.bias`));
  const shift = cond.matmul(W.mx(`${p}.shift.weight`).transpose([1, 0])).add(W.mx(`${p}.shift.bias`));
  const [B, , D] = x.shape;
  const n = x.layerNorm(null, null, 1e-6);
  return n.mul(scale.reshape([B, 1, D])).add(shift.reshape([B, 1, D]));
}

/** A ConvNeXt block: depthwise conv, norm, an inverted bottleneck, layer scale. */
function convNextBlock(W: Weights, p: string, x: MX, cond: MX | null): MX {
  // Depthwise: one 7-tap kernel per channel, stored [C, 1, K].
  const dw = W.mx(`${p}.dwconv.weight`).transpose([0, 2, 1]);
  const C = dw.shape[0];
  let y = x.conv1d(dw, 1, 3, 1, C).add(W.mx(`${p}.dwconv.bias`));
  y = cond
    ? adaLayerNorm(W, `${p}.norm`, y, cond)
    : y.layerNorm(W.mx(`${p}.norm.weight`), W.mx(`${p}.norm.bias`), 1e-6);
  y = y.matmul(W.mx(`${p}.pwconv1.weight`).transpose([1, 0])).add(W.mx(`${p}.pwconv1.bias`));
  y = y.gelu();
  y = y.matmul(W.mx(`${p}.pwconv2.weight`).transpose([1, 0])).add(W.mx(`${p}.pwconv2.bias`));
  return x.add(y.mul(W.mx(`${p}.gamma`))); // layer scale, then residual
}

/** Vocos backbone: an input conv, a norm, N ConvNeXt blocks, a final norm. */
function vocosBackbone(W: Weights, p: string, x: MX, layers: number, cond: MX | null): MX {
  const embed = W.mx(`${p}.embed.weight`).transpose([0, 2, 1]);
  let h = x.conv1d(embed, 1, 3).add(W.mx(`${p}.embed.bias`));
  h = cond
    ? adaLayerNorm(W, `${p}.norm`, h, cond)
    : h.layerNorm(W.mx(`${p}.norm.weight`), W.mx(`${p}.norm.bias`), 1e-6);
  for (let i = 0; i < layers; i++) h = convNextBlock(W, `${p}.convnext.${i}`, h, cond);
  return h.layerNorm(W.mx(`${p}.final_layer_norm.weight`), W.mx(`${p}.final_layer_norm.bias`), 1e-6);
}

/**
 * The prenet: latents -> the features the wave generator consumes.
 *
 * Two downsample stages and a deep Vocos backbone. With sample_ratios [1, 1]
 * the sampling blocks carry no weights, but they are NOT identity: each sums
 * three copies of its input, so the signal is tripled. Reading that as a
 * no-op is the obvious mistake and produces a quiet, wrong result.
 */
export class BiCodecPrenet {
  private W: Weights;
  private layers: number;

  /** `layers` is vocos_num_layers from the checkpoint's config.yaml. */
  constructor(W: Weights, layers = 12) {
    this.W = W;
    this.layers = layers;
  }

  forward(zq: MX, cond: MX): MX {
    return tidy(() => {
      const W = this.W;
      let x = zq
        .matmul(W.mx("prenet.linear_pre.weight").transpose([1, 0]))
        .add(W.mx("prenet.linear_pre.bias")); // [B, T, 384]

      for (let i = 0; i < 2; i++) {
        x = x.mulScalar(3); // the sampling block
        x = vocosBackbone(W, `prenet.downsample.${i}.1`, x, 2, null);
      }

      // The deep backbone is conditioned on the speaker; the shallow ones are not.
      x = vocosBackbone(W, "prenet.vocos_backbone", x, this.layers, cond);
      return x.matmul(W.mx("prenet.linear.weight").transpose([1, 0])).add(W.mx("prenet.linear.bias")); // [B, T, 1024]
    });
  }
}

/**
 * PyTorch weight-norm on a *transposed* convolution.
 *
 * Same `w = g * v / ||v||`, but the stored layout is [C_in, C_out, K] — the
 * input channel leads, so that is the axis the norm is taken over, and the
 * per-channel gain has shape [C_in, 1, 1]. Returned [C_out, K, C_in].
 */
export function weightNormConvTranspose(W: Weights, name: string): MX {
  const v = W.mx(`${name}.weight_v`); // [C_in, C_out, K]
  const g = W.mx(`${name}.weight_g`); // [C_in, 1, 1]
  const norm = v.mul(v).sumAxes([1, 2], true).sqrt();
  return v.mul(g).div(norm).transpose([1, 2, 0]);
}

/**
 * Snake: `x + sin(ax)^2 / a`, a periodic activation with a learned per-channel
 * frequency. Vocoders use it because plain ReLUs cannot extrapolate a periodic
 * signal, and speech is periodic.
 *
 * `alpha` is stored channels-first [1, C, 1]; we run channels-last.
 */
function snake(x: MX, alpha: MX): MX {
  const a = alpha.transpose([0, 2, 1]); // [1, 1, C]
  return x.add(x.mul(a).sin().square().div(a.addScalar(1e-9)));
}

/** Snake, a dilated 7-tap conv, Snake, a 1x1 conv — added back to the input. */
function residualUnit(W: Weights, p: string, x: MX, dilation: number): MX {
  let y = snake(x, W.mx(`${p}.block.0.alpha`));
  y = y
    .conv1d(weightNormConv(W, `${p}.block.1`), 1, (6 * dilation) / 2, dilation)
    .add(W.mx(`${p}.block.1.bias`));
  y = snake(y, W.mx(`${p}.block.2.alpha`));
  y = y.conv1d(weightNormConv(W, `${p}.block.3`), 1, 0).add(W.mx(`${p}.block.3.bias`));
  return x.add(y);
}

/** One upsampling stage: Snake, a transposed conv, three residual units. */
function decoderBlock(W: Weights, p: string, x: MX, kernel: number, stride: number): MX {
  let h = snake(x, W.mx(`${p}.block.0.alpha`));
  h = h
    .convTranspose1d(weightNormConvTranspose(W, `${p}.block.1`), stride, (kernel - stride) / 2)
    .add(W.mx(`${p}.block.1.bias`));
  for (const [i, d] of [
    [2, 1],
    [3, 3],
    [4, 9],
  ] as const)
    h = residualUnit(W, `${p}.block.${i}`, h, d);
  return h;
}

/**
 * The wave generator: features `[B, T, 1024]` -> waveform `[B, T*320, 1]`.
 *
 * Four transposed-convolution stages at rates 8, 5, 4 and 2 — 320 samples per
 * frame, so 16 kHz audio from 50 frames a second.
 *
 * Note for anyone diffing against mlx-audio: its `WNConvTranspose1d` passes
 * `groups` positionally into `conv_transpose1d`'s `output_padding` slot, so it
 * emits one extra sample per stage (5171 rather than 5120 for 16 frames). This
 * follows PyTorch, which is what the checkpoint was trained as.
 */
export class WaveGenerator {
  private W: Weights;
  private rates = [8, 5, 4, 2];
  private kernels = [16, 11, 8, 4];

  constructor(W: Weights) {
    this.W = W;
  }

  /** `[B, T, 1024]` in, `[B, T*320, 1]` in [-1, 1] out. */
  forward(x: MX): MX {
    const W = this.W;
    return tidy(() => {
      let h = tidy(() =>
        x.conv1d(weightNormConv(W, "decoder.model.0"), 1, 3).add(W.mx("decoder.model.0.bias")),
      );

      // Each stage is 8x to 2x wider than the last, so the previous activation
      // is freed the moment it is consumed rather than at the end.
      for (let i = 0; i < this.rates.length; i++) {
        const prev = h;
        h = tidy(() => decoderBlock(W, `decoder.model.${i + 1}`, prev, this.kernels[i], this.rates[i]));
        prev.free();
      }

      const last = h;
      h = tidy(() =>
        snake(last, W.mx("decoder.model.5.alpha"))
          .conv1d(weightNormConv(W, "decoder.model.6"), 1, 3)
          .add(W.mx("decoder.model.6.bias"))
          .tanh(),
      );
      last.free();
      return h;
    });
  }
}
