// BiCodec's speaker encoder — the encode half, which is what voice cloning needs.
//
//   6 s of audio -> mel -> ECAPA-TDNN -> perceiver resampler -> FSQ -> 32 tokens
//
// Those 32 tokens are the same ones bicodec.ts decodes back into a d-vector; the
// difference is where they come from. Generating a voice, the LM invents them.
// Cloning one, they are measured from a recording, and the LM is told what to
// use.
//
// Everything here runs channels-last [B, T, C], where the reference is
// channels-first [B, C, T]. The weights are transposed on load; the axis a
// reduction runs over is the thing to watch.
//
// There is no training path, and BatchNorm below is inference-only by
// construction. That is not a limitation: mlx-audio never leaves training mode,
// so its BatchNorms try to compute batch statistics from a single utterance and
// throw — its cloning path does not run as shipped.
import { MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";

/**
 * BatchNorm at inference, folded to a per-channel affine:
 * `y = (x - mean) / sqrt(var + eps) * w + b`.
 *
 * The running statistics are what a checkpoint stores, so there is nothing to
 * accumulate and no mode to be in.
 */
function batchNorm(W: Weights, p: string, x: MX, eps = 1e-5): MX {
  const scale = W.mx(`${p}.weight`).div(W.mx(`${p}.running_var`).addScalar(eps).sqrt());
  return x.mul(scale).add(W.mx(`${p}.bias`).sub(W.mx(`${p}.running_mean`).mul(scale)));
}

/** A 1-D conv stored [C_out, C_in, K], transposed to the [C_out, K, C_in] MLX wants. */
function convW(W: Weights, p: string): MX {
  return W.mx(`${p}.weight`).transpose([0, 2, 1]);
}

function relu(x: MX): MX {
  return x.maximum(0);
}

/**
 * Conv, ReLU, BatchNorm — in that order.
 *
 * The ReLU comes *before* the norm, which is unusual enough to look like a
 * transcription slip. It is not; the published ECAPA does it this way.
 */
function convReluBn(W: Weights, p: string, x: MX, padding: number, dilation = 1): MX {
  const y = x.conv1d(convW(W, `${p}.conv`), 1, padding, dilation).add(W.mx(`${p}.conv.bias`));
  return batchNorm(W, `${p}.bn`, relu(y));
}

/**
 * Res2Net: split the channels into `scale` groups and run a chain of convs
 * across them, each group adding the previous group's output before its own
 * convolution. The last group is passed through untouched.
 *
 * The effect is several receptive-field sizes inside one block.
 */
function res2ConvReluBn(W: Weights, p: string, x: MX, padding: number, dilation: number, scale = 8): MX {
  const [B, T, C] = x.shape;
  const width = C / scale;
  const part = (i: number) => x.slice([0, 0, i * width], [B, T, (i + 1) * width]);

  let sp = part(0);
  let out: MX | null = null;
  for (let i = 0; i < scale - 1; i++) {
    if (i >= 1) sp = sp.add(part(i));
    sp = sp.conv1d(convW(W, `${p}.convs.${i}`), 1, padding, dilation).add(W.mx(`${p}.convs.${i}.bias`));
    sp = batchNorm(W, `${p}.bns.${i}`, relu(sp));
    out = out ? out.concat(sp, 2) : sp;
  }
  return (out as MX).concat(part(scale - 1), 2);
}

/**
 * Squeeze-and-excitation: average over time, and use that summary to gate each
 * channel. A per-channel attention that ignores position.
 */
function seConnect(W: Weights, p: string, x: MX): MX {
  const [B, T, C] = x.shape;
  const pooled = x.sumAxes([1], false).mulScalar(1 / T); // [B, C]
  let g = relu(pooled.matmul(W.mx(`${p}.linear1.weight`).transpose([1, 0])).add(W.mx(`${p}.linear1.bias`)));
  g = g
    .matmul(W.mx(`${p}.linear2.weight`).transpose([1, 0]))
    .add(W.mx(`${p}.linear2.bias`))
    .sigmoid();
  return x.mul(g.reshape([B, 1, C]));
}

/** 1x1 conv, Res2Net, 1x1 conv, SE gate — added back to the input. */
function seRes2Block(W: Weights, p: string, x: MX, padding: number, dilation: number): MX {
  let h = convReluBn(W, `${p}.se_res2block.0`, x, 0);
  h = res2ConvReluBn(W, `${p}.se_res2block.1`, h, padding, dilation);
  h = convReluBn(W, `${p}.se_res2block.2`, h, 0);
  return x.add(seConnect(W, `${p}.se_res2block.3`, h));
}

/**
 * Attentive statistics pooling: a learned, per-channel weighting over time, then
 * the weighted mean *and* standard deviation.
 *
 * A plain average discards how much a channel varies across the utterance,
 * which is exactly the part that identifies a speaker. With
 * `global_context_att` the attention also sees the utterance-level mean and
 * std, so a frame is scored relative to the whole clip.
 */
function attentiveStatsPool(W: Weights, p: string, x: MX): MX {
  const [B, T, C] = x.shape;
  const inv = 1 / T;
  const mean = x.sumAxes([1], true).mulScalar(inv); // [B, 1, C]
  const varc = x.mul(x).sumAxes([1], true).mulScalar(inv).sub(mean.mul(mean));
  const ctx = mean.broadcastTo([B, T, C]).concat(varc.addScalar(1e-7).sqrt().broadcastTo([B, T, C]), 2);

  let a = x
    .concat(ctx, 2) // [B, T, 3C]
    .conv1d(convW(W, `${p}.linear1`), 1, 0)
    .add(W.mx(`${p}.linear1.bias`))
    .tanh();
  a = a
    .conv1d(convW(W, `${p}.linear2`), 1, 0)
    .add(W.mx(`${p}.linear2.bias`))
    .softmax(1); // over *time*
  const m = a.mul(x).sumAxes([1], false); // [B, C]
  const v = a.mul(x).mul(x).sumAxes([1], false).sub(m.mul(m));
  return m.concat(v.maximum(1e-7).sqrt(), 1); // [B, 2C]
}

/**
 * ECAPA-TDNN. Returns the frame-level features the perceiver consumes, and the
 * utterance-level x-vector.
 *
 * The x-vector is the speaker-verification embedding — useful for *measuring*
 * whether two clips are the same voice. Cloning uses `features`.
 */
export function ecapaTdnn(W: Weights, p: string, mel: MX): { features: MX; xVector: MX } {
  const l1 = convReluBn(W, `${p}.layer1`, mel, 2);
  const l2 = seRes2Block(W, `${p}.layer2`, l1, 2, 2);
  const l3 = seRes2Block(W, `${p}.layer3`, l2, 3, 3);
  const l4 = seRes2Block(W, `${p}.layer4`, l3, 4, 4);

  const cat = l2.concat(l3, 2).concat(l4, 2); // [B, T, 3*512]
  const features = relu(cat.conv1d(convW(W, `${p}.conv`), 1, 0).add(W.mx(`${p}.conv.bias`)));

  let v = attentiveStatsPool(W, `${p}.pool`, features);
  v = batchNorm(W, `${p}.bn`, v);
  const xVector = v.matmul(W.mx(`${p}.linear.weight`).transpose([1, 0])).add(W.mx(`${p}.linear.bias`));
  return { features, xVector };
}

/** RMS norm as the perceiver writes it: L2-normalise, then scale by sqrt(dim). */
function rmsNorm(x: MX, gamma: MX): MX {
  return x.rmsNorm(gamma, 1e-12);
}

/** Cross-attention from a fixed set of latents into the frame features. */
function perceiverAttention(W: Weights, p: string, latents: MX, context: MX, heads: number): MX {
  const [B, N, D] = latents.shape;
  // The latents are prepended to the context, so each one can also see its peers.
  const kvIn = latents.concat(context, 1);
  const M = kvIn.shape[1];
  const inner = W.mx(`${p}.to_q.weight`).shape[0];
  const dh = inner / heads;

  const q = latents
    .matmul(W.mx(`${p}.to_q.weight`).transpose([1, 0]))
    .reshape([B, N, heads, dh])
    .transpose([0, 2, 1, 3]);
  const kv = kvIn.matmul(W.mx(`${p}.to_kv.weight`).transpose([1, 0])); // [B, M, 2*inner]
  const k = kv.slice([0, 0, 0], [B, M, inner]).reshape([B, M, heads, dh]).transpose([0, 2, 1, 3]);
  const v = kv
    .slice([0, 0, inner], [B, M, 2 * inner])
    .reshape([B, M, heads, dh])
    .transpose([0, 2, 1, 3]);

  const o = MX.sdpa(q, k, v, dh ** -0.5, false)
    .transpose([0, 2, 1, 3])
    .reshape([B, N, inner]);
  return o.matmul(W.mx(`${p}.to_out.weight`).transpose([1, 0])).reshape([B, N, D]);
}

/** GEGLU feed-forward. The *second* half is the gate, the first is the value. */
function perceiverFeedForward(W: Weights, p: string, x: MX): MX {
  const [B, N] = x.shape;
  const h = x.matmul(W.mx(`${p}.0.weight`).transpose([1, 0])).add(W.mx(`${p}.0.bias`));
  const inner = h.shape[2] / 2;
  const val = h.slice([0, 0, 0], [B, N, inner]);
  const gate = h.slice([0, 0, inner], [B, N, 2 * inner]);
  return gate
    .gelu()
    .mul(val)
    .matmul(W.mx(`${p}.2.weight`).transpose([1, 0]))
    .add(W.mx(`${p}.2.bias`));
}

/**
 * Perceiver resampler: squeeze a variable-length feature sequence into a fixed
 * 32 latents, whatever the clip's duration.
 *
 * That fixed size is the point — it is what lets a speaker become exactly 32
 * tokens regardless of how long the recording is.
 */
export function perceiverResample(W: Weights, p: string, features: MX, depth = 2, heads = 8): MX {
  const B = features.shape[0];
  const context = features
    .matmul(W.mx(`${p}.proj_context.weight`).transpose([1, 0]))
    .add(W.mx(`${p}.proj_context.bias`));

  const lat = W.mx(`${p}.latents`);
  let latents = lat.reshape([1, lat.shape[0], lat.shape[1]]).broadcastTo([B, lat.shape[0], lat.shape[1]]);
  for (let i = 0; i < depth; i++) {
    latents = latents.add(perceiverAttention(W, `${p}.layers.${i}.0`, latents, context, heads));
    latents = latents.add(perceiverFeedForward(W, `${p}.layers.${i}.1`, latents));
  }
  return rmsNorm(latents, W.mx(`${p}.norm.gamma`));
}

/**
 * Finite scalar quantisation, encode direction: squash each of the 6 dimensions
 * into one of `levels` bins and pack the digits into a single index.
 *
 * The inverse of what SpeakerDetokenizer does, and the arithmetic has to agree
 * digit for digit — the basis is the exclusive cumulative product, so a
 * transposed input scrambles every id without changing the shape.
 */
export function fsqEncode(z: MX, levels = [4, 4, 4, 4, 4, 4]): number[] {
  const D = levels.length;
  const flat = z.toF32();
  const basis: number[] = [];
  for (let i = 0, acc = 1; i < D; i++) {
    basis.push(acc);
    acc *= levels[i];
  }

  const out: number[] = [];
  for (let n = 0; n < flat.length / D; n++) {
    let id = 0;
    for (let d = 0; d < D; d++) {
      const L = levels[d],
        half = Math.floor(L / 2);
      // bound(): tanh into the open interval, shifted so an even level count is
      // centred on a half-integer rather than on a bin edge.
      const halfL = ((L - 1) * (1 + 1e-3)) / 2;
      const offset = L % 2 === 0 ? 0.5 : 0;
      const shift = Math.atanh(offset / halfL);
      const q = Math.round(Math.tanh(flat[n * D + d] + shift) * halfL - offset);
      id += (q + half) * basis[d];
    }
    out.push(id);
  }
  return out;
}

/**
 * A recording -> the 32 global tokens that stand for its speaker.
 *
 * `mel` is `[1, frames, 128]`, from `melSpectrogram` with BiCodec's parameters.
 */
export class SpeakerTokenizer {
  private W: Weights;
  constructor(W: Weights) {
    this.W = W;
  }

  tokenize(mel: MX): number[] {
    return tidy(() => {
      const W = this.W;
      const { features } = ecapaTdnn(W, "speaker_encoder.speaker_encoder", mel);
      const latents = perceiverResample(W, "speaker_encoder.perceiver_sampler", features);
      const q = "speaker_encoder.quantizer";
      const z = latents
        .matmul(W.mx(`${q}.project_in.weight`).transpose([1, 0]))
        .add(W.mx(`${q}.project_in.bias`)); // [1, 32, 6]
      return fsqEncode(z);
    });
  }

  /** The speaker-verification embedding, for measuring similarity between clips. */
  xVector(mel: MX): MX {
    return tidy(() => ecapaTdnn(this.W, "speaker_encoder.speaker_encoder", mel).xVector);
  }
}

/**
 * Spark's volume normalisation: scale so the loud-but-not-peak part of the clip
 * sits near `coeff`, then guard against clipping.
 *
 * Note the quirk, which is load-bearing: `sorted` is taken once, up front, and
 * the quiet-clip rescale below does *not* recompute it. The percentile is
 * therefore measured on the original samples even when the signal was just
 * scaled up. Recomputing it — the obvious "fix" — changes the gain, and with it
 * the mel, and with it every token.
 */
export function volumeNormalize(audio: Float32Array, coeff = 0.2): Float32Array {
  const sorted = Float32Array.from(audio, Math.abs).sort();
  let out = audio;
  if (sorted[sorted.length - 1] < 0.1) {
    const s = Math.max(sorted[sorted.length - 1], 1e-3);
    out = Float32Array.from(audio, (v) => (v / s) * 0.1);
  }

  const loud = sorted.filter((v) => v > 0.01);
  if (loud.length <= 10) return out;

  let sum = 0,
    n = 0;
  for (let i = Math.floor(0.9 * loud.length); i < Math.floor(0.99 * loud.length); i++) {
    sum += loud[i];
    n++;
  }
  const gain = Math.min(10, Math.max(0.1, coeff / (sum / n)));
  out = Float32Array.from(out, (v) => v * gain);

  let peak = 0;
  for (const v of out) peak = Math.max(peak, Math.abs(v));
  return peak > 1 ? Float32Array.from(out, (v) => v / peak) : out;
}

/**
 * The fixed-length window the speaker encoder sees: the first `seconds`,
 * rounded down to a whole number of frames, tiling the clip if it is shorter.
 *
 * Shorter references are repeated rather than padded with silence — silence
 * would be scored as part of the speaker.
 */
export function referenceClip(audio: Float32Array, seconds = 6, sampleRate = 16000, hop = 320): Float32Array {
  const n = Math.floor((sampleRate * seconds) / hop) * hop;
  if (audio.length >= n) return audio.subarray(0, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += audio.length) out.set(audio.subarray(0, Math.min(audio.length, n - i)), i);
  return out;
}
