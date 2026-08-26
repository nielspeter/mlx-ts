// EnCodec decoder — audio tokens back to a waveform.
//
// This is the second half of MusicGen: the LM emits codebook indices, and this
// turns them into samples. Structure follows Apple's mlx-examples/musicgen
// encodec.py, and the weights are mlx-community/encodec-32khz-float32, whose
// weight-norm is already fused (conv.weight/bias only, no weight_g/weight_v).
//
// The LSTM is the interesting part: MLX has no fused one, so it runs on the
// custom Metal kernel in src/ffi/kernel.ts — the reason that spike existed.
import { MX, fromF32, fromI32, tidy, scalar, stack } from "../core/mx.ts";
import { conv1d, convTranspose1d, exp } from "../ffi/generated.ts";
import { metalKernel, scalarI32 } from "../ffi/kernel.ts";
import type { Weights } from "../io/loader.ts";

// ELU, which mlx-c has no op for: x > 0 ? x : exp(x) - 1.
const elu = (x: MX): MX => {
  const pos = x.greater(scalar(0));
  return pos.where(x, new MX(exp(x.h)).sub(scalar(1)));
};

// EnCodec pads by REFLECTION, not zeros (config.pad_mode = "reflect"), and MLX
// has no reflect mode — so mirror the edges by hand, exactly as the reference
// does: prefix = reverse(x[1 : left+1]), suffix = reverse(x[L-right-1 : L-1]).
// Getting this wrong is silent: the shapes are identical and only the numbers
// move.
// Reverse along time. Done with an index gather rather than a negative slice
// stride, because MLX's start/stop semantics under a negative stride are not
// the same as Python's and get this silently wrong.
const reverseTime = (x: MX): MX =>
  x.takeAxis(fromI32(Int32Array.from({ length: x.shape[1] }, (_, i) => x.shape[1] - 1 - i), [x.shape[1]]), 1);

function padTime(x: MX, left: number, right: number): MX {
  if (left === 0 && right === 0) return x;
  const [B, L, C] = x.shape;
  const parts: MX[] = [];
  if (left > 0) parts.push(reverseTime(x.slice([0, 1, 0], [B, left + 1, C])));
  parts.push(x);
  if (right > 0) parts.push(reverseTime(x.slice([0, Math.max(L - (right + 1), 0), 0], [B, L - 1, C])));
  return parts.length === 1 ? x : parts.reduce((a, b) => a.concat(b, 1));
}

// --- EnCodec's convolutions ----------------------------------------------
// Non-causal padding: split padding_total across both sides, and add whatever
// extra the stride needs so the last frame is whole.
function encConv1d(x: MX, w: MX, b: MX, kernel: number, stride: number, dilation: number): MX {
  const padTotal = (kernel - 1) * dilation - (stride - 1);
  const length = x.shape[1];
  const nFrames = Math.ceil((length - kernel + padTotal) / stride + 1) - 1;
  const extra = nFrames * stride + kernel - padTotal - length;
  const right = Math.floor(padTotal / 2);
  const left = padTotal - right;
  const y = new MX(conv1d(padTime(x, left, right + extra).h, w.h, stride, 0, dilation, 1));
  return y.add(b.reshape([1, 1, b.shape[0]]));
}

// Transposed conv trims AFTER the convolution rather than padding before it.
function encConvT1d(x: MX, w: MX, b: MX, kernel: number, stride: number): MX {
  const y = new MX(convTranspose1d(x.h, w.h, stride, 0, 1, 0, 1));
  const withBias = y.add(b.reshape([1, 1, b.shape[0]]));
  const padTotal = kernel - stride;
  const right = Math.floor(padTotal / 2);
  const left = padTotal - right;
  const [B, L, C] = withBias.shape;
  return withBias.slice([0, left, 0], [B, L - right, C]);
}

// --- the LSTM, on a custom Metal kernel ----------------------------------
const lstmStep = metalKernel({
  name: "lstm",
  inputNames: ["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
  outputNames: ["hidden_state", "cell_state"],
  header: `
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
  `,
  source: `
        uint b = thread_position_in_grid.x;
        uint d = hidden_size * 4;
        uint elem = b * d + thread_position_in_grid.y;
        uint index = elem;
        uint x_index = b * num_time_steps * d + time_step * d + index;
        auto i = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size; x_index += hidden_size;
        auto f = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size; x_index += hidden_size;
        auto g = metal::precise::tanh(h_in[index] + x[x_index]);
        index += hidden_size; x_index += hidden_size;
        auto o = sigmoid(h_in[index] + x[x_index]);
        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * metal::precise::tanh(cell_state[elem]);
  `,
});

/** One LSTM layer over [B, T, D]; Wx/Wh are [4H, D] and bias is [4H]. */
function lstmLayer(x: MX, Wx: MX, Wh: MX, bias: MX): MX {
  const [B, T] = x.shape;
  const H = Wh.shape[1];
  // Precompute x @ Wx^T + b for every timestep; the kernel only does the
  // recurrence, which is the part that cannot be batched over time.
  const xProj = x.matmul(Wx.transpose([1, 0])).add(bias.reshape([1, 1, 4 * H]));

  let h = fromF32(new Float32Array(B * H), [B, H]);
  let c = fromF32(new Float32Array(B * H), [B, H]);
  const outs: MX[] = [];
  const nT = scalarI32(T);
  for (let t = 0; t < T; t++) {
    const hProj = h.matmul(Wh.transpose([1, 0]));               // [B, 4H]
    const [hNext, cNext] = lstmStep.apply(
      [xProj, hProj, c, scalarI32(H), scalarI32(t), nT],
      [{ shape: [B, H] }, { shape: [B, H] }],
      // grid.y is h_in.size / 4 == B*H, one thread per hidden unit — NOT B*4H.
      // Four times too many threads writes past each row and the result is
      // quietly wrong rather than a crash.
      [B, B * H, 1], [256, 1, 1],
    );
    h = hNext; c = cNext;
    outs.push(h);
    if (t % 64 === 63) { h.eval(); c.eval(); }                   // bound the graph
  }
  return stack(outs, 1);                                          // [B, T, H]
}

export type EncodecConfig = {
  hidden_size: number; num_filters: number; upsampling_ratios: number[];
  kernel_size: number; last_kernel_size: number; residual_kernel_size: number;
  dilation_growth_rate: number; num_residual_layers: number; compress?: number;
  codebook_size: number; sampling_rate: number;
};

/**
 * EnCodec's decoder half: residual-vector-quantizer codes -> waveform.
 * `codes` is [B, num_codebooks, T] of integer indices.
 */
export class EncodecDecoder {
  cfg: EncodecConfig;
  W: Weights;
  constructor(cfg: EncodecConfig, W: Weights) { this.cfg = cfg; this.W = W; }

  private conv(i: number, x: MX, kernel: number, stride = 1, dilation = 1): MX {
    return encConv1d(x, this.W.mx(`decoder.layers.${i}.conv.weight`), this.W.mx(`decoder.layers.${i}.conv.bias`), kernel, stride, dilation);
  }
  private convT(i: number, x: MX, kernel: number, stride: number): MX {
    return encConvT1d(x, this.W.mx(`decoder.layers.${i}.conv.weight`), this.W.mx(`decoder.layers.${i}.conv.bias`), kernel, stride);
  }
  // ELU -> conv(k=3, dilated) -> ELU -> conv(k=1), added to the input.
  private resBlock(i: number, x: MX, dilation: number): MX {
    const k = this.cfg.residual_kernel_size;
    let h = encConv1d(elu(x), this.W.mx(`decoder.layers.${i}.block.1.conv.weight`), this.W.mx(`decoder.layers.${i}.block.1.conv.bias`), k, 1, dilation);
    h = encConv1d(elu(h), this.W.mx(`decoder.layers.${i}.block.3.conv.weight`), this.W.mx(`decoder.layers.${i}.block.3.conv.bias`), 1, 1, 1);
    return x.add(h);
  }

  /** Sum the per-codebook embeddings: [B, K, T] indices -> [B, T, D]. */
  dequantize(codes: MX): MX {
    const [B, K, T] = codes.shape;
    let sum: MX | null = null;
    for (let k = 0; k < K; k++) {
      const idx = codes.slice([0, k, 0], [B, k + 1, T]).reshape([B * T]);
      const emb = this.W.mx(`quantizer.layers.${k}.codebook.embed`);   // [size, D]
      const picked = emb.takeAxis(idx, 0).reshape([B, T, emb.shape[1]]);
      sum = sum ? sum.add(picked) : picked;
    }
    return sum!;
  }

  /** codes [B, K, T] -> waveform [B, samples]. */
  decode(codes: MX): MX {
    return tidy(() => {
      let x = this.dequantize(codes);
      x = this.conv(0, x, this.cfg.kernel_size);

      // Two stacked LSTM layers, residual around the pair.
      let h = x;
      for (const l of [0, 1]) {
        h = lstmLayer(h, this.W.mx(`decoder.layers.1.lstm.${l}.Wx`), this.W.mx(`decoder.layers.1.lstm.${l}.Wh`), this.W.mx(`decoder.layers.1.lstm.${l}.bias`));
      }
      x = x.add(h);

      // Upsample: ELU -> transposed conv -> residual block, once per ratio.
      let layer = 3;
      for (const ratio of this.cfg.upsampling_ratios) {
        x = this.convT(layer, elu(x), ratio * 2, ratio);
        x = this.resBlock(layer + 1, x, 1);
        layer += 3;
      }
      return this.conv(15, elu(x), this.cfg.last_kernel_size).reshape([codes.shape[0], -1]);
    });
  }
}
