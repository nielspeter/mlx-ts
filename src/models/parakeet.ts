// Parakeet TDT — NVIDIA's FastConformer transducer, the streaming-native
// counterpart to Whisper.
//
// Whisper is an encoder-decoder over a fixed 30 s window: to follow live speech
// you re-transcribe a sliding segment and stitch the results. Parakeet is a
// transducer — the encoder runs once over whatever audio exists, and the decoder
// emits tokens frame by frame — so it streams by construction.
//
//   audio -> mel -> subsampling (8x) -> FastConformer x24 -> encoder states
//
// Checked against `transformers.ParakeetForTDT`, the implementation NVIDIA
// published these weights for, rather than another MLX port.
import { fromF32, fromI32, type MX, tidy } from "../core/mx.ts";
import type { Weights } from "../io/loader.ts";

export type ParakeetEncoderConfig = {
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  intermediate_size: number;
  num_mel_bins: number;
  conv_kernel_size: number;
  subsampling_conv_channels: number;
  subsampling_conv_kernel_size: number;
  subsampling_conv_stride: number;
  subsampling_factor: number;
};

/** A conv2d weight stored [C_out, C_in/groups, KH, KW] -> the [C_out, KH, KW, C_in] MLX wants. */
function conv2dW(W: Weights, name: string): MX {
  return W.mx(`${name}.weight`).transpose([0, 2, 3, 1]);
}

const relu = (x: MX): MX => x.maximum(0);

/**
 * The 8x subsampling stem: mel frames in, encoder-width states out.
 *
 * The mel is treated as a single-channel image [B, 1, time, mels], and every
 * stride-2 convolution halves *both* axes — so 8x fewer frames and 8x fewer
 * frequency bins. With 128 mels that leaves 16, and 256 channels x 16 = 4096,
 * which is exactly what the projection consumes.
 *
 * The first convolution is dense; the other two are depthwise plus pointwise,
 * which is where "FastConformer" gets most of its speed over a plain Conformer.
 */
export function subsample(W: Weights, cfg: ParakeetEncoderConfig, mel: MX): MX {
  const p = "encoder.subsampling";
  const stride: [number, number] = [cfg.subsampling_conv_stride, cfg.subsampling_conv_stride];
  const pad = (cfg.subsampling_conv_kernel_size - 1) >> 1;
  const padding: [number, number] = [pad, pad];
  const C = cfg.subsampling_conv_channels;

  const [B, T, M] = mel.shape;
  // [B, T, mels] -> [B, T, mels, 1]: channels-last, one channel.
  let h = mel.reshape([B, T, M, 1]);
  h = relu(h.conv2d(conv2dW(W, `${p}.layers.0`), stride, padding).add(W.mx(`${p}.layers.0.bias`)));

  // layers 2/3 and 5/6 are depthwise+pointwise pairs; 1/4/7 are the ReLUs.
  for (const [dw, pw] of [
    [2, 3],
    [5, 6],
  ] as const) {
    h = h
      .conv2d(conv2dW(W, `${p}.layers.${dw}`), stride, padding, [1, 1], C)
      .add(W.mx(`${p}.layers.${dw}.bias`));
    h = h.conv2d(conv2dW(W, `${p}.layers.${pw}`), [1, 1], [0, 0]).add(W.mx(`${p}.layers.${pw}.bias`));
    h = relu(h);
  }

  // Flatten channel-major, as the reference does: it transposes to
  // [B, T', C, freq] before reshaping, so channel is the outer axis. Ours is
  // [B, T', freq, C], so it needs the same swap — flattening as-is gives the
  // right shape and interleaves the features wrongly.
  const [, T2, F2, C2] = h.shape;
  h = h.transpose([0, 1, 3, 2]).reshape([B, T2, C2 * F2]);
  return h.matmul(W.mx(`${p}.linear.weight`).transpose([1, 0])).add(W.mx(`${p}.linear.bias`));
}

/**
 * Relative position embeddings, `[1, 2L-1, D]`.
 *
 * One row per *offset* between two frames, from +(L-1) to -(L-1), rather than
 * one per absolute position. That is what lets the encoder generalise past the
 * lengths it saw in training — and what makes the attention below more than a
 * standard one.
 *
 * sin and cos are interleaved, not concatenated: [sin0, cos0, sin1, cos1, ...].
 */
export function relPositionalEncoding(D: number, L: number): MX {
  const P = 2 * L - 1;
  const half = D >> 1;
  const out = new Float32Array(P * D);
  for (let p = 0; p < P; p++) {
    const pos = L - 1 - p; // +(L-1) down to -(L-1)
    for (let i = 0; i < half; i++) {
      const f = pos / 10000 ** ((2 * i) / D);
      out[p * D + 2 * i] = Math.sin(f);
      out[p * D + 2 * i + 1] = Math.cos(f);
    }
  }
  return fromF32(out, [1, P, D]);
}

/** A bias-free linear: x @ W^T, weights stored [out, in]. */
const lin = (W: Weights, name: string, x: MX): MX => x.matmul(W.mx(`${name}.weight`).transpose([1, 0]));

/**
 * Transformer-XL relative attention.
 *
 * Scores are the sum of two parts. The content term is the usual q·k, with a
 * learned per-head bias `bias_u` added to the query. The position term pairs a
 * differently-biased query (`bias_v`) with keys projected from the relative
 * embeddings, giving one score per *offset*; `relShift` then slides that
 * offset-indexed matrix into absolute (query, key) coordinates.
 *
 * MX.sdpa takes no additive mask, so the two terms are combined by hand.
 */
function attention(W: Weights, p: string, x: MX, pos: MX, heads: number): MX {
  const [B, L, D] = x.shape;
  const dh = D / heads;
  const scale = dh ** -0.5;

  const shape = (t: MX, n: number) => t.reshape([B, n, heads, dh]).transpose([0, 2, 1, 3]);
  const q = shape(lin(W, `${p}.q_proj`, x), L);
  const k = shape(lin(W, `${p}.k_proj`, x), L);
  const v = shape(lin(W, `${p}.v_proj`, x), L);

  const bias = (n: string) => W.mx(`${p}.${n}`).reshape([1, heads, 1, dh]);
  const qu = q.add(bias("bias_u"));
  const qv = q.add(bias("bias_v"));

  const P = pos.shape[1];
  // [1, P, D] -> [1, heads, dh, P], so q_v @ relK gives a score per offset.
  const relK = lin(W, `${p}.relative_k_proj`, pos).reshape([1, P, heads, dh]).transpose([0, 2, 3, 1]);

  let bd = qv.matmul(relK.broadcastTo([B, heads, dh, P])); // [B, H, L, P]
  bd = relShift(bd).slice([0, 0, 0, 0], [B, heads, L, L]).mulScalar(scale);

  const ac = qu.matmul(k.transpose([0, 1, 3, 2])).mulScalar(scale);
  const attn = ac.add(bd).softmax(3);
  const o = attn.matmul(v).transpose([0, 2, 1, 3]).reshape([B, L, D]);
  return lin(W, `${p}.o_proj`, o);
}

/**
 * Slide an offset-indexed score matrix into absolute coordinates.
 *
 * Pad one column on the left, reinterpret the last two axes transposed, drop the
 * first row, and reinterpret back — the standard Transformer-XL trick (appendix
 * B of the paper). It costs one pad and two reshapes instead of building an
 * L x L gather.
 */
function relShift(scores: MX): MX {
  const [B, H, L, P] = scores.shape;
  const zeros = fromF32(new Float32Array(B * H * L), [B, H, L, 1]);
  return zeros
    .concat(scores, 3) // [B, H, L, P+1]
    .reshape([B, H, P + 1, L])
    .slice([0, 0, 1, 0], [B, H, P + 1, L])
    .reshape([B, H, L, P]);
}

/** Conformer convolution module: gated 1x1, depthwise, BatchNorm, SiLU, 1x1. */
function convModule(W: Weights, p: string, x: MX): MX {
  const [B, T, C] = x.shape;
  // pointwise_conv1 doubles the channels; GLU halves them again, one half
  // gating the other.
  const h = x.conv1d(W.mx(`${p}.pointwise_conv1.weight`).transpose([0, 2, 1]), 1, 0);
  const a = h.slice([0, 0, 0], [B, T, C]);
  const b = h.slice([0, 0, C], [B, T, 2 * C]);
  let y = a.mul(b.sigmoid());

  const dw = W.mx(`${p}.depthwise_conv.weight`).transpose([0, 2, 1]); // [C, K, 1]
  const K = dw.shape[1];
  y = y.conv1d(dw, 1, (K - 1) >> 1, 1, C);
  y = batchNorm(W, `${p}.norm`, y).silu();
  return y.conv1d(W.mx(`${p}.pointwise_conv2.weight`).transpose([0, 2, 1]), 1, 0);
}

/** BatchNorm at inference, folded to a per-channel affine from the running stats. */
function batchNorm(W: Weights, p: string, x: MX, eps = 1e-5): MX {
  const scale = W.mx(`${p}.weight`).div(W.mx(`${p}.running_var`).addScalar(eps).sqrt());
  return x.mul(scale).add(W.mx(`${p}.bias`).sub(W.mx(`${p}.running_mean`).mul(scale)));
}

const layerNorm = (W: Weights, p: string, x: MX): MX =>
  x.layerNorm(W.mx(`${p}.weight`), W.mx(`${p}.bias`), 1e-5);

/** Macaron feed-forward: two of these bracket every block, each at half weight. */
const feedForward = (W: Weights, p: string, x: MX): MX =>
  lin(W, `${p}.linear2`, lin(W, `${p}.linear1`, x).silu());

/**
 * One FastConformer block: half-FF, attention, convolution, half-FF, norm.
 *
 * The 0.5 on each feed-forward is the Macaron structure from the Conformer
 * paper — two half-weight FFs around the attention and convolution rather than
 * one full-weight FF after them.
 */
function encoderBlock(W: Weights, p: string, x: MX, pos: MX, heads: number): MX {
  let h = x.add(
    feedForward(W, `${p}.feed_forward1`, layerNorm(W, `${p}.norm_feed_forward1`, x)).mulScalar(0.5),
  );
  h = h.add(attention(W, `${p}.self_attn`, layerNorm(W, `${p}.norm_self_att`, h), pos, heads));
  h = h.add(convModule(W, `${p}.conv`, layerNorm(W, `${p}.norm_conv`, h)));
  h = h.add(feedForward(W, `${p}.feed_forward2`, layerNorm(W, `${p}.norm_feed_forward2`, h)).mulScalar(0.5));
  return layerNorm(W, `${p}.norm_out`, h);
}

/** Mel `[1, frames, 128]` -> encoder states `[1, frames/8, 1024]`. */
export function encode(W: Weights, cfg: ParakeetEncoderConfig, mel: MX): MX {
  return tidy(() => {
    let h = subsample(W, cfg, mel);
    const pos = relPositionalEncoding(cfg.hidden_size, h.shape[1]);
    for (let i = 0; i < cfg.num_hidden_layers; i++) {
      h = encoderBlock(W, `encoder.layers.${i}`, h, pos, cfg.num_attention_heads);
    }
    return h;
  });
}

export type ParakeetConfig = {
  encoder_config: ParakeetEncoderConfig;
  decoder_hidden_size: number;
  num_decoder_layers: number;
  vocab_size: number;
  blank_token_id: number;
  durations: number[];
  max_symbols_per_step: number;
};

/** One LSTM cell step. PyTorch packs the gates as [input, forget, cell, output]. */
function lstmStep(W: Weights, p: string, layer: number, x: MX, h: MX, c: MX): { h: MX; c: MX } {
  const g = x
    .matmul(W.mx(`${p}.weight_ih_l${layer}`).transpose([1, 0]))
    .add(W.mx(`${p}.bias_ih_l${layer}`))
    .add(h.matmul(W.mx(`${p}.weight_hh_l${layer}`).transpose([1, 0])))
    .add(W.mx(`${p}.bias_hh_l${layer}`));
  const H = h.shape[1];
  const part = (n: number) => g.slice([0, n * H], [1, (n + 1) * H]);
  const i = part(0).sigmoid(),
    f = part(1).sigmoid(),
    cc = part(2).tanh(),
    o = part(3).sigmoid();
  const cNext = f.mul(c).add(i.mul(cc));
  return { h: o.mul(cNext.tanh()), c: cNext };
}

/** The prediction network's recurrent state — one (h, c) pair per LSTM layer. */
export type PredictorState = { h: MX; c: MX }[];

/**
 * Prediction network: the transducer's language model.
 *
 * It sees only the tokens emitted so far, never the audio — that separation is
 * what lets the joint network below combine "what has been said" with "what is
 * being heard" one frame at a time, instead of attending over a whole window.
 */
export function predict(
  W: Weights,
  cfg: ParakeetConfig,
  token: number,
  state: PredictorState,
): { out: MX; state: PredictorState } {
  const emb = W.mx("decoder.embedding.weight").takeAxis(fromI32(Int32Array.from([token]), [1]), 0); // [1, 640]
  let x = emb;
  const next: PredictorState = [];
  for (let l = 0; l < cfg.num_decoder_layers; l++) {
    const s = lstmStep(W, "decoder.lstm", l, x, state[l].h, state[l].c);
    next.push(s);
    x = s.h;
  }
  const out = x
    .matmul(W.mx("decoder.decoder_projector.weight").transpose([1, 0]))
    .add(W.mx("decoder.decoder_projector.bias"));
  return { out, state: next };
}

/** A zeroed predictor state, for the start of an utterance. */
export function initialState(cfg: ParakeetConfig): PredictorState {
  const z = () => fromF32(new Float32Array(cfg.decoder_hidden_size), [1, cfg.decoder_hidden_size]);
  return Array.from({ length: cfg.num_decoder_layers }, () => ({ h: z(), c: z() }));
}

/**
 * Joint network: add the projected encoder frame to the predictor output, ReLU,
 * and read off both a token and a duration.
 *
 * The duration head is what makes this TDT rather than plain RNN-T. A standard
 * transducer advances one encoder frame per blank; here the model says how many
 * frames to skip, so silence and steady vowels cost one step instead of many.
 * That is most of why it is ~49x faster than Whisper.
 */
export function joint(W: Weights, encFrame: MX, decOut: MX): MX {
  return relu(encFrame.add(decOut))
    .matmul(W.mx("joint.head.weight").transpose([1, 0]))
    .add(W.mx("joint.head.bias"));
}

/** Encoder states `[1, T, 1024]` -> the projected `[1, T, 640]` the joint consumes. */
export const projectEncoder = (W: Weights, enc: MX): MX =>
  enc.matmul(W.mx("encoder_projector.weight").transpose([1, 0])).add(W.mx("encoder_projector.bias"));

/** A token and where in the audio it came from. One encoder frame is 80 ms. */
export type TimedToken = {
  id: number;
  /** Encoder frame the joint network emitted this token at. */
  frame: number;
  /**
   * Frames the model then skipped — its own estimate of the token's extent.
   *
   * This is what makes TDT different from a plain transducer: the duration head
   * is trained, not inferred. A 0 means another token follows at the same frame.
   */
  duration: number;
};

/**
 * TDT greedy decode: encoder states in, token ids out.
 *
 * The frame pointer advances by the predicted duration rather than by one, so
 * the loop runs far fewer steps than there are frames. A blank with duration 0
 * would stall forever, so it is forced to 1 — the reference does the same.
 */
export function decodeGreedy(W: Weights, cfg: ParakeetConfig, enc: MX): number[] {
  return decodeGreedyTimed(W, cfg, enc).map((t) => t.id);
}

/**
 * The same decode, keeping the frame each token was emitted at.
 *
 * Timestamps are nearly free here in a way they are not for an attention
 * decoder: this loop already walks encoder frames, so the pointer *is* the
 * time. Nothing extra is computed — it is only recorded.
 */
export function decodeGreedyTimed(W: Weights, cfg: ParakeetConfig, enc: MX): TimedToken[] {
  const encProj = projectEncoder(W, enc);
  const T = encProj.shape[1];
  const V = cfg.vocab_size;
  const out: TimedToken[] = [];

  let state = initialState(cfg);
  let prev = cfg.blank_token_id;
  let cached: MX | null = null;
  let t = 0,
    sinceAdvance = 0;

  while (t < T) {
    // The predictor only moves when a real token was emitted; after a blank its
    // output is unchanged, so it is reused rather than recomputed.
    if (cached === null) {
      // tidy() keeps what is returned and frees the LSTM's intermediates. Without
      // it every step of a long utterance stays resident until a GC — the pattern
      // tidy exists for, and it is why a batch of clips ran the machine out of
      // memory.
      const r = tidy(() => predict(W, cfg, prev, state));
      for (const layer of state) {
        layer.h.free();
        layer.c.free();
      }
      cached = r.out;
      state = r.state;
    }
    // Everything here is scratch: the frame slice, the reshape, the joint output.
    // `cached` was made outside this scope, so it is not adopted or freed.
    const logits = tidy(() =>
      joint(
        W,
        encProj.slice([0, t, 0], [1, t + 1, cfg.decoder_hidden_size]).reshape([1, cfg.decoder_hidden_size]),
        cached as MX,
      ).toF32(),
    );

    let token = 0,
      best = -Infinity;
    for (let i = 0; i < V; i++)
      if (logits[i] > best) {
        best = logits[i];
        token = i;
      }
    let dur = 0,
      bestD = -Infinity;
    for (let i = 0; i < cfg.durations.length; i++) {
      if (logits[V + i] > bestD) {
        bestD = logits[V + i];
        dur = cfg.durations[i];
      }
    }

    if (token === cfg.blank_token_id) {
      if (dur === 0) dur = 1; // never stall on a blank; the predictor is unchanged
    } else {
      // Recorded before `t` advances: this is the frame the token was emitted at.
      out.push({ id: token, frame: t, duration: dur });
      prev = token;
      cached.free();
      cached = null; // force a predictor step next time
    }
    t += dur;
    // A run of non-blanks at one frame must still terminate.
    sinceAdvance = dur === 0 ? sinceAdvance + 1 : 0;
    if (sinceAdvance >= cfg.max_symbols_per_step) {
      t += 1;
      sinceAdvance = 0;
    }
  }
  cached?.free();
  for (const layer of state) {
    layer.h.free();
    layer.c.free();
  }
  encProj.free();
  return out;
}
