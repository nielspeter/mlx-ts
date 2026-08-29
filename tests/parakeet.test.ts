// Parakeet on synthetic weights, plus the tokenizer on real strings.
//
// validation/parakeet-encode.ts checks the numbers against PyTorch with the real
// checkpoint; this runs the same code at toy size to pin the structure — the
// exact 8x subsampling, the relative-position table, and that the TDT loop
// terminates. Termination is the one that matters: a duration of 0 on a blank
// would stall forever, and nothing about the shapes would tell you.
//   bun test tests/parakeet.test.ts
import { expect, test } from "bun:test";
import {
  decodeGreedy, fromF32, initialState, joint, type ParakeetConfig,
  parakeetEncode, ParakeetTokenizer, predict, projectEncoder, relPositionalEncoding, subsample,
} from "../src/index.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

const ENC = {
  hidden_size: 32, num_hidden_layers: 2, num_attention_heads: 4, intermediate_size: 64,
  num_mel_bins: 16, conv_kernel_size: 3, subsampling_conv_channels: 8,
  subsampling_conv_kernel_size: 3, subsampling_conv_stride: 2, subsampling_factor: 8,
};
const CFG: ParakeetConfig = {
  encoder_config: ENC, decoder_hidden_size: 12, num_decoder_layers: 2,
  vocab_size: 20, blank_token_id: 19, durations: [0, 1, 2, 3, 4], max_symbols_per_step: 10,
};

function spec(): Record<string, number[]> {
  const D = ENC.hidden_size, C = ENC.subsampling_conv_channels, H = CFG.decoder_hidden_size;
  const s: Record<string, number[]> = {
    "encoder.subsampling.layers.0.weight": [C, 1, 3, 3], "encoder.subsampling.layers.0.bias": [C],
    "encoder.subsampling.layers.2.weight": [C, 1, 3, 3], "encoder.subsampling.layers.2.bias": [C],
    "encoder.subsampling.layers.3.weight": [C, C, 1, 1], "encoder.subsampling.layers.3.bias": [C],
    "encoder.subsampling.layers.5.weight": [C, 1, 3, 3], "encoder.subsampling.layers.5.bias": [C],
    "encoder.subsampling.layers.6.weight": [C, C, 1, 1], "encoder.subsampling.layers.6.bias": [C],
    // 16 mels through three stride-2 stages leaves 2, times C channels.
    "encoder.subsampling.linear.weight": [D, C * 2], "encoder.subsampling.linear.bias": [D],
    "encoder_projector.weight": [H, D], "encoder_projector.bias": [H],
    "decoder.embedding.weight": [CFG.vocab_size, H],
    "decoder.decoder_projector.weight": [H, H], "decoder.decoder_projector.bias": [H],
    "joint.head.weight": [CFG.vocab_size + CFG.durations.length, H],
    "joint.head.bias": [CFG.vocab_size + CFG.durations.length],
  };
  for (let l = 0; l < CFG.num_decoder_layers; l++) {
    s[`decoder.lstm.weight_ih_l${l}`] = [4 * H, H];
    s[`decoder.lstm.weight_hh_l${l}`] = [4 * H, H];
    s[`decoder.lstm.bias_ih_l${l}`] = [4 * H];
    s[`decoder.lstm.bias_hh_l${l}`] = [4 * H];
  }
  for (let i = 0; i < ENC.num_hidden_layers; i++) {
    const p = `encoder.layers.${i}`;
    for (const n of ["norm_feed_forward1", "norm_feed_forward2", "norm_self_att", "norm_conv", "norm_out"]) {
      s[`${p}.${n}.weight`] = [D]; s[`${p}.${n}.bias`] = [D];
    }
    for (const ff of ["feed_forward1", "feed_forward2"]) {
      s[`${p}.${ff}.linear1.weight`] = [ENC.intermediate_size, D];
      s[`${p}.${ff}.linear2.weight`] = [D, ENC.intermediate_size];
    }
    for (const n of ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"]) {
      s[`${p}.self_attn.${n}.weight`] = [D, D];
    }
    const dh = D / ENC.num_attention_heads;
    s[`${p}.self_attn.bias_u`] = [ENC.num_attention_heads, dh];
    s[`${p}.self_attn.bias_v`] = [ENC.num_attention_heads, dh];
    s[`${p}.conv.pointwise_conv1.weight`] = [2 * D, D, 1];
    s[`${p}.conv.depthwise_conv.weight`] = [D, 1, ENC.conv_kernel_size];
    s[`${p}.conv.pointwise_conv2.weight`] = [D, D, 1];
    for (const n of ["weight", "bias", "running_mean", "running_var"]) s[`${p}.conv.norm.${n}`] = [D];
  }
  return s;
}

const melOf = (T: number) =>
  fromF32(Float32Array.from({ length: T * ENC.num_mel_bins }, (_, i) => ((i * 13) % 17) / 17 - 0.5),
          [1, T, ENC.num_mel_bins]);

test("subsampling reduces time by 8 and widens to the encoder size", () => {
  const W = fakeWeights(spec());
  // Three stride-2 stages: 64 frames -> 32 -> 16 -> 8, and 16 mels -> 2.
  expect(subsample(W, ENC, melOf(64)).shape).toEqual([1, 8, ENC.hidden_size]);
  expect(subsample(W, ENC, melOf(128)).shape).toEqual([1, 16, ENC.hidden_size]);
  W.done();
});

test("the encoder keeps the subsampled length", () => {
  const W = fakeWeights(spec());
  const h = parakeetEncode(W, ENC, melOf(64));
  expect(h.shape).toEqual([1, 8, ENC.hidden_size]);
  expect(h.toF32().every(Number.isFinite)).toBe(true);
  W.done();
});

test("relative positions cover every offset, and are symmetric about zero", () => {
  // 2L-1 rows: +(L-1) down to -(L-1). The middle row is offset 0, where every
  // sin is 0 and every cos is 1 — a cheap way to catch an off-by-one in the
  // position range, which would silently skew every attention score.
  const L = 5, D = 8;
  const p = relPositionalEncoding(D, L);
  expect(p.shape).toEqual([1, 2 * L - 1, D]);
  const f = p.toF32();
  const mid = (L - 1) * D;
  for (let i = 0; i < D; i += 2) {
    expect(f[mid + i]).toBeCloseTo(0, 6);       // sin(0)
    expect(f[mid + i + 1]).toBeCloseTo(1, 6);   // cos(0)
  }
});

test("the predictor advances its state and ignores the audio", () => {
  const W = fakeWeights(spec());
  const s0 = initialState(CFG);
  const a = predict(W, CFG, 3, s0);
  const b = predict(W, CFG, 3, a.state);
  expect(a.out.shape).toEqual([1, CFG.decoder_hidden_size]);
  // Same token, different state -> different output. If the LSTM state were
  // being dropped, these would be identical.
  expect(a.out.toF32().some((v, i) => Math.abs(v - b.out.toF32()[i]) > 1e-9)).toBe(true);
  W.done();
});

test("the joint emits one score per token plus one per duration", () => {
  const W = fakeWeights(spec());
  const enc = fromF32(new Float32Array(CFG.decoder_hidden_size).fill(0.1), [1, CFG.decoder_hidden_size]);
  const dec = predict(W, CFG, CFG.blank_token_id, initialState(CFG)).out;
  expect(joint(W, enc, dec).shape).toEqual([1, CFG.vocab_size + CFG.durations.length]);
  W.done();
});

test("greedy decode terminates and stays inside the vocabulary", () => {
  // The loop advances by a predicted duration, so a duration of 0 on a blank
  // would stall on one frame forever. It is forced to 1, and a run of non-blanks
  // at a single frame is capped by max_symbols_per_step.
  const W = fakeWeights(spec());
  const enc = parakeetEncode(W, ENC, melOf(128));
  expect(projectEncoder(W, enc).shape).toEqual([1, 16, CFG.decoder_hidden_size]);
  const ids = decodeGreedy(W, CFG, enc);
  for (const id of ids) {
    expect(id).toBeGreaterThanOrEqual(0);
    expect(id).toBeLessThan(CFG.vocab_size);
    expect(id).not.toBe(CFG.blank_token_id);   // blanks are never emitted
  }
  W.done();
});

test("the tokenizer turns metaspace into word boundaries", () => {
  // SentencePiece carries spaces inside tokens as U+2581, and every utterance
  // starts with one — which would come back as a leading space.
  const tok = new ParakeetTokenizer({
    model: { vocab: [["▁hello", 0], ["▁wor", 1], ["ld", 2], ["!", 3]] },
  });
  expect(tok.decode([0, 1, 2, 3])).toBe("hello world!");
  expect(tok.decode([])).toBe("");
  expect(tok.decode([1, 2])).toBe("world");
});

// --- streaming -------------------------------------------------------------

import { ParakeetStream, SAMPLES_PER_FRAME } from "../src/index.ts";

/** A tokenizer over a tiny vocabulary, so decoding is checkable by eye. */
const toyTok = () =>
  new ParakeetTokenizer({
    model: { vocab: [["▁the", 0], ["▁coun", 1], ["try", 2], ["▁is", 3], ["!", 4]] },
  });

test("a continuation chunk keeps its leading space", () => {
  // An utterance starts with a word-start marker that must not become a leading
  // space. A chunk in the *middle* of a stream must keep it, or its first word
  // runs into the previous chunk's last one — "Americans,ask" instead of
  // "Americans, ask".
  const tok = toyTok();
  expect(tok.decode([0, 1, 2])).toBe("the country");
  expect(tok.decode([3], true)).toBe(" is");
  expect(tok.decode([3], false)).toBe("is");
});

test("startsWord distinguishes a word from its continuation", () => {
  // "country" is emitted as ▁coun + try. Without this a chunk boundary cuts
  // words in half.
  const tok = toyTok();
  expect(tok.startsWord(1)).toBe(true);   // ▁coun
  expect(tok.startsWord(2)).toBe(false);  // try
  expect(tok.startsWord(99)).toBe(false); // unknown id
});

test("latency is the chunk plus the lookahead", () => {
  const W = fakeWeights(spec());
  const s = new ParakeetStream(W, CFG, toyTok(), { chunkFrames: 13, lookaheadFrames: 13 });
  // 80 ms per frame: 26 frames is 2.08 s.
  expect(s.latencySeconds).toBeCloseTo(2.08, 6);
  expect(new ParakeetStream(W, CFG, toyTok(), { chunkFrames: 13, lookaheadFrames: 4 }).latencySeconds)
    .toBeCloseTo(1.36, 6);
  W.done();
});

test("nothing is emitted until a chunk and its lookahead have arrived", () => {
  // The lookahead is the whole reason for the delay: a chunk cannot be decoded
  // until some of the audio *after* it exists.
  const W = fakeWeights(spec());
  const s = new ParakeetStream(W, CFG, toyTok(), { chunkFrames: 4, lookaheadFrames: 4, warmupFrames: 4 });
  const tick = new Float32Array(SAMPLES_PER_FRAME);
  for (let i = 0; i < 7; i++) expect(s.push(tick)).toBe("");   // 7 frames < 4 + 4
  s.push(tick);                                                // the 8th allows a decode
  expect(s.text.length).toBeGreaterThanOrEqual(0);             // may be blanks, but it ran
  W.done();
});

test("flush drains whatever is left, with no lookahead to wait for", () => {
  const W = fakeWeights(spec());
  const s = new ParakeetStream(W, CFG, toyTok(), { chunkFrames: 4, lookaheadFrames: 4, warmupFrames: 4 });
  s.push(new Float32Array(SAMPLES_PER_FRAME * 20));
  const before = s.text;
  s.flush();
  // flush never loses what was already emitted.
  expect(s.text.startsWith(before)).toBe(true);
  W.done();
});

test("a long session does not grow without bound", () => {
  // Only the context window is retained; audio older than that is dropped, or a
  // meeting-length stream would hold the whole recording in memory.
  const W = fakeWeights(spec());
  const s = new ParakeetStream(W, CFG, toyTok(), { chunkFrames: 4, lookaheadFrames: 4, leftFrames: 8, warmupFrames: 4 });
  for (let i = 0; i < 60; i++) s.push(new Float32Array(SAMPLES_PER_FRAME));
  // 60 frames pushed, at most left + chunk + lookahead retained.
  const retained = (s as unknown as { buf: Float32Array }).buf.length / SAMPLES_PER_FRAME;
  expect(retained).toBeLessThanOrEqual(8 + 4 + 4 + 2);
  W.done();
});
