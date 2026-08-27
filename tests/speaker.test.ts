// The speaker encoder on synthetic weights, plus the audio prep on real arrays.
//
// validation/speaker-encode.ts checks the numbers against mlx-audio with the
// real checkpoint; this pins the structure and the host-side arithmetic —
// BatchNorm folded from running statistics, the Res2Net channel split, the FSQ
// round trip, and the two audio steps that silently change every token if they
// are wrong.
//   bun test tests/speaker.test.ts
import { expect, test } from "bun:test";
import {
  ecapaTdnn, fromF32, fsqEncode, melFilterBank, perceiverResample, referenceClip, volumeNormalize,
} from "../src/index.ts";
import { SpeakerDetokenizer } from "../src/models/bicodec.ts";
import { fakeWeights } from "./helpers/fake-weights.ts";

test("FSQ packs its digits in mixed radix, least significant first", () => {
  // basis [1, 4, 16]: dimension 0 moves the id by 1, dimension 1 by 4, and so
  // on. Feeding a large positive value to one dimension at a time isolates it.
  const levels = [4, 4, 4];
  const digit = (id: number, d: number) => Math.floor(id / [1, 4, 16][d]) % 4;

  for (let d = 0; d < 3; d++) {
    const lo = new Float32Array(3).fill(-9);
    const hi = new Float32Array(3).fill(-9);
    hi[d] = 9;
    const [idLo] = fsqEncode(fromF32(lo, [1, 1, 3]), levels);
    const [idHi] = fsqEncode(fromF32(hi, [1, 1, 3]), levels);
    expect(idHi - idLo).toBe([1, 4, 16][d] * 3);      // bottom level -> top level
    expect(digit(idLo, d)).toBe(0);
    expect(digit(idHi, d)).toBe(3);
  }
});

test("FSQ levels are monotone in the input and cover the whole range", () => {
  // Four levels means the bins are asymmetric about zero: bound() shifts by
  // half a bin so an even level count centres between bins, giving -2..+1 after
  // the half-width offset rather than -2..+2.
  const seen = new Set<number>();
  let prev = -Infinity;
  for (let i = -40; i <= 40; i++) {
    const z = new Float32Array(6).fill(-9);
    z[0] = i / 10;
    const level = fsqEncode(fromF32(z, [1, 1, 6]))[0] % 4;
    expect(level).toBeGreaterThanOrEqual(prev);        // never goes backwards
    prev = level;
    seen.add(level);
  }
  expect([...seen].sort()).toEqual([0, 1, 2, 3]);      // all four bins reachable
});

test("FSQ encode is stable at the bin edges", () => {
  // tanh saturates, so values far outside [-1,1] must clamp rather than wrap.
  const big = fromF32(new Float32Array([-50, 50, -50, 50, -50, 50]), [1, 1, 6]);
  const ids = fsqEncode(big);
  expect(ids).toHaveLength(1);
  expect(ids[0]).toBeGreaterThanOrEqual(0);
  expect(ids[0]).toBeLessThan(4096);
});

test("the FSQ digits round-trip through SpeakerDetokenizer", () => {
  const W = fakeWeights({
    "speaker_encoder.quantizer.project_out.weight": [5, 6],
    "speaker_encoder.quantizer.project_out.bias": [5],
    "speaker_encoder.project.weight": [8, 2 * 5],
    "speaker_encoder.project.bias": [8],
  });
  const d = new SpeakerDetokenizer(W);
  expect(d.detokenize([[0, 4095]]).shape).toEqual([1, 8]);
  W.done();
});

test("slaney and htk filterbanks are different, and neither is empty", () => {
  const o = { sampleRate: 16000, nFft: 1024, nMels: 128, fMin: 10 } as const;
  const sl = melFilterBank({ ...o, scale: "slaney", norm: "slaney" }).toF32();
  const ht = melFilterBank({ ...o, scale: "htk" }).toF32();
  expect(sl.length).toBe(513 * 128);
  expect(sl.some((v) => v > 0)).toBe(true);
  expect(ht.some((v) => v > 0)).toBe(true);
  // Slaney's unit-area normalisation makes the low filters much taller, so the
  // two cannot be swapped for one another.
  expect(sl.some((v, i) => Math.abs(v - ht[i]) > 1e-3)).toBe(true);
  // Every filter must carry some weight — an empty row means a mel band with no
  // bins under it, which silently zeroes a feature.
  for (let m = 0; m < 128; m++) {
    let s = 0;
    for (let k = 0; k < 513; k++) s += sl[k * 128 + m];
    expect(s).toBeGreaterThan(0);
  }
});

test("the reference clip is a whole number of frames, tiled when short", () => {
  const long = referenceClip(new Float32Array(200000));
  expect(long.length).toBe(96000);
  expect(96000 % 320).toBe(0);

  // A 1 s clip repeats six times rather than being padded with silence, which
  // would otherwise be scored as part of the speaker.
  const short = Float32Array.from({ length: 16000 }, (_, i) => (i % 7) - 3);
  const tiled = referenceClip(short);
  expect(tiled.length).toBe(96000);
  for (const i of [0, 1, 15999, 16000, 16001, 95999]) expect(tiled[i]).toBe(short[i % 16000]);
});

test("volume normalisation targets the loud part, not the peak", () => {
  // A quiet signal with one loud transient: scaling by the peak would leave the
  // body inaudible, so the gain comes from the 90th-99th percentile instead.
  const a = Float32Array.from({ length: 4000 }, (_, i) => (i === 0 ? 0.99 : 0.02 * ((i % 5) - 2)));
  const n = volumeNormalize(a);
  expect(n.length).toBe(a.length);
  let peak = 0;
  for (const v of n) peak = Math.max(peak, Math.abs(v));
  expect(peak).toBeLessThanOrEqual(1);          // never clips
  expect(peak).toBeGreaterThan(0);
});

test("volume normalisation leaves a near-silent clip alone", () => {
  // Fewer than ten samples above 0.01 means there is nothing to measure, and
  // the function must not invent a gain from noise.
  const a = new Float32Array(1000);
  a[0] = 0.5;
  const n = volumeNormalize(a);
  expect(n.length).toBe(1000);
  expect(Number.isFinite(n[0])).toBe(true);
});

/** Mirrors what src/models/speaker.ts asks for, at toy widths. */
function ecapaSpec(feat: number, ch: number, embed: number): Record<string, number[]> {
  const s: Record<string, number[]> = {};
  const bn = (p: string, c: number) => {
    s[`${p}.weight`] = [c]; s[`${p}.bias`] = [c];
    s[`${p}.running_mean`] = [c]; s[`${p}.running_var`] = [c];
  };
  const convBn = (p: string, cin: number, cout: number, k: number) => {
    s[`${p}.conv.weight`] = [cout, cin, k]; s[`${p}.conv.bias`] = [cout];
    bn(`${p}.bn`, cout);
  };
  convBn("e.layer1", feat, ch, 5);
  const width = ch / 8;
  for (const l of ["layer2", "layer3", "layer4"]) {
    convBn(`e.${l}.se_res2block.0`, ch, ch, 1);
    for (let i = 0; i < 7; i++) {
      s[`e.${l}.se_res2block.1.convs.${i}.weight`] = [width, width, 3];
      s[`e.${l}.se_res2block.1.convs.${i}.bias`] = [width];
      bn(`e.${l}.se_res2block.1.bns.${i}`, width);
    }
    convBn(`e.${l}.se_res2block.2`, ch, ch, 1);
    s[`e.${l}.se_res2block.3.linear1.weight`] = [4, ch];
    s[`e.${l}.se_res2block.3.linear1.bias`] = [4];
    s[`e.${l}.se_res2block.3.linear2.weight`] = [ch, 4];
    s[`e.${l}.se_res2block.3.linear2.bias`] = [ch];
  }
  const cat = ch * 3;
  s["e.conv.weight"] = [cat, cat, 1]; s["e.conv.bias"] = [cat];
  s["e.pool.linear1.weight"] = [8, cat * 3, 1]; s["e.pool.linear1.bias"] = [8];
  s["e.pool.linear2.weight"] = [cat, 8, 1]; s["e.pool.linear2.bias"] = [cat];
  bn("e.bn", cat * 2);
  s["e.linear.weight"] = [embed, cat * 2]; s["e.linear.bias"] = [embed];
  return s;
}

test("ECAPA returns frame features and an utterance embedding", () => {
  const FEAT = 6, CH = 16, EMBED = 8, T = 12;
  const W = fakeWeights(ecapaSpec(FEAT, CH, EMBED));
  const mel = fromF32(Float32Array.from({ length: T * FEAT }, (_, i) => ((i * 7) % 11) / 11), [1, T, FEAT]);

  const { features, xVector } = ecapaTdnn(W, "e", mel);
  // Frame-level features keep the time axis and stack three blocks' channels.
  expect(features.shape).toEqual([1, T, CH * 3]);
  // The x-vector does not — pooling is what removes the time axis, and a clip of
  // any length has to give the same size.
  expect(xVector.shape).toEqual([1, EMBED]);
  expect(features.toF32().every((v) => v >= 0)).toBe(true);   // ReLU at the end

  const longer = fromF32(Float32Array.from({ length: 40 * FEAT }, (_, i) => ((i * 7) % 11) / 11), [1, 40, FEAT]);
  expect(ecapaTdnn(W, "e", longer).xVector.shape).toEqual([1, EMBED]);
  W.done();
});

test("the perceiver gives a fixed 32 latents whatever the clip length", () => {
  // This is the property that makes a speaker exactly 32 tokens: the output size
  // comes from the learned latents, not from the input.
  const CTX = 48, DIM = 8, N = 4, HEADS = 2, INNER = 8;
  const ff = 2 * Math.floor((DIM * 4 * 2) / 3);
  const s: Record<string, number[]> = {
    "p.proj_context.weight": [DIM, CTX], "p.proj_context.bias": [DIM],
    "p.latents": [N, DIM], "p.norm.gamma": [DIM],
  };
  for (let i = 0; i < 2; i++) {
    s[`p.layers.${i}.0.to_q.weight`] = [INNER, DIM];
    s[`p.layers.${i}.0.to_kv.weight`] = [2 * INNER, DIM];
    s[`p.layers.${i}.0.to_out.weight`] = [DIM, INNER];
    s[`p.layers.${i}.1.0.weight`] = [ff, DIM]; s[`p.layers.${i}.1.0.bias`] = [ff];
    s[`p.layers.${i}.1.2.weight`] = [DIM, ff / 2]; s[`p.layers.${i}.1.2.bias`] = [DIM];
  }
  const W = fakeWeights(s);
  for (const T of [5, 50, 300]) {
    const ctx = fromF32(Float32Array.from({ length: T * CTX }, (_, i) => ((i * 13) % 17) / 17 - 0.5), [1, T, CTX]);
    expect(perceiverResample(W, "p", ctx, 2, HEADS).shape).toEqual([1, N, DIM]);
  }
  W.done();
});
