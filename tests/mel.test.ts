// The log-Mel front-end Whisper eats.
//
// tests/audio-test.ts already checks logMel against a numpy FFT, but only on
// one fixture and only when the mel filter file is present. These cover the
// surrounding behaviour with nothing on disk: padding and trimming, the frame
// count, and that the spectrogram actually tracks the signal.
//   bun test tests/mel.test.ts
import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
// HOP/N_FFT/N_MELS/N_SAMPLES are deliberately not in the public API — they are
// implementation detail of the front-end — so they come from the module.
import {
  decodeAudio,
  HOP,
  loadMelFilters,
  logMel,
  melSpectrogram,
  N_FFT,
  N_MELS,
  N_SAMPLES,
  padOrTrim,
  SR,
} from "../src/audio/mel.ts";
import { saveAudio } from "../src/audio/wav.ts";

const dir = mkdtempSync(join(tmpdir(), "mlx-ts-mel-"));

/** A tone at `hz`, which gives the spectrogram something to find. */
const tone = (n: number, hz: number) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * hz * i) / SR));

test("padOrTrim pads a short clip with silence", () => {
  const out = padOrTrim(tone(100, 440), 256);
  expect(out.length).toBe(256);
  expect(out[255]).toBe(0);
  expect(out[50]).not.toBe(0);
});

test("padOrTrim truncates a long clip", () => {
  const src = tone(500, 440);
  const out = padOrTrim(src, 256);
  expect(out.length).toBe(256);
  expect(out[10]).toBeCloseTo(src[10], 6);
});

test("padOrTrim leaves an exact-length clip alone", () => {
  expect(padOrTrim(tone(256, 440), 256).length).toBe(256);
});

test("the default length is Whisper's 30-second window", () => {
  expect(padOrTrim(tone(10, 440)).length).toBe(N_SAMPLES);
  expect(N_SAMPLES).toBe(30 * SR);
});

test("frames advance by the hop length", () => {
  const n = HOP * 20 + N_FFT;
  const { F, nMels } = logMel(tone(n, 440));
  expect(nMels).toBe(N_MELS);
  expect(F).toBeGreaterThan(15);
  expect(F).toBeLessThanOrEqual(Math.floor(n / HOP) + 1);
});

test("dropLast removes exactly one frame", () => {
  const pcm = tone(HOP * 20 + N_FFT, 440);
  const all = logMel(pcm).F;
  expect(logMel(pcm, { dropLast: true }).F).toBe(all - 1);
});

test("the spectrogram has one value per mel band per frame", () => {
  const { mel, F, nMels } = logMel(tone(HOP * 10 + N_FFT, 440));
  expect(mel.length).toBe(F * nMels);
});

test("a tone and silence do not produce the same spectrogram", () => {
  const n = HOP * 10 + N_FFT;
  const loud = logMel(tone(n, 440)).mel;
  const quiet = logMel(new Float32Array(n)).mel;
  expect([...loud]).not.toEqual([...quiet]);
});

test("different pitches land in different bands", () => {
  const n = HOP * 10 + N_FFT;
  const low = logMel(tone(n, 220)).mel;
  const high = logMel(tone(n, 3000)).mel;
  expect([...low]).not.toEqual([...high]);
});

test("decodeAudio round-trips a file written by saveAudio", async () => {
  // Also covers the WAVE chunk walk. An earlier version of this decoder assumed
  // the classic 44-byte header, which reads metadata as audio on any file that
  // carries extra chunks — the signal shifts and it looks like the decoder
  // disagreeing rather than the parser being lazy.
  const path = join(dir, "round.wav");
  const src = tone(SR, 440);                        // one second
  await saveAudio(path, src, SR);
  const back = await decodeAudio(path);
  expect(back.length).toBe(src.length);
  // int16 quantisation is the only loss.
  let max = 0;
  for (let i = 0; i < src.length; i++) max = Math.max(max, Math.abs(src[i] - back[i]));
  expect(max).toBeLessThan(1e-4);
});

test("decodeAudio resamples to the model's rate", async () => {
  const path = join(dir, "highrate.wav");
  await saveAudio(path, tone(44100, 440), 44100);   // one second at 44.1 kHz
  const back = await decodeAudio(path);
  expect(Math.abs(back.length - SR)).toBeLessThan(SR * 0.02);
});

test("loadMelFilters reads a filterbank of the declared width", async () => {
  const bins = N_FFT / 2 + 1, mels = 4;
  const path = join(dir, "filters.f32");
  writeFileSync(path, Buffer.from(new Float32Array(bins * mels).fill(0.5).buffer));
  const f = await loadMelFilters(path, mels);
  expect(f.shape).toEqual([bins, mels]);
});

test("a short window is centred inside n_fft, the way torch.stft pads it", () => {
  // The regression this exists for: with win_length 640 inside n_fft 1024, the
  // window can be laid at offset 0 or at offset 192. Both produce something that
  // looks like a spectrogram; only the centred one matches torch.stft, which is
  // what the checkpoints were trained under. Left-aligning it moved 12 of
  // BiCodec's 32 speaker tokens.
  //
  // The two constants below are measured, not chosen:
  //   centred  0.106751  <- torchaudio.transforms.MelSpectrogram, and us
  //   aligned  0.106592  <- mlx-audio
  const N = ((6 * 16000) / 320) * 320;
  const wav = Float32Array.from({ length: N }, (_, i) => ((i * 131 + 7) % 1009) / 1009 - 0.5);

  const mel = melSpectrogram(wav);
  expect(mel.shape).toEqual([1, 301, 128]);

  const f = mel.toF32();
  const mean = f.reduce((a, b) => a + b, 0) / f.length;
  expect(mean).toBeCloseTo(0.106751, 5);
  expect(Math.abs(mean - 0.106592)).toBeGreaterThan(1e-4);   // not the left-aligned value
  mel.free();
});

test("melSpectrogram is linear magnitudes, not Whisper's normalized log", () => {
  // No log and no per-clip normalisation, so silence is 0 and doubling the
  // input doubles the output. logMel does neither.
  const quiet = melSpectrogram(new Float32Array(8000));
  expect(quiet.toF32().every((v) => v === 0)).toBe(true);
  quiet.free();

  const sig = Float32Array.from({ length: 8000 }, (_, i) => Math.sin(i / 7) * 0.3);
  const a = melSpectrogram(sig);
  const b = melSpectrogram(Float32Array.from(sig, (v) => v * 2));
  const fa = a.toF32(), fb = b.toF32();
  for (let i = 0; i < fa.length; i += 97) expect(fb[i]).toBeCloseTo(2 * fa[i], 4);
  a.free();
  b.free();
});
