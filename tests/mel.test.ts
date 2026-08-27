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
import { decodeAudio, HOP, loadMelFilters, logMel, N_FFT, N_MELS, N_SAMPLES, padOrTrim, SR } from "../src/audio/mel.ts";
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
