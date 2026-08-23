// Audio front-end for speech models (Whisper-style log-Mel spectrogram).
//
// The real rfft of each short frame is just a matmul by the DFT basis
// (frame @ cos, frame @ sin) — so the FFT-heavy part runs on ops we already have
// and trust, no FFT binding needed for a fixed small n_fft. Framing/windowing are
// host-side; the two DFT matmuls + power + mel matmul run on MLX.
//
// Validated against numpy's FFT (reference-mel.py / audio-test.ts). NOTE: the mel
// filterbank here is HTK-mel triangles, not librosa's exact Slaney filters — fine
// as a validated front-end, but matching a real Whisper checkpoint means loading
// its shipped mel_filters when weights are added.

import { MX, fromF32, tidy } from "../src/core/mx.ts";
import { spawn } from "node:child_process";
import { readBytes } from "../src/io/fs.ts";

export const SR = 16000, N_FFT = 400, HOP = 160, N_MELS = 80, N_SAMPLES = 30 * SR; // 30 s chunk

const TWO_PI = 2 * Math.PI;
const hann = (N: number) => Float32Array.from({ length: N }, (_, n) => 0.5 * (1 - Math.cos(TWO_PI * n / N)));

// DFT basis for an N-point rfft -> (N/2+1) bins. cos[n,k], sin[n,k].
function dftBasis(N: number): { cos: MX; sin: MX } {
  const bins = N / 2 + 1;
  const cos = new Float32Array(N * bins), sin = new Float32Array(N * bins);
  for (let n = 0; n < N; n++)
    for (let k = 0; k < bins; k++) {
      const a = TWO_PI * k * n / N;
      cos[n * bins + k] = Math.cos(a);
      sin[n * bins + k] = Math.sin(a);
    }
  return { cos: fromF32(cos, [N, bins]), sin: fromF32(sin, [N, bins]) };
}

// HTK-mel triangular filterbank, returned transposed [bins, N_MELS] for power @ fb.
function melFilterT(): MX {
  const bins = N_FFT / 2 + 1;
  const hz2mel = (f: number) => 2595 * Math.log10(1 + f / 700);
  const mel2hz = (m: number) => 700 * (10 ** (m / 2595) - 1);
  const lo = hz2mel(0), hi = hz2mel(SR / 2);
  const pts = Array.from({ length: N_MELS + 2 }, (_, i) => Math.floor((N_FFT + 1) * mel2hz(lo + (hi - lo) * i / (N_MELS + 1)) / SR));
  const fbT = new Float32Array(bins * N_MELS); // [bins, mels]
  for (let m = 1; m <= N_MELS; m++) {
    const l = pts[m - 1], c = pts[m], r = pts[m + 1];
    for (let k = l; k < c; k++) if (c > l) fbT[k * N_MELS + (m - 1)] = (k - l) / (c - l);
    for (let k = c; k < r; k++) if (r > c) fbT[k * N_MELS + (m - 1)] = (r - k) / (r - c);
  }
  return fromF32(fbT, [bins, N_MELS]);
}

// Decode any audio file to 16 kHz mono PCM via ffmpeg, as s16 -> float32/32768
// (matches Whisper's load_audio so mel/transcription line up bit-for-bit).
export async function decodeAudio(path: string): Promise<Float32Array> {
  // node:child_process rather than Bun.spawn — Deno and Node implement it too.
  const p = spawn("ffmpeg", ["-nostdin", "-i", path, "-f", "s16le", "-ac", "1", "-ar", String(SR), "-"],
    { stdio: ["ignore", "pipe", "ignore"] });
  const chunks: Uint8Array[] = [];
  for await (const c of p.stdout) chunks.push(c as Uint8Array);
  const buf = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let off = 0; for (const c of chunks) { buf.set(c, off); off += c.length; }
  const i16 = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 2));
  const f = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
  return f;
}

// Pad with zeros (or trim) to `length` samples — Whisper feeds the encoder a
// fixed 30 s (N_SAMPLES) chunk.
export function padOrTrim(pcm: Float32Array, length = N_SAMPLES): Float32Array {
  if (pcm.length >= length) return pcm.subarray(0, length);
  const out = new Float32Array(length);
  out.set(pcm);
  return out;
}

// Load Whisper's shipped librosa mel filterbank ([n_mels, bins]) -> [bins, n_mels]
// for `power @ filtersT`. (The HTK filterbank above is only for the self-test.)
export async function loadMelFilters(path = "models/whisper-mel-filters-80.f32", nMels = N_MELS): Promise<MX> {
  const b = await readBytes(path);
  const data = new Float32Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 4));
  return fromF32(data, [nMels, data.length / nMels]).transpose([1, 0]);
}

// Reflect-pad, frame (hop), and apply the Hann window -> [F, N_FFT] host buffer.
function frameSignal(pcm: Float32Array): { frames: Float32Array; F: number } {
  const pad = N_FFT / 2;
  const x = new Float32Array(pcm.length + 2 * pad);
  x.set(pcm, pad);
  for (let i = 0; i < pad; i++) { x[pad - 1 - i] = pcm[i + 1]; x[pad + pcm.length + i] = pcm[pcm.length - 2 - i]; } // reflect
  const F = 1 + Math.floor((x.length - N_FFT) / HOP);
  const W = hann(N_FFT);
  const frames = new Float32Array(F * N_FFT);
  for (let f = 0; f < F; f++)
    for (let n = 0; n < N_FFT; n++) frames[f * N_FFT + n] = x[f * HOP + n] * W[n];
  return { frames, F };
}

// log-Mel spectrogram [F, nMels] (row-major), Whisper-normalized.
//   opts.filtersT: [bins, nMels] filterbank (default: HTK self-test filters).
//   opts.dropLast: drop the final STFT frame (Whisper does this -> exactly 3000
//                  frames for a 30 s chunk).
export function logMel(pcm: Float32Array, opts: { filtersT?: MX; dropLast?: boolean } = {}): { mel: Float32Array; F: number; nMels: number } {
  let { frames, F } = frameSignal(pcm);
  if (opts.dropLast) { F -= 1; frames = frames.subarray(0, F * N_FFT); }
  const { cos, sin } = dftBasis(N_FFT);
  const ownFilters = !opts.filtersT;
  const filtersT = opts.filtersT ?? melFilterT();
  const nMels = filtersT.shape[1];
  // DFT via matmul -> power -> mel; read [F, nMels] back, then log/normalize on host.
  const melPow = tidy(() => {
    const fr = fromF32(frames, [F, N_FFT]);
    const re = fr.matmul(cos), im = fr.matmul(sin);
    return re.mul(re).add(im.mul(im)).matmul(filtersT);   // |rfft|^2 @ mel  [F, nMels]
  });
  const raw = melPow.toF32(); melPow.free();
  cos.free(); sin.free(); if (ownFilters) filtersT.free();

  const mel = new Float32Array(raw.length);
  let max = -Infinity;
  for (let i = 0; i < raw.length; i++) { const v = Math.log10(Math.max(raw[i], 1e-10)); mel[i] = v; if (v > max) max = v; }
  const floor = max - 8;
  for (let i = 0; i < mel.length; i++) mel[i] = (Math.max(mel[i], floor) + 4) / 4; // Whisper normalize
  return { mel, F, nMels };
}
