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

import { spawn } from "node:child_process";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fromF32, MX, tidy } from "../core/mx.ts";
import { readBytes } from "../io/fs.ts";

export const SR = 16000,
  N_FFT = 400,
  HOP = 160,
  N_MELS = 80,
  N_SAMPLES = 30 * SR; // 30 s chunk

const TWO_PI = 2 * Math.PI;
// Periodic (denominator N) is what spectral analysis wants and what
// torch.hann_window(N) gives by default. Symmetric (denominator N-1) is
// torch.hann_window(N, periodic=False), which NeMo's front end uses — the two
// differ by one sample of phase and are not interchangeable.
const hann = (N: number, periodic = true) =>
  Float32Array.from({ length: N }, (_, n) => 0.5 * (1 - Math.cos((TWO_PI * n) / (periodic ? N : N - 1))));

// DFT basis for an N-point rfft -> (N/2+1) bins. cos[n,k], sin[n,k].
function dftBasis(N: number): { cos: MX; sin: MX } {
  const bins = N / 2 + 1;
  const cos = new Float32Array(N * bins),
    sin = new Float32Array(N * bins);
  for (let n = 0; n < N; n++)
    for (let k = 0; k < bins; k++) {
      const a = (TWO_PI * k * n) / N;
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
  const lo = hz2mel(0),
    hi = hz2mel(SR / 2);
  const pts = Array.from({ length: N_MELS + 2 }, (_, i) =>
    Math.floor(((N_FFT + 1) * mel2hz(lo + ((hi - lo) * i) / (N_MELS + 1))) / SR),
  );
  const fbT = new Float32Array(bins * N_MELS); // [bins, mels]
  for (let m = 1; m <= N_MELS; m++) {
    const l = pts[m - 1],
      c = pts[m],
      r = pts[m + 1];
    for (let k = l; k < c; k++) if (c > l) fbT[k * N_MELS + (m - 1)] = (k - l) / (c - l);
    for (let k = c; k < r; k++) if (r > c) fbT[k * N_MELS + (m - 1)] = (r - k) / (r - c);
  }
  return fromF32(fbT, [bins, N_MELS]);
}

/**
 * Triangular mel filterbank, returned [bins, nMels] — the layout `magnitudes @ fb`
 * wants.
 *
 * Two conventions, and they are not interchangeable. HTK is one log curve
 * throughout; Slaney (librosa's default, and BiCodec's) is linear below 1 kHz
 * and logarithmic above, and additionally normalises each triangle to unit area
 * when `norm` is "slaney". Using the wrong one shifts every filter and quietly
 * changes the speaker embedding.
 */
export function melFilterBank(o: {
  sampleRate: number;
  nFft: number;
  nMels: number;
  fMin?: number;
  fMax?: number;
  scale?: "htk" | "slaney";
  norm?: "slaney" | null;
}): MX {
  const { sampleRate, nFft, nMels, fMin = 0, scale = "htk", norm = null } = o;
  const fMax = o.fMax ?? sampleRate / 2;
  const bins = nFft / 2 + 1;

  const MIN_LOG_HZ = 1000,
    F_SP = 200 / 3,
    LOGSTEP = Math.log(6.4) / 27;
  const minLogMel = MIN_LOG_HZ / F_SP;
  const hz2mel = (f: number) =>
    scale === "htk"
      ? 2595 * Math.log10(1 + f / 700)
      : f >= MIN_LOG_HZ
        ? minLogMel + Math.log(f / MIN_LOG_HZ) / LOGSTEP
        : f / F_SP;
  const mel2hz = (m: number) =>
    scale === "htk"
      ? 700 * (10 ** (m / 2595) - 1)
      : m >= minLogMel
        ? MIN_LOG_HZ * Math.exp(LOGSTEP * (m - minLogMel))
        : F_SP * m;

  // Bin centre frequencies, and nMels+2 mel-spaced edges around them.
  const nyquist = Math.floor(sampleRate / 2);
  const freqs = Array.from({ length: bins }, (_, i) => (nyquist * i) / (bins - 1));
  const mLo = hz2mel(fMin),
    mHi = hz2mel(fMax);
  const pts = Array.from({ length: nMels + 2 }, (_, i) => mel2hz(mLo + ((mHi - mLo) * i) / (nMels + 1)));

  const fbT = new Float32Array(bins * nMels);
  for (let m = 0; m < nMels; m++) {
    const l = pts[m],
      c = pts[m + 1],
      r = pts[m + 2];
    const gain = norm === "slaney" ? 2 / (r - l) : 1;
    for (let k = 0; k < bins; k++) {
      const up = (freqs[k] - l) / (c - l),
        down = (r - freqs[k]) / (r - c);
      const v = Math.min(up, down);
      if (v > 0) fbT[k * nMels + m] = v * gain;
    }
  }
  return fromF32(fbT, [bins, nMels]);
}

/**
 * Short-time Fourier magnitudes `[F, nFft/2+1]`, centred with reflect padding.
 *
 * `winLength` may be shorter than `nFft`, in which case the window is padded
 * with zeros on the *right* — left-aligned, not centred. Getting that backwards
 * shifts every frame by half the difference.
 */
type StftOptions = {
  /** "reflect" mirrors the signal at the edges; "constant" pads with zeros. */
  padMode?: "reflect" | "constant";
  /** torch.hann_window's `periodic` flag. */
  periodicWindow?: boolean;
  /** |X|^2 rather than |X|. Mel front ends differ on this and it is not a scale factor. */
  power?: boolean;
};

function stftMagnitude(
  pcm: Float32Array,
  nFft: number,
  hop: number,
  winLength: number,
  o: StftOptions = {},
): { mag: MX; F: number } {
  const { padMode = "reflect", periodicWindow = true, power = false } = o;
  const pad = nFft / 2;
  const x = new Float32Array(pcm.length + 2 * pad);
  x.set(pcm, pad);
  if (padMode === "reflect") {
    for (let i = 0; i < pad; i++) {
      x[pad - 1 - i] = pcm[i + 1];
      x[pad + pcm.length + i] = pcm[pcm.length - 2 - i];
    }
  } // "constant" leaves the zeros already there

  const w = hann(winLength, periodicWindow);
  const off = (nFft - winLength) >> 1; // centred, as torch.stft pads it
  const F = 1 + Math.floor((x.length - nFft) / hop);
  const frames = new Float32Array(F * nFft); // zeros either side of the window
  for (let f = 0; f < F; f++)
    for (let n = 0; n < winLength; n++) frames[f * nFft + off + n] = x[f * hop + off + n] * w[n];

  const { cos, sin } = dftBasis(nFft);
  const mag = tidy(() => {
    const fr = fromF32(frames, [F, nFft]);
    const re = fr.matmul(cos),
      im = fr.matmul(sin);
    const sq = re.mul(re).add(im.mul(im)); // |rfft|^2
    return power ? sq : sq.sqrt();
  });
  cos.free();
  sin.free();
  return { mag, F };
}

/**
 * Linear mel magnitudes `[1, F, nMels]` — no log, no normalisation.
 *
 * This is BiCodec's front end, and it differs from Whisper's in every parameter
 * that matters: magnitudes rather than power, Slaney filters rather than HTK,
 * and no log10 at the end.
 */
export function melSpectrogram(
  pcm: Float32Array,
  o: {
    sampleRate?: number;
    nFft?: number;
    hop?: number;
    winLength?: number;
    nMels?: number;
    fMin?: number;
    fMax?: number;
  } = {},
): MX {
  const { sampleRate = SR, nFft = 1024, hop = 320, winLength = 640, nMels = 128, fMin = 10 } = o;
  const { mag, F } = stftMagnitude(pcm, nFft, hop, winLength);
  const fb = melFilterBank({ sampleRate, nFft, nMels, fMin, fMax: o.fMax, scale: "slaney", norm: "slaney" });
  const mel = tidy(() => mag.matmul(fb).reshape([1, F, nMels]));
  mag.free();
  fb.free();
  return mel;
}

/**
 * NeMo's log-mel front end, as Parakeet expects it.
 *
 * Every parameter here differs from the two front ends above, and each
 * difference is silent if you get it wrong — a plausible spectrogram that moves
 * the transcript. Against Whisper's and BiCodec's:
 *
 *   preemphasis      x[i] - 0.97*x[i-1] first; neither of the others has it
 *   window           Hann *symmetric* (periodic=False), not periodic
 *   padding          zeros, not reflect
 *   spectrum         power |X|^2, not magnitude
 *   log              natural log with a 2^-24 floor, not log10
 *   normalisation    per mel bin over time, with the sample (n-1) variance
 *
 * Returned `[1, frames, 128]`, channels-last, which is the layout the encoder's
 * subsampling stack takes.
 */
export function parakeetMel(pcm: Float32Array, o: { nMels?: number; preemphasis?: number } = {}): MX {
  const { nMels = 128, preemphasis = 0.97 } = o;
  const N_FFT = 512,
    HOP = 160,
    WIN = 400;

  // Preemphasis: a one-tap high-pass that flattens the spectral tilt of speech.
  // The first sample has no predecessor and is kept as-is.
  const pre = new Float32Array(pcm.length);
  if (pcm.length) pre[0] = pcm[0];
  for (let i = 1; i < pcm.length; i++) pre[i] = pcm[i] - preemphasis * pcm[i - 1];

  const { mag, F } = stftMagnitude(pre, N_FFT, HOP, WIN, {
    padMode: "constant",
    periodicWindow: false,
    power: true,
  });
  // librosa.filters.mel(..., fmin=0, fmax=sr/2, norm="slaney") — and librosa's
  // default mel_scale is slaney too, so both are.
  const fb = melFilterBank({ sampleRate: SR, nFft: N_FFT, nMels, fMin: 0, scale: "slaney", norm: "slaney" });
  const raw = tidy(() => mag.matmul(fb));
  mag.free();
  fb.free();

  // log(x + 2^-24), then zero-mean unit-variance per mel bin across time. The
  // variance divides by (n - 1), not n: the sample variance, not the population
  // one, which is a real difference at short utterances.
  const f = raw.toF32();
  raw.free();
  const LOG_GUARD = 2 ** -24,
    EPS = 1e-5;
  const out = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = Math.log(f[i] + LOG_GUARD);

  // Centred framing yields one more frame than the model counts as real:
  // floor(samples / hop) are valid and the trailing one is masked to zero. The
  // statistics are taken over the valid frames only — including the extra one
  // shifts every value, which is 3% of absmean here.
  const valid = Math.min(F, Math.floor(pcm.length / HOP));
  for (let m = 0; m < nMels; m++) {
    let sum = 0;
    for (let t = 0; t < valid; t++) sum += out[t * nMels + m];
    const mean = sum / valid;
    let sq = 0;
    for (let t = 0; t < valid; t++) {
      const d = out[t * nMels + m] - mean;
      sq += d * d;
    }
    const std = Math.sqrt(sq / Math.max(1, valid - 1));
    for (let t = 0; t < valid; t++) out[t * nMels + m] = (out[t * nMels + m] - mean) / (std + EPS);
    for (let t = valid; t < F; t++) out[t * nMels + m] = 0;
  }
  return fromF32(out, [1, F, nMels]);
}

// Decode any audio file to 16 kHz mono PCM via ffmpeg, as s16 -> float32/32768
// (matches Whisper's load_audio so mel/transcription line up bit-for-bit).
/** Run a command to completion, collecting stdout. */
function run(cmd: string, args: string[]): Promise<Uint8Array> {
  // node:child_process rather than Bun.spawn — Deno and Node implement it too.
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: ["ignore", "pipe", "ignore"] });
    const chunks: Uint8Array[] = [];
    p.stdout.on("data", (c: Uint8Array) => chunks.push(c));
    p.on("error", () => reject(new Error(`${cmd} not found on PATH`)));
    p.on("close", (code) => {
      if (code !== 0) return reject(new Error(`${cmd} exited ${code}`));
      const buf = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
      let off = 0;
      for (const c of chunks) {
        buf.set(c, off);
        off += c.length;
      }
      resolve(buf);
    });
  });
}

/**
 * The samples of a RIFF/WAVE file, found by walking its chunks.
 *
 * Not by skipping 44 bytes: afconvert writes extra chunks, and assuming the
 * classic header silently reads metadata as audio — which shifts the signal and
 * looks like a decoder disagreeing when it is only a parser being lazy.
 */
function wavSamples(buf: Uint8Array): Int16Array {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let p = 12; // past "RIFF" size "WAVE"
  while (p + 8 <= buf.length) {
    const id = String.fromCharCode(buf[p], buf[p + 1], buf[p + 2], buf[p + 3]);
    const size = v.getUint32(p + 4, true);
    if (id === "data") {
      const start = buf.byteOffset + p + 8;
      return new Int16Array(buf.buffer.slice(start, start + size));
    }
    p += 8 + size + (size & 1); // chunks are word-aligned
  }
  throw new Error("decodeAudio: no data chunk in the decoded WAVE");
}

/**
 * Any audio file -> mono float PCM at `SR`.
 *
 * Uses macOS's own `afconvert`, which is built in — MLX is Apple-Silicon-only,
 * so there is nothing to gain by requiring a separate install. Set
 * `MLXTS_AUDIO_DECODER=ffmpeg` to use ffmpeg instead; the Whisper parity check
 * does, because its oracle decodes that way and a ~1% difference in the mel
 * spectrogram could flip a token.
 */
export async function decodeAudio(path: string): Promise<Float32Array> {
  let i16: Int16Array;

  if (process.env.MLXTS_AUDIO_DECODER === "ffmpeg") {
    const raw = await run("ffmpeg", [
      "-nostdin",
      "-i",
      path,
      "-f",
      "s16le",
      "-ac",
      "1",
      "-ar",
      String(SR),
      "-",
    ]);
    i16 = new Int16Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 2));
  } else {
    // afconvert writes a file rather than a stream, so it goes via a temp path.
    const out = join(tmpdir(), `mlx-ts-decode-${process.pid}-${basename(path)}.wav`);
    await run("afconvert", ["-f", "WAVE", "-d", `LEI16@${SR}`, "-c", "1", path, out]);
    i16 = wavSamples(await readBytes(out));
    await rm(out, { force: true });
  }

  const f = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
  return f;
}

/** Play an audio file through the system output. macOS only, like the rest. */
export async function playAudio(path: string): Promise<void> {
  await run("afplay", [path]);
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
export async function loadMelFilters(
  path = "models/whisper-mel-filters-80.f32",
  nMels = N_MELS,
): Promise<MX> {
  const b = await readBytes(path);
  const data = new Float32Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 4));
  return fromF32(data, [nMels, data.length / nMels]).transpose([1, 0]);
}

// Reflect-pad, frame (hop), and apply the Hann window -> [F, N_FFT] host buffer.
function frameSignal(pcm: Float32Array): { frames: Float32Array; F: number } {
  const pad = N_FFT / 2;
  const x = new Float32Array(pcm.length + 2 * pad);
  x.set(pcm, pad);
  for (let i = 0; i < pad; i++) {
    x[pad - 1 - i] = pcm[i + 1];
    x[pad + pcm.length + i] = pcm[pcm.length - 2 - i];
  } // reflect
  const F = 1 + Math.floor((x.length - N_FFT) / HOP);
  const W = hann(N_FFT);
  const frames = new Float32Array(F * N_FFT);
  for (let f = 0; f < F; f++) for (let n = 0; n < N_FFT; n++) frames[f * N_FFT + n] = x[f * HOP + n] * W[n];
  return { frames, F };
}

// log-Mel spectrogram [F, nMels] (row-major), Whisper-normalized.
//   opts.filtersT: [bins, nMels] filterbank (default: HTK self-test filters).
//   opts.dropLast: drop the final STFT frame (Whisper does this -> exactly 3000
//                  frames for a 30 s chunk).
export function logMel(
  pcm: Float32Array,
  opts: { filtersT?: MX; dropLast?: boolean } = {},
): { mel: Float32Array; F: number; nMels: number } {
  let { frames, F } = frameSignal(pcm);
  if (opts.dropLast) {
    F -= 1;
    frames = frames.subarray(0, F * N_FFT);
  }
  const { cos, sin } = dftBasis(N_FFT);
  const ownFilters = !opts.filtersT;
  const filtersT = opts.filtersT ?? melFilterT();
  const nMels = filtersT.shape[1];
  // DFT via matmul -> power -> mel; read [F, nMels] back, then log/normalize on host.
  const melPow = tidy(() => {
    const fr = fromF32(frames, [F, N_FFT]);
    const re = fr.matmul(cos),
      im = fr.matmul(sin);
    return re.mul(re).add(im.mul(im)).matmul(filtersT); // |rfft|^2 @ mel  [F, nMels]
  });
  const raw = melPow.toF32();
  melPow.free();
  cos.free();
  sin.free();
  if (ownFilters) filtersT.free();

  const mel = new Float32Array(raw.length);
  let max = -Infinity;
  for (let i = 0; i < raw.length; i++) {
    const v = Math.log10(Math.max(raw[i], 1e-10));
    mel[i] = v;
    if (v > max) max = v;
  }
  const floor = max - 8;
  for (let i = 0; i < mel.length; i++) mel[i] = (Math.max(mel[i], floor) + 4) / 4; // Whisper normalize
  return { mel, F, nMels };
}
