// SPIKE: can we do the iSTFT vocoder step in mlx-c/TS? The inverse rfft is a
// matmul by the inverse-DFT basis (Hermitian 2x weighting), then a windowed
// overlap-add. Validate against mlx-audio's istft (reference-istft.py).
//   /tmp/wvenv/bin/python reference-istft.py && bun spike-istft.ts
import { MX, fromF32 } from "./mx.ts";

const N_FFT = 400, HOP = 100, BINS = N_FFT / 2 + 1, F = 50;
const TWO_PI = 2 * Math.PI;

const readF32 = async (p: string) => {
  const b = new Uint8Array(await Bun.file(p).arrayBuffer());
  return new Float32Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 4));
};
const hann = (n: number) => 0.5 * (1 - Math.cos(TWO_PI * n / N_FFT)); // periodic, length N_FFT

// Inverse-DFT basis [BINS, N_FFT]: x[n] = (1/N) Σ_k c_k (Re·cos - Im·sin),
// c_0 = c_{N/2} = 1, else 2; imag of bins 0 and N/2 is discarded (real signal).
function inverseDftBasis(): { cinv: MX; sinv: MX } {
  const cinv = new Float32Array(BINS * N_FFT), sinv = new Float32Array(BINS * N_FFT);
  for (let k = 0; k < BINS; k++) {
    const c = (k === 0 || k === N_FFT / 2) ? 1 : 2;
    const real0 = k === 0 || k === N_FFT / 2; // these bins are real-only
    for (let n = 0; n < N_FFT; n++) {
      const a = TWO_PI * k * n / N_FFT;
      cinv[k * N_FFT + n] = (c / N_FFT) * Math.cos(a);
      sinv[k * N_FFT + n] = real0 ? 0 : -(c / N_FFT) * Math.sin(a);
    }
  }
  return { cinv: fromF32(cinv, [BINS, N_FFT]), sinv: fromF32(sinv, [BINS, N_FFT]) };
}

const re = fromF32(await readF32("/tmp/istft-re.f32"), [BINS, F]);
const im = fromF32(await readF32("/tmp/istft-im.f32"), [BINS, F]);
const ref = await readF32("/tmp/istft-ref.f32");

// irfft per frame, as matmul: framesTime[F, N] = ReXᵀ @ Cinv + ImXᵀ @ Sinv
const { cinv, sinv } = inverseDftBasis();
const framesTime = re.transpose([1, 0]).matmul(cinv).add(im.transpose([1, 0]).matmul(sinv)); // [F, N_FFT]
const ft = framesTime.copy().toF32();

// windowed overlap-add (host bookkeeping), then normalize by window sum, trim center
const W = Float32Array.from({ length: N_FFT }, (_, n) => hann(n));
const t = (F - 1) * HOP + N_FFT;
const recon = new Float32Array(t), wsum = new Float32Array(t);
for (let f = 0; f < F; f++) {
  const off = f * HOP;
  for (let n = 0; n < N_FFT; n++) { recon[off + n] += ft[f * N_FFT + n] * W[n]; wsum[off + n] += W[n]; }
}
for (let i = 0; i < t; i++) if (wsum[i] > 1e-10) recon[i] /= wsum[i];
const out = recon.subarray(N_FFT / 2, t - N_FFT / 2); // center=true trim

let maxErr = 0, maxMag = 0;
for (let i = 0; i < out.length; i++) { maxErr = Math.max(maxErr, Math.abs(out[i] - ref[i])); maxMag = Math.max(maxMag, Math.abs(ref[i])); }
console.log(`waveform len: ts=${out.length} ref=${ref.length}`);
console.log(`maxErr=${maxErr.toExponential(2)} rel=${(maxErr / maxMag).toExponential(2)}`);
console.log(out.length === ref.length && maxErr / maxMag < 1e-4 ? "ISTFT OK" : "ISTFT MISMATCH");
