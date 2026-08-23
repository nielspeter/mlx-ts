// Validates the mlx-ts log-Mel front-end (audio.ts) against numpy's FFT.
// reference-mel.py writes the PCM it used + its reference log-mel; here we feed
// the identical PCM through logMel() (rfft as a DFT matmul) and assert allclose.
//   python3 reference-mel.py && bun audio-test.ts
import { logMel } from "../src/audio/mel.ts";

const readF32 = async (p: string) => {
  const b = new Uint8Array(await Bun.file(p).arrayBuffer());
  return new Float32Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 4));
};

const pcm = await readF32("/tmp/mel-pcm.f32");
const ref = await readF32("/tmp/mel-ref.f32");
const { mel, F, nMels } = logMel(pcm);

let maxErr = 0, sum = 0, sumsq = 0;
for (let i = 0; i < mel.length; i++) {
  maxErr = Math.max(maxErr, Math.abs(mel[i] - (ref[i] ?? NaN)));
  sum += mel[i]; sumsq += mel[i] * mel[i];
}
const ok = mel.length === ref.length && maxErr < 5e-3;
console.log(`frames=${F} mels=${nMels} sum=${sum.toFixed(4)} sum_sq=${sumsq.toFixed(4)} maxErr=${maxErr.toExponential(2)} (ref len ${ref.length})`);
console.log(ok ? "MEL OK" : "MEL MISMATCH");
