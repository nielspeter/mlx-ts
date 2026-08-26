// Write a mono waveform to a RIFF/WAVE file (16-bit PCM).
//
// MusicGen emits float samples in roughly [-1, 1]; 16-bit PCM is what every
// player reads without argument.
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

/** Write `samples` (float, ~[-1,1]) to `path` as 16-bit mono PCM. */
export async function saveAudio(path: string, samples: ArrayLike<number>, sampleRate: number): Promise<void> {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const ascii = (off: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

  ascii(0, "RIFF"); view.setUint32(4, 36 + n * 2, true); ascii(8, "WAVE");
  ascii(12, "fmt "); view.setUint32(16, 16, true);       // PCM chunk size
  view.setUint16(20, 1, true);                            // format = PCM
  view.setUint16(22, 1, true);                            // channels = mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);               // byte rate
  view.setUint16(32, 2, true);                            // block align
  view.setUint16(34, 16, true);                           // bits per sample
  ascii(36, "data"); view.setUint32(40, n * 2, true);

  // MusicGen overshoots ±1 now and then. Clamping there flattens the peak into
  // audible distortion, so scale the whole clip instead — same waveform, just
  // quieter, and only when something actually exceeds full scale.
  // Non-finite samples are skipped here: one infinity would otherwise make the
  // gain zero and silence the entire clip. They are clamped individually below.
  let peak = 0;
  for (let i = 0; i < n; i++) { const a = Math.abs(samples[i]); if (a > peak && a !== Infinity) peak = a; }
  const gain = peak > 1 ? 1 / peak : 1;

  for (let i = 0; i < n; i++) {
    // Still clamped, so a NaN or an infinity cannot wrap to the opposite extreme.
    const s = Math.max(-1, Math.min(1, samples[i] * gain));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  await mkdir(dirname(path), { recursive: true }).catch(() => {});
  await writeFile(path, new Uint8Array(buf));
}
