// Validate whisper.ts against mlx_whisper: identical mel + tokens -> compare
// encoder audio features and decoder logits (incl. the discrete argmax token).
//   /tmp/wvenv/bin/python reference-whisper.py && bun whisper-test.ts
import { fromF32, fromI32, MX } from "../src/core/mx.ts";
import { loadWhisper } from "../src/models/whisper.ts";

const readF32 = async (p: string) => {
  const b = new Uint8Array(await Bun.file(p).arrayBuffer());
  return new Float32Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 4));
};
const readI32 = async (p: string) => {
  const b = new Uint8Array(await Bun.file(p).arrayBuffer());
  return new Int32Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 4));
};
const stats = (a: number[] | Float32Array, b: Float32Array) => {
  let maxErr = 0, maxMag = 0, sumAbs = 0;
  for (let i = 0; i < a.length; i++) { const e = Math.abs(a[i] - b[i]); maxErr = Math.max(maxErr, e); maxMag = Math.max(maxMag, Math.abs(b[i])); sumAbs += e; }
  return { maxErr, rel: maxErr / maxMag, mean: sumAbs / a.length };
};

const model = await loadWhisper();
const FP16 = 9;

// --- encoder ---
const mel = (await readF32("/tmp/whisper-mel.f32"));
const melMX = fromF32(mel, [1, 3000, 80]).astype(FP16);
const af = model.encoder(melMX);
const afData = af.copy().toF32();
const encRef = await readF32("/tmp/whisper-enc.f32");
const enc = stats(afData, encRef);

// --- decoder ---
const toks = await readI32("/tmp/whisper-tokens.i32");
const T = toks.length;
const logits = model.decoder(fromI32(toks, [1, T]), af);
const lgData = logits.copy().toF32();
const lgRef = await readF32("/tmp/whisper-logits.f32");
const lg = stats(lgData, lgRef);

// discrete check: argmax of the last position
const V = lgData.length / T;
const argmax = (arr: Float32Array | number[], off: number, n: number) => {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < n; i++) if (arr[off + i] > bv) { bv = arr[off + i]; bi = i; }
  return bi;
};
const tsTok = argmax(lgData, (T - 1) * V, V);
const refTok = argmax(lgRef, (T - 1) * V, V);

// fp16 through 4 layers leaves a few outliers (LayerNorm/GELU), so judge the
// encoder by mean error and the decoder by the discrete argmax token (the bar).
console.log(`encoder:        meanErr=${enc.mean.toExponential(2)} maxErr=${enc.maxErr.toFixed(3)}`);
console.log(`decoder logits: meanErr=${lg.mean.toExponential(2)} rel=${lg.rel.toExponential(2)}`);
console.log(`argmax last token: ts=${tsTok} ref=${refTok}`);
const ok = enc.mean < 1e-2 && lg.rel < 5e-2 && tsTok === refTok;
console.log(ok ? "WHISPER OK" : "WHISPER MISMATCH");
