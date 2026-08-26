// MusicGen end to end: text in, audio out — and a leak check.
//
// The memory assertion is the point. The KV cache escapes each step's tidy(),
// so the step that replaces it must free the pair it replaced. When it did not,
// this leaked ~393 KB per step per layer and reached 55 GB on a 39 GB machine.
//
// Growth is measured BETWEEN two mid-generation marks, never from before the
// model loads: the first steps materialise 2 GB of lazily-mmapped weights,
// which is not a leak and would swamp the signal.
//   bun validation/musicgen-e2e.ts
import { MusicGen } from "../src/index.ts";
import { activeMemoryMB } from "../src/core/mx.ts";

const model = await MusicGen.fromPretrained();

let atWarm = 0, atEnd = 0;
const WARM = 10, TOTAL = 30;
const audio = model.generate("trance", {
  maxSteps: TOTAL,
  onStep: (i) => {
    if (i === WARM) atWarm = activeMemoryMB();
    if (i === TOTAL) atEnd = activeMemoryMB();
  },
});

const s = audio.toF32();
const rms = Math.sqrt(s.reduce((t, v) => t + v * v, 0) / s.length);
const grewPerStep = (atEnd - atWarm) / (TOTAL - WARM);

console.log(`  samples : ${s.length}`);
console.log(`  rms     : ${rms.toFixed(4)}`);
console.log(`  memory  : ${atWarm.toFixed(0)} -> ${atEnd.toFixed(0)} MB over ${TOTAL - WARM} steps ` +
            `(${grewPerStep.toFixed(2)} MB/step)`);
// The cache legitimately grows by one token per step; the leak was ~10 MB/step.
console.log(`  verdict : ${s.length > 0 && rms > 0.001 && grewPerStep < 1.0 ? "OK" : "FAIL"}`);
