// MusicGen end to end: text in, audio out — and a memory ceiling.
//
// The memory assertion is the point, and its first version was wrong in a way
// worth recording. It watched active memory only, saw a flat 0.39 MB/step, and
// passed — while the process was actually holding 28 GB and pushing the machine
// into swap. Active memory was never where the growth was: MLX parks freed
// Metal buffers in a reuse pool whose default ceiling is the machine's RAM, and
// a generation loop fills it. So this measures active + cache, which is what
// the OS actually has to find, and RSS as a cross-check.
//
// The two failures are separate and need separate assertions:
//   a leak      grows ACTIVE memory without bound (the escaped KV cache did),
//   a footprint is active + cache, which the OS has to find all at once.
// Cache growth on its own is fine — it rises until it hits its cap and stops —
// so asserting per-step growth on the total reports a failure every run.
//
// Leak growth is measured BETWEEN two mid-generation marks, never from before
// the model loads: the first steps materialise GBs of lazily-mmapped weights,
// which is not a leak and would swamp the signal.
//   bun validation/musicgen-e2e.ts
import { MusicGen } from "../src/index.ts";
import { activeMemoryMB, cacheMemoryMB } from "../src/core/mx.ts";

const total = () => activeMemoryMB() + cacheMemoryMB();
const rssMB = () => process.memoryUsage.rss() / 1e6;

const model = await MusicGen.fromPretrained();

let atWarm = 0, atEnd = 0, peak = 0;
const WARM = 10, TOTAL = 60;
const audio = model.generate("trance", {
  maxSteps: TOTAL,
  onStep: (i) => {
    peak = Math.max(peak, total());
    if (i === WARM) atWarm = activeMemoryMB();
    if (i === TOTAL) atEnd = activeMemoryMB();
  },
});

const s = audio.toF32();
const rms = Math.sqrt(s.reduce((t, v) => t + v * v, 0) / s.length);
const perStep = (atEnd - atWarm) / (TOTAL - WARM);

// -small is ~2.2 GB of weights; with the cache capped at 512 MB the whole run
// fits under 4 GB. Uncapped it passed 16 GB at 300 steps, so this is the line
// that would have caught it.
const CEILING = 4000;
console.log(`  samples : ${s.length}`);
console.log(`  rms     : ${rms.toFixed(4)}`);
console.log(`  active  : ${activeMemoryMB().toFixed(0)} MB   cache: ${cacheMemoryMB().toFixed(0)} MB   rss: ${rssMB().toFixed(0)} MB`);
console.log(`  leak    : active ${atWarm.toFixed(0)} -> ${atEnd.toFixed(0)} MB over ${TOTAL - WARM} steps ` +
            `(${perStep.toFixed(2)} MB/step, under 1.0 required)`);
console.log(`  peak    : ${peak.toFixed(0)} MB of ${CEILING} allowed (active + cache)`);
const ok = s.length > 0 && rms > 0.001 && perStep < 1.0 && peak < CEILING;
console.log(`  verdict : ${ok ? "OK" : "FAIL"}`);
