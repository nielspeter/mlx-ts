// The 30-second tour: arrays, ops, and the one thing you must know about memory.
// Needs no model weights — this is the example that runs on a fresh clone.
//   bun examples/basics.ts   |   deno run --allow-all examples/basics.ts   |   node examples/basics.ts
import {
  fromF32, scalar, stack, tidy, evalAll,
  activeMemoryMB, peakMemoryMB, clearCache, backend,
} from "../src/index.ts";

console.log(`runtime: ${backend.version}\n`);

// --- arrays ---------------------------------------------------------------
// fromF32(data, shape). Ops are lazy: nothing runs on the GPU until .eval()
// (or a read-back like .toF32(), which evals for you).
const a = fromF32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const b = fromF32(new Float32Array([0.5, -1, 2, 0.25, -0.75, 1.5]), [3, 2]);

const c = a.matmul(b);
console.log("a @ b      =", c.toF32(), "shape", c.shape);

// Ops chain, and scalars are arrays too.
const d = a.mul(scalar(2)).add(scalar(1)).silu();
console.log("silu(2a+1) =", d.toF32().map((x) => +x.toFixed(3)));

// Reductions, reshapes, stacking.
console.log("mean(a)    =", a.meanAll().itemF().toFixed(4));
console.log("a^T shape  =", a.transpose([1, 0]).shape);
console.log("stacked    =", stack([a, a], 0).shape);

// evalAll forces a batch of lazy graphs in one go.
evalAll(c, d);

// --- memory: why tidy() is not optional -----------------------------------
// Every MX wraps a native handle. A FinalizationRegistry frees it after a GC —
// but a tight synchronous loop never gives the GC a chance, so handles pile up.
// tidy() is the deterministic fix: it frees everything created in its scope
// except what you return.
const big = () => fromF32(new Float32Array(512 * 512).fill(0.01), [512, 512]);
const x = big(), y = big();

clearCache();
const beforeLeak = activeMemoryMB();
for (let i = 0; i < 200; i++) x.matmul(y).eval();          // no tidy: handles accumulate
const leaked = activeMemoryMB() - beforeLeak;

clearCache();
const beforeTidy = activeMemoryMB();
for (let i = 0; i < 200; i++) tidy(() => { x.matmul(y).eval(); });
const tidied = activeMemoryMB() - beforeTidy;

console.log(`\n200 x [512,512] matmul:`);
console.log(`  without tidy(): +${leaked.toFixed(1)} MB active`);
console.log(`  with    tidy(): +${tidied.toFixed(1)} MB active`);
console.log(`  peak: ${peakMemoryMB().toFixed(1)} MB`);
