// TS port of benchmarks/python/rope_bench.py — fast rope, vec + matrix, looped 32x.
//   bun benchmarks/rope.ts

import { randomUniform, astype, FLOAT16, timeFn, loop } from "./time-utils.ts";

const DIMS = 64, THETA = 10000, LOOP = 32;

// vec: [1, 32, 1, 128] at offset 100
const xv = astype(randomUniform([1, 32, 1, 128]), FLOAT16); xv.eval();
timeFn(() => loop(xv, LOOP, (y) => y.rope(DIMS, THETA, 100)), "rope_vec (32x)");

// matrix: [1, 32, 1024, 128] at offset 0
const xm = astype(randomUniform([1, 32, 1024, 128]), FLOAT16); xm.eval();
timeFn(() => loop(xm, LOOP, (y) => y.rope(DIMS, THETA, 0)), "rope_mat (32x)");
