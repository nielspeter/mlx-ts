// TS port of benchmarks/python/sdpa_vector_bench.py — the fused fast SDPA path,
// single-query GQA decode against a long KV cache, looped 10x (the inner loop of
// the Python bench). The hand-rolled primitive path and the masked variants are
// omitted; this is the kernel that matters for decode throughput.
//   bun benchmarks/sdpa-vector.ts

import { MX } from "../src/core/mx.ts";
import { astype, FLOAT16, loop, randomUniform, timeFn } from "./time-utils.ts";

const L = 16384, H = 32, H_k = H / 4, D = 128, V = 128, LOOP = 10;
const scale = 1.0;

const q = astype(randomUniform([1, H, 1, D]), FLOAT16);
const k = astype(randomUniform([1, H_k, L, D]), FLOAT16);
const v = astype(randomUniform([1, H_k, L, V]), FLOAT16);
q.eval(); k.eval(); v.eval();

// V === D here, so the Python up-projection is None — just loop SDPA.
timeFn(() => loop(q, LOOP, (o) => MX.sdpa(o, k, v, scale, false)), "sdpa_vector (fast, GQA decode, 10x)");
