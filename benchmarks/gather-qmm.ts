// TS port of benchmarks/python/gather_qmm_bench.py — the quantized expert-dispatch
// kernel (gather_qmm) that drives MoE, plus the equivalent dense quantized_matmul
// for comparison. The argsort/scatter "sorted" path is omitted (unsorted is the
// representative kernel; mirrors nn.MoE's dispatch).
//   bun benchmarks/gather-qmm.ts

import { fromI32 } from "../mx.ts";
import { randomNormal, quantize, astype, UINT32, timeFn } from "./time-utils.ts";

const N = 1024, D = 1024, M = 1024, E = 32, I = 4, GS = 64, BITS = 4;

// two-layer expert MLP: x[N,1,1,D] -> gather_qmm(w1[E,M,D]) -> gather_qmm(w2[E,D,M])
{
  const x = randomNormal([N, 1, 1, D]); x.eval();
  const w1 = quantize(randomNormal([E, M, D]), GS, BITS);
  const w2 = quantize(randomNormal([E, D, M]), GS, BITS);
  const inds = astype(fromI32(Int32Array.from({ length: N * I }, () => Math.floor(Math.random() * E)), [N, I]), UINT32);
  inds.eval();
  timeFn(() => {
    const h = x.gatherQmm(w1.wq, w1.scales, w1.biases, inds, GS, BITS); // [N,I,1,M]
    return h.gatherQmm(w2.wq, w2.scales, w2.biases, inds, GS, BITS);    // [N,I,1,D]
  }, "gather_qmm (MoE dispatch, E=32 top-4)");
}

// equivalent dense quantized matmul (no routing): x[N*I,D] -> qmm(w1) -> qmm(w2)
{
  const x = randomNormal([N * I, D]); x.eval();
  const w1 = quantize(randomNormal([M, D]), GS, BITS);
  const w2 = quantize(randomNormal([D, M]), GS, BITS);
  timeFn(() => {
    const h = x.qmm(w1.wq, w1.scales, w1.biases, GS, BITS);
    return h.qmm(w2.wq, w2.scales, w2.biases, GS, BITS);
  }, "quantized_matmul (dense equivalent)");
}
