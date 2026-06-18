// TS port of benchmarks/python/batch_matmul_bench.py — forward matmul only
// (the vjp/transpose-matmul variants need autograd, out of scope here).
//   bun benchmarks/batch-matmul.ts

import { randomUniform, timeFn } from "./time-utils.ts";

const B = 8, T = 1024, D = 512;

// batched: [B,T,D] @ [D,D]
{
  const a = randomUniform([B, T, D]); const b = randomUniform([D, D]);
  a.eval(); b.eval();
  timeFn(() => a.matmul(b), "batch_matmul [B,T,D]@[D,D]");
}

// unbatched: [B*T,D] @ [D,D]
{
  const a = randomUniform([B * T, D]); const b = randomUniform([D, D]);
  a.eval(); b.eval();
  timeFn(() => a.matmul(b), "unbatch_matmul [B*T,D]@[D,D]");
}
