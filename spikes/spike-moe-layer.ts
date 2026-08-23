// SPIKE: full MoE layer end-to-end (router -> top-K -> quantized expert dispatch
// -> weighted combine) via the nn.MoE module, vs the Python reference on the
// same weights. Turns the op-level gather_qmm proof into a full-layer proof.
//   (python prototype saved models/moe-layer.safetensors) ; bun spike-moe-layer.ts

import { MX } from "../src/core/mx.ts";
import { MoE, Linear } from "../src/nn/nn.ts";
import { m, stream } from "../src/ffi/generated.ts";
import { ptr } from "../src/ffi/index.ts";
import { loadSafetensors, get } from "../src/io/loader.ts";

const E = 8, K = 2, D = 64, I = 128, T = 5, GS = 64, BITS = 4;
const w = loadSafetensors("models/moe-layer.safetensors");
const g = (n: string) => new MX(get(w, n));

const moe = new MoE(
  new Linear(g("Wg")),
  { wq: g("gq"), scales: g("gs"), biases: g("gb") },
  { wq: g("uq"), scales: g("us"), biases: g("ub") },
  { wq: g("dq"), scales: g("ds"), biases: g("db") },
  K, GS, BITS,
);

const out = moe.forward(g("x"));   // [T, D]

// sum + sumsq
function reduceSum(a: MX): number {
  const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_array_new() as number) ?? 0);
  m.mlx_sum(ptr(s), a.h, false, stream); return new MX(Number(s[0])).itemF();
}
const sum = reduceSum(out);
const sumsq = reduceSum(out.mul(out));

console.log("=== full MoE layer spike (8 experts, top-2, 4-bit) — nn.MoE ===");
console.log(`TS    sum=${sum.toFixed(6)}  sumsq=${sumsq.toFixed(3)}`);
console.log(`Python ref sum=9.353439  sumsq=11773.206`);
console.log(`match: ${Math.abs(sum - 9.353439) < 2e-2 && Math.abs(sumsq - 11773.206) < 5}`);
