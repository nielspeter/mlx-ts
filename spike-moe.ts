// SPIKE: is MoE callable over FFI, or are MoE models simply out?
// Reproduces the exact gather_qmm expert-dispatch call from spike (Python) and
// checks the sum matches -> proves the routing args (lhs/rhs indices, optional
// group_size/bits) wire correctly over Bun FFI. Python ref sum = 0.409543.
//   python3 (saves moe-test.safetensors) ; bun spike-moe.ts

import { ptr } from "./src/ffi/index.ts";
import { MX } from "./mx.ts";
import { m, stream, gatherQmm } from "./src/ffi/generated.ts";
import { loadSafetensors, get } from "./loader.ts";

const T = 6, IN = 64, GS = 64, BITS = 4;
const w = loadSafetensors("moe-test.safetensors");
const wq = new MX(get(w, "wq")), scales = new MX(get(w, "scales")), biases = new MX(get(w, "biases"));
const x = new MX(get(w, "x"));

// uint32 index arrays (dtype 3 = UINT32)
function u32(data: number[], shape: number[]) {
  const buf = Uint32Array.from(data);
  return new MX(m.mlx_array_new_data(ptr(buf), ptr(new Int32Array(shape)), shape.length, 3) as number, buf);
}
const lhs = u32([0, 1, 2, 3, 4, 5], [T]);    // token t -> x row t
const idx = u32([0, 1, 2, 3, 0, 1], [T]);    // token t -> expert idx[t]

// x[:, None, :] then per-token quantized expert matmul
const x3 = x.reshape([T, 1, IN]);
const out = new MX(gatherQmm(x3.h, wq.h, scales.h, biases.h, lhs.h, idx.h, true, GS, BITS, "affine", false));

// sum
const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_array_new() as number) ?? 0);
m.mlx_sum(ptr(s), out.h, false, stream);
const sum = new MX(Number(s[0])).itemF();

console.log("=== MoE gather_qmm spike (4 experts, quantized, FFI routing) ===");
console.log(`TS gather_qmm sum: ${sum.toFixed(6)}   (Python ref: 0.409543)`);
console.log(`match: ${Math.abs(sum - 0.409543) < 1e-3}`);
