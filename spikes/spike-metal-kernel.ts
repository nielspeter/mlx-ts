// SPIKE: can TypeScript author and dispatch a CUSTOM METAL KERNEL over mlx-c?
//
// This is the last capability gap. codegen.ts skips the kernel-builder API —
// opaque builder structs, no single-output shape to wrap — so FINDINGS listed
// it as the one place a native shim might be needed. But the symbols are all in
// the FFI table anyway, so the question is whether a hand-written wrapper (the
// way mlx.ts bootstrapped everything before codegen) is enough.
//
// The kernel is Apple's own, lifted verbatim from mlx-examples/musicgen's
// encodec.py: a fused LSTM step, which MLX has no built-in for. If this works,
// MusicGen's EnCodec decoder is reachable — and more importantly, so is any
// kernel someone wants to write.
//
//   bun spikes/spike-metal-kernel.ts
import { m, stream } from "../src/ffi/generated.ts";
import { ptr } from "../src/ffi/index.ts";
import { MX, fromF32, tidy } from "../src/core/mx.ts";

const FLOAT32 = 10;
const KEEP: unknown[] = [];
const cstr = (s: string) => { const b = new Uint8Array([...new TextEncoder().encode(s), 0]); KEEP.push(b); return ptr(b); };
const slot = () => { const s = new BigUint64Array(1); s[0] = BigInt((m.mlx_array_new() as number) ?? 0); return s; };

// mlx_vector_string of JS strings.
function vecString(items: string[]): number {
  const v = m.mlx_vector_string_new() as number;
  for (const s of items) m.mlx_vector_string_append_value(v, m.mlx_string_new_data(cstr(s)));
  return v;
}
function vecArray(items: MX[]): number {
  const v = m.mlx_vector_array_new() as number;
  for (const a of items) m.mlx_vector_array_append_value(v, a.h);
  return v;
}

/** A compiled custom Metal kernel. `apply` dispatches it. */
export function metalKernel(opts: {
  name: string; inputNames: string[]; outputNames: string[];
  source: string; header?: string; ensureRowContiguous?: boolean; atomicOutputs?: boolean;
}) {
  const k = m.mlx_fast_metal_kernel_new(
    cstr(opts.name), vecString(opts.inputNames), vecString(opts.outputNames),
    cstr(opts.source), cstr(opts.header ?? ""),
    opts.ensureRowContiguous ?? true, opts.atomicOutputs ?? false,
  ) as number;

  return {
    apply(inputs: MX[], outputs: { shape: number[]; dtype?: number }[],
          grid: [number, number, number], threadGroup: [number, number, number]): MX[] {
      const cfg = m.mlx_fast_metal_kernel_config_new() as number;
      for (const o of outputs) {
        const sh = new Int32Array(o.shape); KEEP.push(sh);
        m.mlx_fast_metal_kernel_config_add_output_arg(cfg, ptr(sh), BigInt(o.shape.length), o.dtype ?? FLOAT32);
      }
      m.mlx_fast_metal_kernel_config_set_grid(cfg, ...grid);
      m.mlx_fast_metal_kernel_config_set_thread_group(cfg, ...threadGroup);

      const outVec = slot();
      const rc = m.mlx_fast_metal_kernel_apply(ptr(outVec), k, vecArray(inputs), cfg, stream);
      if (rc !== 0) throw new Error(`mlx_fast_metal_kernel_apply failed (${rc})`);
      m.mlx_fast_metal_kernel_config_free(cfg);

      const vh = Number(outVec[0]);
      const n = Number(m.mlx_vector_array_size(vh));
      const out: MX[] = [];
      for (let i = 0; i < n; i++) { const s = slot(); m.mlx_vector_array_get(ptr(s), vh, BigInt(i)); out.push(new MX(Number(s[0]))); }
      return out;
    },
  };
}

// --- Apple's LSTM kernel, verbatim from musicgen/encodec.py ---------------
const lstm = metalKernel({
  name: "lstm",
  inputNames: ["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
  outputNames: ["hidden_state", "cell_state"],
  header: `
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
  `,
  source: `
        uint b = thread_position_in_grid.x;
        uint d = hidden_size * 4;

        uint elem = b * d + thread_position_in_grid.y;
        uint index = elem;
        uint x_index = b * num_time_steps * d + time_step * d + index;

        auto i = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto f = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto g = metal::precise::tanh(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto o = sigmoid(h_in[index] + x[x_index]);

        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * metal::precise::tanh(cell_state[elem]);
  `,
});

// Deterministic inputs so reference-metal-kernel.py can reproduce them exactly.
const B = 2, H = 4, T = 3;
const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

// mlx_array_new_int makes a genuine 0-d int32 array. Building one through
// fromI32(..., []) does not work: ptr() on a zero-length shape array is null.
const scalarI32 = (v: number) => new MX(m.mlx_array_new_int(v) as number);

const x = fromF32(det(B * T * H * 4, 1), [B, T, H * 4]);
const hIn = fromF32(det(B * H * 4, 2), [B, H * 4]);
const cell = fromF32(det(B * H, 3), [B, H]);

const [hidden, cellOut] = tidy(() =>
  lstm.apply(
    // Scalars must be 0-d int arrays, not [1]-shaped float ones: MLX turns a
    // 0-d array into a scalar kernel parameter, while any shaped array becomes
    // a device buffer — which is why `uint hidden_size` arrived as a float*.
    [x, hIn, cell, scalarI32(H), scalarI32(0), scalarI32(T)],
    [{ shape: [B, H] }, { shape: [B, H] }],
    [B, H * 4, 1], [256, 1, 1],
  ));

console.log("=== custom Metal kernel from TypeScript (Apple's LSTM, verbatim) ===");
console.log(`  hidden_state ${JSON.stringify(hidden.shape)}: ${hidden.toF32().map((v) => +v.toFixed(6)).join(", ")}`);
console.log(`  cell_state   ${JSON.stringify(cellOut.shape)}: ${cellOut.toF32().map((v) => +v.toFixed(6)).join(", ")}`);
