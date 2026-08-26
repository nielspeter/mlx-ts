// Custom Metal kernels, authored in TypeScript.
//
// codegen.ts cannot wrap this API — it is builder-shaped, with opaque structs
// and no single-output signature — but every symbol it needs is in the FFI
// table, so the wrapper is ordinary hand-written code, the way src/ffi/ was
// bootstrapped before codegen existed.
//
// This is what makes the stack open-ended: if MLX has no fused op for what you
// need, you write the kernel. Apple's own EnCodec does exactly that for LSTM.
//
//   const k = metalKernel({ name, inputNames, outputNames, source, header });
//   const [out] = k.apply([x], [{ shape: [B, H] }], [B, H, 1], [256, 1, 1]);
//
// Two things that are easy to get wrong:
//   - A scalar argument must be a 0-d array (`scalarI32`). Anything shaped
//     binds as a device buffer, and the Metal compiler rejects the shader with
//     "incompatible pointer to integer conversion".
//   - Grid and threadgroup follow Metal's own semantics; MLX does not infer
//     them.

import { MX } from "../core/mx.ts";
import { m, stream } from "./generated.ts";
import { ptr } from "./index.ts";

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

/** A 0-d int array — the shape a scalar kernel argument must have. */
export const scalarI32 = (v: number): MX => new MX(m.mlx_array_new_int(v) as number);
