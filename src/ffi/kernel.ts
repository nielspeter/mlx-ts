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
// No module-global retention array: an FFI argument only has to outlive the
// synchronous call it is passed to, and mlx-c copies what it keeps. Holding
// every shape buffer and C string forever grew the JS heap per dispatch.
const cstr = (s: string) => ptr(new Uint8Array([...new TextEncoder().encode(s), 0]));
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

  let freed = false;
  return {
    apply(inputs: MX[], outputs: { shape: number[]; dtype?: number }[],
          grid: [number, number, number], threadGroup: [number, number, number]): MX[] {
      // Three native objects per dispatch, all of which used to leak: the config
      // (on the throw path), the input vector, and the vector the outputs come
      // back in. That last one is the expensive one — it holds a reference to
      // every output array, so freeing the returned MX released nothing. A
      // 1024-float kernel dispatched 1000 times grew active memory by 4.096 MB.
      const cfg = m.mlx_fast_metal_kernel_config_new() as number;
      let inVec = 0, outVecH = 0;
      try {
        for (const o of outputs) {
          const sh = new Int32Array(o.shape);
          m.mlx_fast_metal_kernel_config_add_output_arg(cfg, ptr(sh), BigInt(o.shape.length), o.dtype ?? FLOAT32);
        }
        m.mlx_fast_metal_kernel_config_set_grid(cfg, ...grid);
        m.mlx_fast_metal_kernel_config_set_thread_group(cfg, ...threadGroup);

        const outVec = slot();
        inVec = vecArray(inputs);
        const rc = m.mlx_fast_metal_kernel_apply(ptr(outVec), k, inVec, cfg, stream);
        outVecH = Number(outVec[0]);
        if (rc !== 0) throw new Error(`mlx_fast_metal_kernel_apply failed (${rc})`);

        // Each get() takes its own reference, so the vector itself can go.
        const n = Number(m.mlx_vector_array_size(outVecH));
        const out: MX[] = [];
        for (let i = 0; i < n; i++) { const s = slot(); m.mlx_vector_array_get(ptr(s), outVecH, BigInt(i)); out.push(new MX(Number(s[0]))); }
        return out;
      } finally {
        m.mlx_fast_metal_kernel_config_free(cfg);
        if (inVec) m.mlx_vector_array_free(inVec);
        if (outVecH) m.mlx_vector_array_free(outVecH);
      }
    },

    /** Release the compiled kernel. `using k = metalKernel(...)` does it for you. */
    free() { if (!freed) { freed = true; m.mlx_fast_metal_kernel_free(k); } },
    [Symbol.dispose]() { this.free(); },
  };
}

/** A 0-d int array — the shape a scalar kernel argument must have. */
export const scalarI32 = (v: number): MX => new MX(m.mlx_array_new_int(v) as number);
