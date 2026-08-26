// Differential tests of the mlx-ts FFI binding against MLX Python.
// Fixtures (tests/fixtures.json) carry the exact inputs MLX saw and the output it
// produced; here we feed the identical inputs through mlx-ts and assert allclose.
// Regenerate fixtures with:  python3 tests/gen-fixtures.py
//   bun test tests/
import { expect, test } from "bun:test";
import { fromF32, fromI32, fromU32, MX } from "../src/core/mx.ts";

type Tensor = { shape: number[]; dtype?: "f32" | "i32" | "u32"; data: number[] };
type Case = { name: string; op: string; inputs: Tensor[]; params: any; expected: Tensor; atol?: number; rtol?: number };

const build = (t: Tensor): MX =>
  t.dtype === "u32" ? fromU32(Uint32Array.from(t.data), t.shape)
  : t.dtype === "i32" ? fromI32(Int32Array.from(t.data), t.shape)
  : fromF32(Float32Array.from(t.data), t.shape);

const fixtures: Case[] = await Bun.file(new URL("./fixtures.json", import.meta.url)).json();

// op registry: name -> (inputs, params) -> result MX. Exercises the public MX surface.
const ops: Record<string, (i: MX[], p: any) => MX> = {
  add: ([a, b]) => a.add(b),
  mul: ([a, b]) => a.mul(b),
  matmul: ([a, b]) => a.matmul(b),
  sumAxis: ([a], p) => a.sumAxes([p.axis], false),
  softmax: ([a], p) => a.softmax(p.axis),
  argmax: ([a], p) => a.argmax(p.axis),
  reshape: ([a], p) => a.reshape(p.shape),
  transpose: ([a], p) => a.transpose(p.axes),
  rmsNorm: ([a, w], p) => a.rmsNorm(w, p.eps),
  rope: ([a], p) => a.rope(p.dims, p.base, p.offset),
  sdpa: ([q, k, v], p) => MX.sdpa(q, k, v, p.scale, p.causal),
  qmm: ([x, wq, sc, bi], p) => x.qmm(wq, sc, bi, p.gs, p.bits),
  dequantize: ([wq, sc, bi], p) => MX.dequantize(wq, sc, bi, p.gs, p.bits),
  gatherQmm: ([x, wq, sc, bi, inds], p) => x.gatherQmm(wq, sc, bi, inds, p.gs, p.bits),
};

const allclose = (got: number[], want: number[], atol = 1e-4, rtol = 1e-4) =>
  got.length === want.length &&
  got.every((g, i) => Math.abs(g - want[i]) <= atol + rtol * Math.abs(want[i]));

for (const c of fixtures) {
  test(`${c.op}: ${c.name}`, () => {
    const inputs = c.inputs.map(build);
    const fn = ops[c.op];
    expect(fn, `no mlx-ts op registered for "${c.op}"`).toBeDefined();
    const out = fn(inputs, c.params);
    expect(out.shape).toEqual(c.expected.shape);
    // Read through a contiguous copy: ops like transpose return a strided view,
    // and toF32/toU32 read the raw buffer (row-major), which would see the
    // pre-transpose layout. copy() (stack+reshape) materializes logical order.
    const mat = out.copy();
    const got = c.expected.dtype === "u32" || c.op === "argmax" ? mat.toU32() : mat.toF32();
    expect(allclose([...got], c.expected.data, c.atol, c.rtol)).toBe(true);
    out.free();
    mat.free();
    inputs.forEach((x) => x.free());
  });
}
