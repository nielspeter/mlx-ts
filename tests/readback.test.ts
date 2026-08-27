// Reading tensors back to the host — the boundary where a wrong answer looks
// right.
//
// Both hazards here are silent: mlx_array_data_float32 hands back the raw
// buffer, so a transposed array reads out in its *pre*-transpose order under
// the new shape, and on a non-float32 array it returns null and the read
// segfaults instead of throwing. Neither shows up in the shape.
//   bun test tests/readback.test.ts
import { expect, test } from "bun:test";
import { fromF32, fromI32 } from "../src/index.ts";

test("toF32 follows a transpose instead of the underlying buffer", () => {
  const a = fromF32(Float32Array.from([0, 1, 2, 3, 4, 5]), [2, 3]);
  const t = a.transpose([1, 0]);
  expect(t.shape).toEqual([3, 2]);
  expect(t.toF32()).toEqual([0, 3, 1, 4, 2, 5]);   // not [0,1,2,3,4,5]
  expect(a.toF32()).toEqual([0, 1, 2, 3, 4, 5]);   // the source is untouched
});

test("contiguity describes the materialised layout, so it needs an eval", () => {
  const a = fromF32(Float32Array.from([1, 2, 3, 4]), [2, 2]);
  expect(a.contiguous).toBe(true);

  const t = a.transpose([1, 0]);
  t.eval();                       // a pending transpose still reports its source
  expect(t.contiguous).toBe(false);

  // A size-1 axis has no meaningful stride, so it must not count as a break.
  const thin = fromF32(Float32Array.from([1, 2]), [1, 2, 1]);
  thin.eval();
  expect(thin.contiguous).toBe(true);
});

test("toF32 casts rather than reading another dtype's buffer", () => {
  // Reading an int32 array as float32 used to return null and segfault.
  const i = fromI32(Int32Array.from([1, -2, 3]), [3]);
  expect(i.toF32()).toEqual([1, -2, 3]);
});

test("a transposed non-float32 array survives both fixes at once", () => {
  const i = fromI32(Int32Array.from([0, 1, 2, 3, 4, 5]), [2, 3]);
  expect(i.transpose([1, 0]).toF32()).toEqual([0, 3, 1, 4, 2, 5]);
});
