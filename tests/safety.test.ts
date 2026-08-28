// The native boundary: errors, host-buffer validation, and arena ownership.
//
// Every case here is a defect a source review turned up and that reproduced
// before it was fixed. They share a failure mode — the SDK looked fine and did
// the wrong thing silently, or took the process down with it — so they are
// pinned rather than trusted.
//   bun test tests/safety.test.ts
import { expect, test } from "bun:test";
import { fromF32, fromI32, MX, tidy } from "../src/index.ts";

const arr = () => fromF32(Float32Array.from([1, 2, 3, 4]), [2, 2]);

// --- mlx-c errors are catchable, not fatal ---------------------------------

test("an invalid op raises a TypeScript Error instead of killing the process", () => {
  // mlx-c's default handler prints and calls exit(). Before the handler was
  // installed this exited with status 255 and no catch block ever ran, which
  // meant a bad shape from an HTTP request could take down a server.
  const a = fromF32(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
  const b = fromF32(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
  expect(() => a.matmul(b).eval()).toThrow(/matmul/);
});

test("the error names the operation", () => {
  const a = fromF32(Float32Array.from([1, 2]), [2]);
  const b = fromF32(Float32Array.from([1, 2, 3]), [3]);
  expect(() => a.add(b).eval()).toThrow(/mlx_/);
});

test("a failed op leaves the runtime usable", () => {
  // The error state has to be consumed, not left pending — otherwise the next
  // unrelated op inherits it and throws for no reason.
  const a = fromF32(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
  try { a.matmul(a).eval(); } catch { /* expected */ }
  expect(arr().add(arr()).toF32()).toEqual([2, 4, 6, 8]);
});

// --- host-array constructors ----------------------------------------------

test("a shape larger than the buffer is rejected, not read past", () => {
  // mlx-c takes a pointer and a shape but no byte length. This used to return
  // [42, 0, 0, 0] — three values read beyond a one-element TypedArray.
  expect(() => fromF32(Float32Array.from([42]), [4])).toThrow(/needs 4 elements, got 1/);
});

test("non-integer and negative dimensions are rejected", () => {
  // [2.5] was silently truncated to [2]; [-1, 2] reached the allocator as a
  // huge unsigned size and killed the process.
  expect(() => fromF32(Float32Array.from([1, 2, 3, 4]), [2.5])).toThrow(/non-integer or negative/);
  expect(() => fromF32(Float32Array.from([1, 2, 3, 4]), [-1, 2])).toThrow(/non-integer or negative/);
});

test("a shape whose product overflows is rejected", () => {
  expect(() => fromF32(Float32Array.from([1]), [1e12, 1e12])).toThrow(/overflows/);
});

test("a 0-d array is constructible", () => {
  // product([]) is 1, so one element is exactly right. This used to fail inside
  // the FFI layer, which cannot take a pointer to a zero-length shape buffer.
  const z = fromI32(Int32Array.from([7]), []);
  expect(z.shape).toEqual([]);
  expect(z.size).toBe(1);
});

// --- tidy() ownership ------------------------------------------------------

test("tidy frees its arena when the callback throws", () => {
  // Cleanup used to sit after a try/finally, so an exception skipped it — on
  // the error path, where a long-lived process needs it most.
  let made: MX | undefined;
  expect(() => tidy(() => { made = arr(); throw new Error("boom"); })).toThrow("boom");
  expect((made as MX).h).toBe(0);
});

test("tidy does not adopt an array it did not create", () => {
  // A nested tidy() returning one of its own inputs used to hand that array to
  // the parent arena, which then freed something its caller still owned.
  const external = arr();
  tidy(() => {
    tidy(() => external);
    return arr();
  });
  expect(external.h).not.toBe(0);
  expect(external.toF32()).toEqual([1, 2, 3, 4]);
  external.free();
});

test("tidy still frees what it does create, and keeps what it returns", () => {
  let temp: MX | undefined;
  const kept = tidy(() => { temp = arr(); return arr().add(arr()); });
  expect((temp as MX).h).toBe(0);
  expect(kept.h).not.toBe(0);
  expect(kept.toF32()).toEqual([2, 4, 6, 8]);
});

test("a nested tidy hands its own results to the parent", () => {
  // The counterpart to the case above: values the child created must still be
  // adopted, or the parent leaks them.
  let inner: MX | undefined;
  tidy(() => {
    inner = tidy(() => arr().add(arr()));
    return 0;
  });
  expect((inner as MX).h).toBe(0);
});

// --- host buffers are copied, not borrowed ---------------------------------

test("constructing from a TypedArray copies it", () => {
  // The MX class used to retain the source buffer for "zero-copy" arrays. It is
  // not zero-copy: mutating the source before any eval leaves the array
  // unchanged, so the retention only kept JS memory alive.
  const src = Float32Array.from([1, 2, 3, 4]);
  const a = fromF32(src, [4]);
  src[0] = 999;
  expect(a.toF32()).toEqual([1, 2, 3, 4]);
});

// --- custom Metal kernel lifetime ------------------------------------------

test("dispatching a kernel does not leak its outputs", async () => {
  // apply() used to leak three native objects per call. The costly one was the
  // vector the outputs come back in: it holds a reference to every output, so
  // freeing the returned MX released nothing. 1000 dispatches of a 1024-float
  // kernel grew MLX active memory by exactly 4.096 MB.
  const { activeMemoryMB, metalKernel } = await import("../src/index.ts");
  const N = 1024;
  const copy = metalKernel({
    name: "copyk_test",
    inputNames: ["x"],
    outputNames: ["y"],
    source: `uint i = thread_position_in_grid.x; y[i] = x[i];`,
  });
  const x = fromF32(Float32Array.from({ length: N }, (_, i) => i), [N]);

  const [warm] = copy.apply([x], [{ shape: [N] }], [N, 1, 1], [256, 1, 1]);
  warm.eval();
  warm.free();

  const before = activeMemoryMB();
  for (let i = 0; i < 200; i++) {
    const [y] = copy.apply([x], [{ shape: [N] }], [N, 1, 1], [256, 1, 1]);
    y.eval();
    y.free();
  }
  const grew = activeMemoryMB() - before;
  // Would be 0.819 MB at 200 dispatches if every output were retained.
  expect(grew).toBeLessThan(0.1);
  (copy as { free?: () => void }).free?.();
  x.free();
});

test("a failing op is named for itself, not for whatever runs next", async () => {
  // Ops that build their own result slot — concat, layerNorm, sdpa, the
  // quantized ones — bypass MX.r(). Until each of them consumed the error state
  // too, a failure here surfaced on the *next unrelated* call: a valid add threw
  // "mlx_add: [concatenate] All the input array dimensions must match...".
  const a = fromF32(Float32Array.from([1, 2, 3, 4]), [2, 2]);
  const b = fromF32(Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9]), [3, 3]);

  expect(() => a.concat(b, 0)).toThrow(/mlx_concatenate_axis: \[concatenate\]/);
  // And the error is consumed, so the next valid op is unaffected.
  expect(a.add(a).toF32()).toEqual([2, 4, 6, 8]);
});

test("a failed weight load does not poison the next operation", async () => {
  // Same class as the concat case, in a different file: loadSafetensors checked
  // its return code but never consumed mlx-c's message, so a missing checkpoint
  // surfaced later as
  //   "mlx_add: [load_safetensors] Failed to open file /nonexistent.safetensors".
  const { singleFileWeights } = await import("../src/io/loader.ts");
  const a = fromF32(Float32Array.from([1, 2, 3, 4]), [2, 2]);

  expect(() => singleFileWeights("/nonexistent-checkpoint.safetensors")).toThrow(/load_safetensors/);
  expect(a.add(a).toF32()).toEqual([2, 4, 6, 8]);
});
