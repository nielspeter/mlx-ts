// Unit tests for the arena itself — tidy()/escape() ownership semantics.
//
// The parity suite checks numbers against MLX-Python and is very good at that,
// but it never tested the SDK's own memory contract. That gap is why Adam was
// unusable inside tidy() for as long as it was: nothing exercised "state
// created during a tidy() must survive it".
//   bun test tests/arena.test.ts
import { expect, test } from "bun:test";
import type { Tree } from "../src/index.ts";
import { Adam, escape, evalAll, fromF32, MX, scalar, tidy, valueAndGrad } from "../src/index.ts";

const arr = (v = 1) => fromF32(new Float32Array([v, v, v, v]), [2, 2]);
const freed = (x: MX) => x.h === 0;

test("tidy frees intermediates and keeps the returned value", () => {
  let inner!: MX;
  const kept = tidy(() => {
    inner = arr(2);
    return arr(3);
  });
  expect(freed(inner)).toBe(true);
  expect(freed(kept)).toBe(false);
  kept.free();
});

test("tidy keeps values nested in objects and arrays", () => {
  let loose!: MX;
  const kept = tidy(() => {
    loose = arr();
    return { pair: [arr(1), arr(2)] };
  });
  expect(freed(loose)).toBe(true);
  for (const x of kept.pair) expect(freed(x)).toBe(false);
  kept.pair.forEach((x) => x.free());
});

test("nested tidy: what the inner one keeps is adopted by the outer", () => {
  let innerTemp!: MX;
  const outerTemp: MX[] = [];
  const kept = tidy(() => {
    const fromInner = tidy(() => { innerTemp = arr(); return arr(5); });
    outerTemp.push(fromInner);
    return arr(6);
  });
  expect(freed(innerTemp)).toBe(true);
  // adopted by the outer arena, so the outer tidy freed it too
  expect(freed(outerTemp[0])).toBe(true);
  expect(freed(kept)).toBe(false);
  kept.free();
});

test("escape() survives the enclosing tidy and transfers ownership", () => {
  let escaped!: MX;
  tidy(() => { escaped = escape(arr(7)); });
  expect(freed(escaped)).toBe(false);
  escaped.free();
  expect(freed(escaped)).toBe(true);
});

test("Adam's state survives across steps inside tidy()", () => {
  // Regression: the moment buffers are created during update(), so an enclosing
  // tidy() used to free them and the next step read freed handles.
  let params: Tree = { w: arr(0.5) };
  const step = valueAndGrad(params, (p: Tree) => (p as { w: MX }).w.meanAll());
  const opt = new Adam(0.01);
  let last = 0;
  for (let i = 0; i < 5; i++) {
    const r = tidy(() => {
      const { loss, grads } = step(params);
      return { loss, next: opt.update(params, grads) };
    });
    params = r.next;
    last = r.loss;
    evalAll(...[(params as { w: MX }).w]);
  }
  expect(Number.isFinite(last)).toBe(true);
  expect(freed((params as { w: MX }).w)).toBe(false);
});

test("scalar and fromF32 round-trip through the arena", () => {
  const v = tidy(() => scalar(3).mul(scalar(4)));
  expect(v.itemF()).toBeCloseTo(12, 5);
  v.free();
});
