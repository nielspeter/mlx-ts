// Unit tests for Owned — the cross-scope half of the memory model.
//
// tidy() covers values that die with their scope; Owned covers the ones that
// must outlive it. The bug it exists to prevent — not freeing the value being
// replaced — leaked ~10 MB per step and reached 55 GB, and was invisible to
// every test we had, because nothing exercised ownership transfer directly.
//   bun test tests/owned.test.ts
import { test, expect } from "bun:test";
import { MX, fromF32, tidy, activeMemoryMB, evalAll, Owned, freeAll } from "../src/index.ts";

const arr = () => fromF32(new Float32Array(256).fill(1), [256]);
/** A freed MX has its handle zeroed, so liveness is directly observable. */
const alive = (x: MX) => x.h !== 0;

test("set() frees the value it replaces", () => {
  using slots = new Owned<MX>(1);
  const first = slots.set(0, arr());
  expect(alive(first)).toBe(true);
  const second = slots.set(0, arr());
  expect(alive(first)).toBe(false);      // replaced, so freed
  expect(alive(second)).toBe(true);
});

test("set() keeps anything the new value still holds", () => {
  using slots = new Owned<{ k: MX; v: MX }>(1);
  const k = arr();
  slots.set(0, { k, v: arr() });
  slots.set(0, { k, v: arr() });          // same k, new v
  expect(alive(k)).toBe(true);            // still referenced — must not be freed
});

test("a value set inside tidy() survives the tidy()", () => {
  const slots = new Owned<MX>(1);
  const kept = tidy(() => {
    const x = arr();
    slots.set(0, x);                      // escapes the arena
    arr();                                // a scope-local, should not survive
    return x;
  });
  expect(alive(kept)).toBe(true);
  slots.free();
  expect(alive(kept)).toBe(false);
});

test("free() empties the table and is idempotent", () => {
  const slots = new Owned<MX>(3);
  const xs = [slots.set(0, arr()), slots.set(1, arr()), slots.set(2, arr())];
  slots.free();
  for (const x of xs) expect(alive(x)).toBe(false);
  expect(() => slots.free()).not.toThrow();
  expect(slots.get(0)).toBeNull();
});

test("using disposes the table at scope exit", () => {
  let escaped: MX;
  {
    using slots = new Owned<MX>(1);
    escaped = slots.set(0, arr());
    expect(alive(escaped)).toBe(true);
  }
  expect(alive(escaped!)).toBe(false);
});

test("the table grows on demand", () => {
  using slots = new Owned<MX>(0);
  slots.set(5, arr());
  expect(slots.length).toBe(6);
  expect(slots.get(5)).not.toBeNull();
  expect(slots.get(2)).toBeNull();
});

test("repeated replacement does not accumulate memory", () => {
  using slots = new Owned<{ k: MX; v: MX }>(4);
  for (let l = 0; l < 4; l++) slots.set(l, { k: arr(), v: arr() });
  evalAll();
  const before = activeMemoryMB();
  for (let step = 0; step < 200; step++)
    for (let l = 0; l < 4; l++) slots.set(l, { k: arr(), v: arr() });
  evalAll();
  expect(activeMemoryMB() - before).toBeLessThan(1);   // 800 replacements
});

test("freeAll walks arrays and nested objects", () => {
  const xs = [arr(), { a: arr(), b: [arr()] }];
  freeAll(xs);
  expect(alive(xs[0] as MX)).toBe(false);
  expect(alive((xs[1] as { a: MX }).a)).toBe(false);
  expect(alive((xs[1] as { b: MX[] }).b[0])).toBe(false);
});
