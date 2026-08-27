// pytree: the parameter-tree walk every optimizer and autograd call goes
// through. Pure structure, no MLX maths, so it is cheap to pin exactly.
//   bun test tests/pytree.test.ts
import { expect, test } from "bun:test";
import { fromF32, type MX, type Tree, treeFlatten, treeMap, treeUnflattenLike } from "../src/index.ts";

const leaf = (v: number) => fromF32(Float32Array.from([v]), [1]);
const val = (x: MX) => x.toF32()[0];

test("flatten walks objects and arrays in a stable order", () => {
  const t: Tree = { a: leaf(1), b: [leaf(2), leaf(3)], c: { d: leaf(4) } };
  expect(treeFlatten(t).map(val)).toEqual([1, 2, 3, 4]);
});

test("unflattenLike restores the template's shape", () => {
  const t: Tree = { a: leaf(1), b: [leaf(2), leaf(3)] };
  const rebuilt = treeUnflattenLike(t, [leaf(9), leaf(8), leaf(7)]) as {
    a: MX; b: MX[];
  };
  expect(val(rebuilt.a)).toBe(9);
  expect(rebuilt.b.map(val)).toEqual([8, 7]);
});

test("flatten then unflattenLike is a round trip", () => {
  const t: Tree = { w: leaf(1), inner: { b: leaf(2), c: [leaf(3), leaf(4)] } };
  const back = treeUnflattenLike(t, treeFlatten(t));
  expect(treeFlatten(back).map(val)).toEqual([1, 2, 3, 4]);
});

test("treeMap applies to every leaf and keeps the structure", () => {
  const t: Tree = { a: leaf(1), b: [leaf(2)] };
  const doubled = treeMap((x) => x.add(x), t) as { a: MX; b: MX[] };
  expect(val(doubled.a)).toBe(2);
  expect(val(doubled.b[0])).toBe(4);
});

test("a bare leaf is a tree", () => {
  expect(treeFlatten(leaf(5)).map(val)).toEqual([5]);
  expect(val(treeUnflattenLike(leaf(0), [leaf(6)]) as MX)).toBe(6);
});
