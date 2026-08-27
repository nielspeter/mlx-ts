// Cross-entropy, against values worked out by hand.
//
// The parity suite checks it inside real training runs, where a wrong constant
// factor would still converge and look fine. These pin the actual number.
//   bun test tests/loss.test.ts
import { expect, test } from "bun:test";
import { crossEntropy, fromF32, fromU32, maskedCrossEntropy } from "../src/index.ts";

const logits = (v: number[][]) =>
  fromF32(Float32Array.from(v.flat()), [v.length, v[0].length]);
const targets = (v: number[]) => fromU32(Uint32Array.from(v), [v.length, 1]);

test("a confident correct prediction has near-zero loss", () => {
  const l = crossEntropy(logits([[20, 0, 0]]), targets([0])).toF32()[0];
  expect(l).toBeLessThan(1e-6);
});

test("a uniform distribution costs log(classes)", () => {
  const l = crossEntropy(logits([[0, 0, 0]]), targets([1])).toF32()[0];
  expect(l).toBeCloseTo(Math.log(3), 5);
});

test("a confident wrong prediction is expensive", () => {
  const l = crossEntropy(logits([[20, 0, 0]]), targets([2])).toF32()[0];
  expect(l).toBeGreaterThan(15);
});

test("the batch loss is the mean over rows", () => {
  const both = crossEntropy(logits([[0, 0, 0], [20, 0, 0]]), targets([1, 0])).toF32()[0];
  const a = crossEntropy(logits([[0, 0, 0]]), targets([1])).toF32()[0];
  const b = crossEntropy(logits([[20, 0, 0]]), targets([0])).toF32()[0];
  expect(both).toBeCloseTo((a + b) / 2, 5);
});

test("masking to a single row matches that row alone", () => {
  const lg = logits([[0, 0, 0], [20, 0, 0]]);
  const tg = targets([1, 0]);
  const mask = fromF32(Float32Array.from([1, 0]), [2, 1]);
  const masked = maskedCrossEntropy(lg, tg, mask).toF32()[0];
  const alone = crossEntropy(logits([[0, 0, 0]]), targets([1])).toF32()[0];
  expect(masked).toBeCloseTo(alone, 5);
});

test("a mask that keeps everything equals the plain loss", () => {
  const lg = logits([[1, 2, 3], [3, 2, 1]]);
  const tg = targets([2, 0]);
  const mask = fromF32(Float32Array.from([1, 1]), [2, 1]);
  expect(maskedCrossEntropy(lg, tg, mask).toF32()[0])
    .toBeCloseTo(crossEntropy(lg, tg).toF32()[0], 5);
});
