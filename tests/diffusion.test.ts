// The noise schedule, checked as arithmetic.
//
// validation/scheduler.ts compares it against mlx-examples; these pin the
// properties that make it a schedule at all, so a mismatch says which
// assumption broke rather than just "the numbers moved".
//   bun test tests/diffusion.test.ts
import { expect, test } from "bun:test";
import { EulerSampler, fromF32 } from "../src/index.ts";

const SD = { beta_start: 0.00085, beta_end: 0.012, beta_schedule: "scaled_linear", num_train_timesteps: 1000 };
const make = (over: Partial<typeof SD> = {}) => new EulerSampler({ ...SD, ...over });

test("there is one sigma per training step, plus zero at the start", () => {
  const s = make();
  expect(s.sigmas.length).toBe(1001);
  expect(s.maxTime).toBe(1000);
  expect(s.sigmas[0]).toBe(0);
});

test("sigma rises monotonically with time", () => {
  const s = make();
  for (let i = 1; i < s.sigmas.length; i++) expect(s.sigmas[i]).toBeGreaterThan(s.sigmas[i - 1]);
});

test("sigma interpolates between the tabulated values", () => {
  const s = make();
  const mid = s.sigma(10.5);
  expect(mid).toBeGreaterThan(s.sigmas[10]);
  expect(mid).toBeLessThan(s.sigmas[11]);
  expect(mid).toBeCloseTo((s.sigmas[10] + s.sigmas[11]) / 2, 9);
});

test("integer times land exactly on the table", () => {
  const s = make();
  expect(s.sigma(0)).toBe(s.sigmas[0]);
  expect(s.sigma(500)).toBeCloseTo(s.sigmas[500], 12);
  expect(s.sigma(1000)).toBeCloseTo(s.sigmas[1000], 12);
});

test("scaled_linear is not the same schedule as linear", () => {
  // SD trains on scaled_linear; the plain one still makes an image, just worse,
  // so nothing downstream would flag the swap.
  expect(make().sigmas[500]).not.toBeCloseTo(make({ beta_schedule: "linear" }).sigmas[500], 3);
});

test("timesteps walk from maxTime down to zero without gaps", () => {
  const pairs = make().timesteps(8);
  expect(pairs.length).toBe(8);
  expect(pairs[0][0]).toBe(1000);
  expect(pairs[7][1]).toBe(0);
  for (let i = 1; i < pairs.length; i++) expect(pairs[i][0]).toBe(pairs[i - 1][1]);
});

test("a step towards the same time changes nothing", () => {
  const s = make();
  const x = fromF32(Float32Array.from([1, 2, 3]), [3]);
  const eps = fromF32(Float32Array.from([9, 9, 9]), [3]);
  const same = s.step(eps, x, 100, 100).toF32();
  expect([...same]).toEqual([1, 2, 3]);
});

test("stepping to zero leaves no noise scaling", () => {
  const s = make();
  const x = fromF32(Float32Array.from([1]), [1]);
  const eps = fromF32(Float32Array.from([0]), [1]);
  // With zero predicted noise, x is only rescaled by the sigma ratio.
  const out = s.step(eps, x, 500, 0).toF32()[0];
  expect(out).toBeCloseTo(Math.sqrt(s.sigma(500) ** 2 + 1), 4);
});

test("the prior scale is below one and rises with the largest sigma", () => {
  const p = make().priorScale();
  expect(p).toBeGreaterThan(0.9);
  expect(p).toBeLessThan(1);
});
