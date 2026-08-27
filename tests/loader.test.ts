// The safetensors loader, round-tripped through a file this test writes.
//
// Every model in the repo loads weights this way, but each existing check needs
// a real checkpoint. Saving a handful of small tensors exercises the same
// path — open, look up by name, read shapes, enumerate, free — with nothing
// downloaded.
//   bun test tests/loader.test.ts
import { expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fromF32, MX, saveSafetensors } from "../src/index.ts";
import { entries, freeMap, get, loadSafetensors, shapeOf } from "../src/io/loader.ts";
import { singleFileWeights } from "../src/io/loader.ts";

const dir = mkdtempSync(join(tmpdir(), "mlx-ts-loader-"));

function writeFixture(name: string): string {
  const path = join(dir, name);
  saveSafetensors(path, {
    "a.weight": fromF32(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]),
    "a.bias": fromF32(Float32Array.from([7, 8]), [2]),
    scalarish: fromF32(Float32Array.from([9]), [1]),
  });
  return path;
}

test("a saved tensor comes back with its values and shape", () => {
  const w = loadSafetensors(writeFixture("basic.safetensors"));
  const a = new MX(get(w, "a.weight"));
  expect(a.shape).toEqual([2, 3]);
  expect([...a.toF32()]).toEqual([1, 2, 3, 4, 5, 6]);
  freeMap(w);
});

test("looking up a name that is not there throws, naming it", () => {
  const w = loadSafetensors(writeFixture("missing.safetensors"));
  expect(() => get(w, "nope.weight")).toThrow(/nope\.weight/);
  freeMap(w);
});

test("entries lists every tensor with its shape", () => {
  const w = loadSafetensors(writeFixture("entries.safetensors"));
  const found = entries(w).sort((x, y) => x.name.localeCompare(y.name));
  expect(found.map((e) => e.name)).toEqual(["a.bias", "a.weight", "scalarish"]);
  expect(found.find((e) => e.name === "a.weight")!.shape).toEqual([2, 3]);
  freeMap(w);
});

test("shapeOf reads a handle's shape", () => {
  const w = loadSafetensors(writeFixture("shape.safetensors"));
  expect(shapeOf(get(w, "a.bias"))).toEqual([2]);
  freeMap(w);
});

test("singleFileWeights hands tensors out by name", () => {
  const W = singleFileWeights(writeFixture("weights.safetensors"));
  expect([...W.mx("a.bias").toF32()]).toEqual([7, 8]);
  expect(() => W.mx("absent")).toThrow();
  W.done();
});

test("the same tensor can be fetched twice", () => {
  const W = singleFileWeights(writeFixture("twice.safetensors"));
  expect([...W.mx("a.weight").toF32()]).toEqual([...W.mx("a.weight").toF32()]);
  W.done();
});
