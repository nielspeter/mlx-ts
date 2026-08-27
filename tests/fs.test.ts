// The tiny filesystem helpers every loader goes through.
//   bun test tests/fs.test.ts
import { expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readBytes, readJson, readText, writeJson, writeText } from "../src/io/fs.ts";

const dir = mkdtempSync(join(tmpdir(), "mlx-ts-fs-"));

test("text round-trips", async () => {
  const p = join(dir, "a.txt");
  await writeText(p, "hello\nworld");
  expect(await readText(p)).toBe("hello\nworld");
});

test("json round-trips, including nesting", async () => {
  const p = join(dir, "a.json");
  const value = { a: 1, b: [1, 2, { c: "x" }], d: null };
  await writeJson(p, value);
  expect(await readJson<typeof value>(p)).toEqual(value);
});

test("writeJson is indented, so a config stays readable", async () => {
  const p = join(dir, "indent.json");
  await writeJson(p, { a: 1 });
  expect(await readText(p)).toContain("\n  ");
});

test("readBytes returns the raw bytes", async () => {
  const p = join(dir, "bytes.txt");
  await writeText(p, "AB");
  expect([...(await readBytes(p))]).toEqual([65, 66]);
});

test("writing creates missing directories", async () => {
  const p = join(dir, "nested", "deep", "x.txt");
  await writeText(p, "ok");
  expect(await readText(p)).toBe("ok");
});

test("reading a missing file rejects", async () => {
  await expect(readText(join(dir, "nope.txt"))).rejects.toThrow();
});
