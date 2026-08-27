// Comparing against the committed reference numbers in spark-golden.json.
//
// Those numbers come from the original PyTorch Spark-TTS, generated once by
// reference/gen-spark-fixtures.py and checked in — so these checks need nothing
// but mlx-ts and a cached checkpoint. No Python, no PyTorch, no fetching a
// package that is not on PyPI. That matters: an oracle nobody can run is an
// oracle that always skips.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { MX } from "../src/index.ts";

export type Stats = { shape: number[]; mean: number; absmean: number; first4: number[] };

export const golden = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "spark-golden.json"), "utf8"),
);

let failures = 0;

/** Relative tolerance. Both sides are float32; 1e-4 is far tighter than any real bug. */
const close = (a: number, b: number, rel = 1e-4) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));

/**
 * Compare a tensor's fingerprint. `first4` is only checked when the layouts
 * agree — ours is channels-last where PyTorch is channels-first, so a 3-D
 * tensor flattens differently even when every element matches. mean and absmean
 * are layout-invariant, which is why they carry the comparison.
 */
export function check(tag: string, a: MX, want: Stats, sameLayout = false): void {
  const f = a.toF32();
  let sum = 0, abs = 0;
  for (const v of f) { sum += v; abs += Math.abs(v); }
  const mean = sum / f.length, absmean = abs / f.length;

  const bad: string[] = [];
  if (a.shape.length !== want.shape.length || f.length !== want.shape.reduce((x, y) => x * y, 1)) {
    bad.push(`shape [${a.shape}] vs [${want.shape}]`);
  }
  if (!close(mean, want.mean)) bad.push(`mean ${mean.toFixed(6)} vs ${want.mean.toFixed(6)}`);
  if (!close(absmean, want.absmean)) bad.push(`absmean ${absmean.toFixed(6)} vs ${want.absmean.toFixed(6)}`);
  if (sameLayout) {
    for (let i = 0; i < 4; i++) {
      if (!close(f[i], want.first4[i], 1e-3)) bad.push(`[${i}] ${f[i].toFixed(5)} vs ${want.first4[i].toFixed(5)}`);
    }
  }

  console.log(`  ${bad.length ? "FAIL" : "ok  "} ${tag.padEnd(10)} shape=[${a.shape.join(", ")}] ` +
              `mean=${mean.toFixed(6)} absmean=${absmean.toFixed(6)}` +
              (bad.length ? `\n         ${bad.join("; ")}` : ""));
  if (bad.length) failures++;
}

/** Integer sequences — token ids — must match exactly. There is no tolerance. */
export function checkIds(tag: string, got: number[], want: number[]): void {
  const at = got.length === want.length ? got.findIndex((v, i) => v !== want[i]) : -2;
  const ok = at === -1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${tag.padEnd(10)} ${got.length} ids` +
              (ok ? "" : at === -2 ? `\n         length ${got.length} vs ${want.length}`
                                   : `\n         differs at ${at}: ${got[at]} vs ${want[at]}`));
  if (!ok) failures++;
}

/** Print the verdict and set the exit code. */
export function verdict(name: string): void {
  console.log(failures === 0 ? `${name}: ok` : `${name}: ${failures} MISMATCH`);
  if (failures) process.exit(1);
}
