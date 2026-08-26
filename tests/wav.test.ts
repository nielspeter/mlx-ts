// Unit tests for the WAV writer's level handling.
//
// MusicGen overshoots ±1 often enough to matter, and the first fix for that —
// scale by 1/peak — silenced an entire clip the moment one sample was
// Infinity, because the gain became zero. Both directions are pinned here.
//   bun test tests/wav.test.ts
import { test, expect } from "bun:test";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { saveAudio } from "../src/audio/wav.ts";

/** Round-trip through a real file: the samples a player would actually read. */
async function roundTrip(samples: number[]): Promise<number[]> {
  const path = join(tmpdir(), `mlx-ts-wav-${samples.length}-${samples[1]}.wav`);
  await saveAudio(path, samples, 32000);
  const b = readFileSync(path);
  const v = new DataView(b.buffer, b.byteOffset);
  return Array.from({ length: (b.length - 44) / 2 }, (_, i) => v.getInt16(44 + i * 2, true) / 32768);
}

test("a clip that overshoots is scaled down, not clipped flat", async () => {
  const got = await roundTrip([0, 0.75, 1.5, -1.5, -0.75, 0]);
  expect(got[2]).toBeCloseTo(1, 3);            // the peak lands at full scale
  expect(got[3]).toBeCloseTo(-1, 3);
  expect(got[1] / got[2]).toBeCloseTo(0.5, 3); // ...and the shape is preserved
});

test("a clip within range is left alone", async () => {
  const got = await roundTrip([0, 0.25, -0.5]);
  expect(got[1]).toBeCloseTo(0.25, 3);         // not boosted to full scale
  expect(got[2]).toBeCloseTo(-0.5, 3);
});

test("one non-finite sample does not silence the clip", async () => {
  const got = await roundTrip([NaN, Infinity, -Infinity, 0.5]);
  expect(got[0]).toBe(0);                      // NaN -> silence
  expect(got[1]).toBeCloseTo(1, 3);            // clamped, not wrapped
  expect(got[2]).toBeCloseTo(-1, 3);
  expect(got[3]).toBeCloseTo(0.5, 3);          // the real sample survives
});
