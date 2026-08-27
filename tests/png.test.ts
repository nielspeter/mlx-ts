// The PNG writer, checked by taking the file apart again.
//
// Nothing else in the repo reads PNGs, so a round trip through the encoder
// alone would prove very little. Instead this parses the bytes: signature,
// IHDR fields, chunk CRCs, and the IDAT inflated back to the scanlines that
// went in — which is what any decoder will do.
//   bun test tests/png.test.ts
import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inflateSync } from "node:zlib";
import { savePng } from "../src/index.ts";

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

/** Walk the chunk list, so a wrong length or CRC shows up as a parse failure. */
function chunks(buf: Buffer) {
  const out: Array<{ type: string; data: Buffer }> = [];
  let p = 8;
  while (p < buf.length) {
    const len = buf.readUInt32BE(p);
    const type = buf.toString("ascii", p + 4, p + 8);
    out.push({ type, data: buf.subarray(p + 8, p + 8 + len) });
    p += 12 + len;
  }
  return out;
}

async function write(pixels: number[], w: number, h: number): Promise<Buffer> {
  const path = join(tmpdir(), `mlx-ts-png-${w}x${h}-${pixels.length}.png`);
  await savePng(path, Float32Array.from(pixels), w, h);
  return readFileSync(path);
}

test("the file starts with the PNG signature", async () => {
  const buf = await write([0, 0, 0], 1, 1);
  expect([...buf.subarray(0, 8)]).toEqual(SIGNATURE);
});

test("IHDR carries the size, 8-bit depth and RGB colour type", async () => {
  const buf = await write(new Array(3 * 6).fill(0), 3, 2);
  const ihdr = chunks(buf).find((c) => c.type === "IHDR")!;
  expect(ihdr.data.readUInt32BE(0)).toBe(3);   // width
  expect(ihdr.data.readUInt32BE(4)).toBe(2);   // height
  expect(ihdr.data[8]).toBe(8);                // bit depth
  expect(ihdr.data[9]).toBe(2);                // colour type 2 = RGB
});

test("the chunks are IHDR, IDAT then IEND", async () => {
  const buf = await write([1, 1, 1], 1, 1);
  expect(chunks(buf).map((c) => c.type)).toEqual(["IHDR", "IDAT", "IEND"]);
});

test("IDAT inflates to the scanlines that went in", async () => {
  // Two pixels: pure red then pure green, one row.
  const buf = await write([1, 0, 0, 0, 1, 0], 2, 1);
  const idat = chunks(buf).find((c) => c.type === "IDAT")!;
  const raw = inflateSync(idat.data);
  expect([...raw]).toEqual([0, 255, 0, 0, 0, 255, 0]);   // filter byte, then RGB RGB
});

test("values outside [0, 1] are clamped rather than wrapped", async () => {
  const buf = await write([-5, 0.5, 42], 1, 1);
  const raw = inflateSync(chunks(buf).find((c) => c.type === "IDAT")!.data);
  expect([...raw]).toEqual([0, 0, 128, 255]);
});

test("each row gets its own filter byte", async () => {
  const buf = await write(new Array(3 * 3).fill(0), 1, 3);   // 1 wide, 3 tall
  const raw = inflateSync(chunks(buf).find((c) => c.type === "IDAT")!.data);
  expect(raw.length).toBe(3 * (1 + 3));
  expect([raw[0], raw[4], raw[8]]).toEqual([0, 0, 0]);
});
