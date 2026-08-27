// Write an RGB image to a PNG file.
//
// The counterpart to src/audio/wav.ts: the smallest correct encoder that gets
// pixels out of the process and into something you can look at. No filtering
// (every scanline uses filter type 0) and one IDAT chunk — larger than an
// optimised PNG, identical to view.
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { deflateSync } from "node:zlib";

const SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

// Standard CRC-32, table built once.
const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/**
 * Write `pixels` — float RGB in [0, 1], row-major, `width * height * 3` long —
 * to `path` as an 8-bit RGB PNG. Values outside the range are clamped.
 */
export async function savePng(
  path: string, pixels: ArrayLike<number>, width: number, height: number,
): Promise<void> {
  // Each scanline is prefixed with its filter type; 0 means "no filtering".
  const raw = new Uint8Array(height * (1 + width * 3));
  let p = 0;
  for (let y = 0; y < height; y++) {
    raw[p++] = 0;
    for (let i = 0; i < width * 3; i++) {
      const v = pixels[y * width * 3 + i];
      raw[p++] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(v * 255);
    }
  }

  const ihdr = new Uint8Array(13);
  const hv = new DataView(ihdr.buffer);
  hv.setUint32(0, width);
  hv.setUint32(4, height);
  ihdr[8] = 8;    // bit depth
  ihdr[9] = 2;    // colour type 2 = RGB
  // bytes 10-12: compression, filter and interlace methods, all 0

  const parts = [
    SIGNATURE,
    chunk("IHDR", ihdr),
    chunk("IDAT", new Uint8Array(deflateSync(raw))),
    chunk("IEND", new Uint8Array(0)),
  ];
  const file = new Uint8Array(parts.reduce((n, b) => n + b.length, 0));
  let off = 0;
  for (const b of parts) { file.set(b, off); off += b.length; }

  await mkdir(dirname(path), { recursive: true }).catch(() => {});
  await writeFile(path, file);
}
