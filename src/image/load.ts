// Decode an image file into the pixels a vision model expects.
//
// Uses macOS's own `sips`, which is built in. MLX is Apple-Silicon-only, so
// requiring a separate install to read a JPEG buys nothing — and sips reads
// everything ImageIO does, which is every format anyone will hand it.
//
// sips writes images, not raw buffers, so it emits an uncompressed 24-bit BMP
// and this file reads that back. BMP is a header and a pixel block: no
// decompression, about fifteen lines.
import { spawn } from "node:child_process";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { readBytes } from "../io/fs.ts";

/** CLIP's channel statistics. Skipping them shifts every embedding. */
export const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
export const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

export type LoadImageOptions = {
  /** Square output edge, in pixels. CLIP ViT-L/14 wants 224. */
  size?: number;
  /** Per-channel mean and standard deviation; defaults to CLIP's. */
  mean?: number[];
  std?: number[];
};

function run(cmd: string, args: string[]): Promise<string> {
  // node:child_process rather than Bun.spawn — Deno and Node implement it too.
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: ["ignore", "pipe", "ignore"] });
    let out = "";
    p.stdout.on("data", (c: Buffer) => { out += c.toString(); });
    p.on("error", () => reject(new Error(`${cmd} not found on PATH`)));
    p.on("close", (code) => (code === 0 ? resolve(out) : reject(new Error(`${cmd} exited ${code}`))));
  });
}

/** Pixel dimensions, so the shorter side can be the one that gets resampled. */
async function dimensions(path: string): Promise<{ width: number; height: number }> {
  const out = await run("sips", ["-g", "pixelWidth", "-g", "pixelHeight", path]);
  const width = Number(out.match(/pixelWidth:\s*(\d+)/)?.[1]);
  const height = Number(out.match(/pixelHeight:\s*(\d+)/)?.[1]);
  if (!width || !height) throw new Error(`loadImage: could not read the size of ${path}`);
  return { width, height };
}

/** Uncompressed 24-bit BMP -> RGB bytes, top-down, row padding removed. */
function readBmp(buf: Uint8Array): { rgb: Uint8Array; width: number; height: number } {
  if (buf[0] !== 0x42 || buf[1] !== 0x4d) throw new Error("loadImage: sips did not produce a BMP");
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const offset = v.getUint32(10, true);
  const width = v.getInt32(18, true);
  const signedHeight = v.getInt32(22, true);
  const bpp = v.getUint16(28, true);
  if (bpp !== 24) throw new Error(`loadImage: expected a 24-bit BMP, got ${bpp}`);

  // A negative height means the rows are stored top-down; positive is bottom-up.
  const height = Math.abs(signedHeight);
  const topDown = signedHeight < 0;
  const stride = (width * 3 + 3) & ~3;            // rows pad to 4 bytes

  const rgb = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const src = offset + (topDown ? y : height - 1 - y) * stride;
    for (let x = 0; x < width; x++) {
      const s = src + x * 3, d = (y * width + x) * 3;
      rgb[d] = buf[s + 2];                        // BMP stores BGR
      rgb[d + 1] = buf[s + 1];
      rgb[d + 2] = buf[s];
    }
  }
  return { rgb, width, height };
}

/**
 * `path` -> normalised RGB, row-major, `size * size * 3` long.
 *
 * The shorter side is scaled to `size` and the result centre-cropped, which is
 * what CLIP's own preprocessor does — resampling both sides to a square
 * distorts the aspect ratio and measurably moves the embedding.
 */
export async function loadImage(path: string, opts: LoadImageOptions = {}): Promise<Float32Array> {
  const { size = 224, mean = CLIP_MEAN, std = CLIP_STD } = opts;
  const { width, height } = await dimensions(path);
  const out = join(tmpdir(), `mlx-ts-image-${process.pid}-${basename(path)}.bmp`);

  try {
    await run("sips", [
      // Scale the shorter side up to `size`; the longer one follows.
      ...(width < height ? ["--resampleWidth", String(size)] : ["--resampleHeight", String(size)]),
      "--cropToHeightWidth", String(size), String(size),
      "-s", "format", "bmp", path, "--out", out,
    ]);
    const { rgb, width: w, height: h } = readBmp(await readBytes(out));
    if (w !== size || h !== size) throw new Error(`loadImage: got ${w}x${h}, expected ${size}x${size}`);

    const px = new Float32Array(rgb.length);
    for (let i = 0; i < rgb.length; i++) px[i] = (rgb[i] / 255 - mean[i % 3]) / std[i % 3];
    return px;
  } finally {
    await rm(out, { force: true });
  }
}
