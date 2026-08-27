// Decode an image file into the pixels a vision model expects.
//
// The counterpart to decodeAudio in src/audio/mel.ts, and the same bargain:
// shelling out to ffmpeg buys every format it supports — JPEG, PNG, WebP, HEIC
// — for one dependency the project already asks for, rather than a decoder per
// format in TypeScript.
import { spawn } from "node:child_process";

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

/**
 * `path` -> normalised RGB, row-major, `size * size * 3` long.
 *
 * The image is scaled so its shorter side reaches `size` and then centre
 * cropped, which is what CLIP's own preprocessor does — squashing to a square
 * instead distorts the aspect ratio and measurably moves the embedding.
 */
export async function loadImage(path: string, opts: LoadImageOptions = {}): Promise<Float32Array> {
  const { size = 224, mean = CLIP_MEAN, std = CLIP_STD } = opts;

  // node:child_process rather than Bun.spawn — Deno and Node implement it too.
  const p = spawn("ffmpeg", [
    "-nostdin", "-i", path,
    "-vf", `scale=${size}:${size}:force_original_aspect_ratio=increase,crop=${size}:${size}`,
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
  ], { stdio: ["ignore", "pipe", "ignore"] });

  const chunks: Uint8Array[] = [];
  for await (const c of p.stdout) chunks.push(c as Uint8Array);
  const buf = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }

  const want = size * size * 3;
  if (buf.length < want) {
    throw new Error(`loadImage: ffmpeg produced ${buf.length} bytes, expected ${want} for ${path}`);
  }

  const px = new Float32Array(want);
  for (let i = 0; i < want; i++) px[i] = (buf[i] / 255 - mean[i % 3]) / std[i % 3];
  return px;
}
