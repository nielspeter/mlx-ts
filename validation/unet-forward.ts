// TS side of the UNet parity check, against mlx-examples' own port.
//
// Fixed latents, timestep and conditioning, so this tests the UNet alone —
// CLIP and the VAE are checked separately. 16x16 latents keep it cheap; the
// architecture is identical at SD's native 64.
//   /tmp/sdvenv/bin/python reference/reference-unet.py && bun validation/unet-forward.ts
import { fromF32 } from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { Unet, type UnetConfig } from "../src/models/unet.ts";

const REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5";

const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

const cfg = await readJson<UnetConfig>(await hubFile(REPO, "unet/config.json"));
const W = singleFileWeights(await hubFile(REPO, "unet/diffusion_pytorch_model.safetensors"));
const unet = new Unet(cfg, W);

const B = 1, H = 16, Wd = 16, C = cfg.in_channels, D = cfg.cross_attention_dim;
const x = fromF32(det(B * H * Wd * C, 1), [B, H, Wd, C]);
const cond = fromF32(det(B * 77 * D, 2), [B, 77, D]);

const out = unet.forward(x, 500, cond);
const f = out.toF32();
let sum = 0, abs = 0;
for (const v of f) { sum += v; abs += Math.abs(v); }
console.log(`unet shape=[${out.shape.join(", ")}] mean=${(sum / f.length).toFixed(6)} ` +
            `absmean=${(abs / f.length).toFixed(6)} ` +
            `first4=[${Array.from(f.slice(0, 4)).map((v) => +v.toFixed(4)).join(", ")}]`);
