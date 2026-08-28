// The UNet: fixed latents, timestep and conditioning, so this is the denoiser
// alone — CLIP and the VAE are checked separately.
//
// 16x16 latents keep it cheap; the architecture is identical at SD's native 64,
// but it still needs the 3.2 GB checkpoint cached.
//
// Reference: diffusers' UNet2DConditionModel (see validation/golden.ts), stored
// channels-last to match ours.
//   bun validation/unet-forward.ts
import { fromF32 } from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { Unet, type UnetConfig } from "../src/models/unet.ts";
import { check, loadGolden, verdict } from "./golden.ts";

const REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5";
const g = loadGolden("sd-golden.json").unet;

const cfg = await readJson<UnetConfig>(await hubFile(REPO, "unet/config.json"));
const unet = new Unet(cfg, singleFileWeights(await hubFile(REPO, "unet/diffusion_pytorch_model.safetensors")));

const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);
const [B, H, W, C] = g.latent as number[];
const [, T, D] = g.cond as number[];

const x = fromF32(det(B * H * W * C, 1), [B, H, W, C]);
const cond = fromF32(det(B * T * D, 2), [B, T, D]);
check("unet", unet.forward(x, g.timestep, cond), g, true);
verdict("UNet");
