// The VAE decoder: latents in, pixels out.
//
// Reference: diffusers' AutoencoderKL on the same checkpoint, so a difference is
// our decoder rather than the weights (see validation/golden.ts). The reference
// fingerprint is stored channels-last, matching ours, so the leading pixels
// compare directly.
//   bun validation/vae-decode.ts
import { fromF32, tidy } from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { type VaeConfig, VaeDecoder } from "../src/models/vae.ts";
import { check, loadGolden, verdict } from "./golden.ts";

const REPO = "stabilityai/sd-vae-ft-mse";
const g = loadGolden("sd-golden.json").vae;

const cfg = await readJson<VaeConfig & { scaling_factor?: number }>(await hubFile(REPO, "config.json"));
const vae = new VaeDecoder({ ...cfg, scaling_factor: cfg.scaling_factor ?? 0.18215 },
                           singleFileWeights(await hubFile(REPO, "diffusion_pytorch_model.safetensors")));

const [B, h, w, C] = g.latent as number[];
const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

check("vae_decode", tidy(() => vae.decode(fromF32(det(B * h * w * C, g.seed), [B, h, w, C]))), g, true);
verdict("VAE decoder");
