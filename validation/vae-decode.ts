// TS side of the VAE decoder parity check: latents in, pixels out.
//
// Both sides load stabilityai/sd-vae-ft-mse and decode the same deterministic
// latents; the oracle is mlx-examples' own Stable Diffusion port, so this is
// our decoder against theirs, not against a re-derivation of the same maths.
//   MLX_SD=... /tmp/sdvenv/bin/python reference/reference-vae.py && bun validation/vae-decode.ts
import { fromF32, tidy } from "../src/index.ts";
import { hubFile } from "../src/io/hub.ts";
import { readJson } from "../src/io/fs.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { VaeDecoder, type VaeConfig } from "../src/models/vae.ts";

const REPO = "stabilityai/sd-vae-ft-mse";

const det = (n: number, seed: number) =>
  Float32Array.from({ length: n }, (_, i) => ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5);

const cfg = await readJson<VaeConfig & { scaling_factor?: number }>(await hubFile(REPO, "config.json"));
const W = singleFileWeights(await hubFile(REPO, "diffusion_pytorch_model.safetensors"));
const vae = new VaeDecoder({ ...cfg, scaling_factor: cfg.scaling_factor ?? 0.18215 }, W);

const B = 1, h = 16, w = 16, C = cfg.latent_channels;
const img = tidy(() => vae.decode(fromF32(det(B * h * w * C, 1), [B, h, w, C])));

const f = img.toF32();
let sum = 0; for (const v of f) sum += v;
console.log(`vae_decode shape=[${img.shape.join(", ")}] sum=${sum.toFixed(3)} ` +
            `mean=${(sum / f.length).toFixed(5)} ` +
            `first4=[${Array.from(f.slice(0, 4)).map((v) => +v.toFixed(4)).join(", ")}]`);
