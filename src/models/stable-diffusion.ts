// Stable Diffusion, end to end: a prompt in, an image out.
//
// The four pieces are checked separately against mlx-examples — CLIP's
// tokenizer and text encoder, the UNet, the VAE decoder, and the noise
// schedule — so this file is only the loop that joins them.
//
// Memory: the UNet alone is 3.2 GB of weights and the loop allocates hard, so
// MLX's buffer-reuse cache is capped for the duration. Uncapped it grows to
// whatever the machine has (see docs/FINDINGS.md 6.7).
import { evalAll, fromU32, type MX, randomNormal, setCacheLimit, tidy } from "../core/mx.ts";
import { readJson } from "../io/fs.ts";
import { hubFile } from "../io/hub.ts";
import { singleFileWeights } from "../io/loader.ts";
import { ClipTokenizer } from "../text/clip-tokenizer.ts";
import { type ClipConfig, ClipTextEncoder } from "./clip.ts";
import { type DiffusionConfig, EulerSampler } from "./diffusion.ts";
import { Unet, type UnetConfig } from "./unet.ts";
import { type VaeConfig, VaeDecoder } from "./vae.ts";

const DEFAULT_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5";

export type ImageOptions = {
  /** Output size in pixels; the VAE upsamples by 8, so this must be a multiple. */
  width?: number;
  height?: number;
  steps?: number;
  /** Classifier-free guidance: how hard to push towards the prompt. */
  guidance?: number;
  negativePrompt?: string;
  seed?: number;
  cacheLimitMB?: number;
  onStep?: (step: number, total: number) => void;
};

export class StableDiffusion {
  private tok: ClipTokenizer;
  private clip: ClipTextEncoder;
  private unet: Unet;
  private vae: VaeDecoder;
  private sampler: EulerSampler;

  private constructor(tok: ClipTokenizer, clip: ClipTextEncoder, unet: Unet,
                      vae: VaeDecoder, sampler: EulerSampler) {
    this.tok = tok; this.clip = clip; this.unet = unet;
    this.vae = vae; this.sampler = sampler;
  }

  static async fromPretrained(repo = DEFAULT_REPO): Promise<StableDiffusion> {
    const tok = await ClipTokenizer.fromFiles(
      await hubFile(repo, "tokenizer/vocab.json"), await hubFile(repo, "tokenizer/merges.txt"));

    const clipCfg = (await readJson<{ text_config?: ClipConfig } & ClipConfig>(
      await hubFile(repo, "text_encoder/config.json")));
    const clip = new ClipTextEncoder(clipCfg.text_config ?? clipCfg,
      singleFileWeights(await hubFile(repo, "text_encoder/model.safetensors")));

    const unet = new Unet(await readJson<UnetConfig>(await hubFile(repo, "unet/config.json")),
      singleFileWeights(await hubFile(repo, "unet/diffusion_pytorch_model.safetensors")));

    const vaeCfg = await readJson<VaeConfig>(await hubFile(repo, "vae/config.json"));
    const vae = new VaeDecoder({ ...vaeCfg, scaling_factor: vaeCfg.scaling_factor ?? 0.18215 },
      singleFileWeights(await hubFile(repo, "vae/diffusion_pytorch_model.safetensors")));

    const sampler = new EulerSampler(
      await readJson<DiffusionConfig>(await hubFile(repo, "scheduler/scheduler_config.json")));

    return new StableDiffusion(tok, clip, unet, vae, sampler);
  }

  /** A prompt's CLIP states, padded to the 77-token window SD always feeds. */
  private conditioning(text: string): MX {
    const ids = this.tok.encode(text, { padTo: 77 });
    return this.clip.encode(fromU32(Uint32Array.from(ids), [1, ids.length]));
  }

  /** Prompt -> image, `[height, width, 3]` with values in [0, 1]. */
  generate(prompt: string, opts: ImageOptions = {}): MX {
    const {
      width = 512, height = 512, steps = 20, guidance = 7.5,
      negativePrompt = "", seed, cacheLimitMB = 2048, onStep,
    } = opts;

    const prev = setCacheLimit(cacheLimitMB);
    try {
      // Both prompts in one batch, so a single UNet pass yields the conditional
      // and unconditional predictions that guidance mixes.
      const cond = tidy(() => this.conditioning(negativePrompt).concat(this.conditioning(prompt), 0));

      const C = this.unet.cfg.in_channels;
      let latents = tidy(() =>
        randomNormal([1, height / 8, width / 8, C], seed).mulScalar(this.sampler.priorScale()));

      const schedule = this.sampler.timesteps(steps);
      for (let i = 0; i < schedule.length; i++) {
        const [t, tPrev] = schedule[i];
        latents = tidy(() => {
          const both = latents.concat(latents, 0);
          const eps = this.unet.forward(both, t, cond);
          const [B, H, W, K] = eps.shape;
          const uncond = eps.slice([0, 0, 0, 0], [1, H, W, K]);
          const text = eps.slice([1, 0, 0, 0], [B, H, W, K]);
          const guided = uncond.add(text.sub(uncond).mulScalar(guidance));
          return this.sampler.step(guided, latents, t, tPrev);
        });
        // MLX is lazy: without forcing the step, the loop builds one graph
        // holding every UNet pass and evaluates them all at the end — the
        // memory blows up and the progress meter reports a fictional rate.
        evalAll(latents);
        onStep?.(i + 1, schedule.length);
      }

      // The VAE returns roughly [-1, 1]; images want [0, 1].
      const img = tidy(() => this.vae.decode(latents).divScalar(2).addScalar(0.5));
      latents.free();
      cond.free();
      const out = img.reshape([height, width, 3]);
      // MLX is lazy: without this the decode would run at the caller's first
      // read, after the cache cap is restored — which is where it first blew
      // past 5 GB of reuse buffers.
      evalAll(out);
      return out;
    } finally {
      setCacheLimit(prev);
    }
  }
}
