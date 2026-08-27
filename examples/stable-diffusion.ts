// Text to image: Stable Diffusion, end to end, in TypeScript.
//
//   bun examples/stable-diffusion.ts "a photo of an astronaut riding a horse"
//   bun examples/stable-diffusion.ts "a watercolour fox" --size 512 --steps 30 --seed 7
//
// First run downloads ~5 GB (UNet, VAE, CLIP) into ~/.cache/mlx-ts. Every run
// after that is local.
//
// The pipeline: CLIP tokenizes the prompt and encodes it; the UNet denoises a
// latent for `--steps` Euler steps, guided each step by the gap between its
// conditional and unconditional predictions; the VAE turns the final latent
// into pixels. Each of those four matches mlx-examples numerically — see
// scripts/validate-all.sh.
import { savePng, StableDiffusion } from "../src/index.ts";

const argv = process.argv.slice(2);
const flag = (name: string, dflt: number) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? Number(argv[i + 1]) : dflt;
};
const str = (name: string, dflt: string) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const prompt = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")))
  .join(" ") || "a photo of an astronaut riding a horse";

const size = flag("size", 512);
const steps = flag("steps", 20);
const out = str("out", "out.png");
const seedArg = argv.includes("--seed") ? flag("seed", 0) : undefined;

console.log(`prompt: ${JSON.stringify(prompt)}  (${size}x${size}, ${steps} steps` +
            (seedArg === undefined ? "" : `, seed ${seedArg}`) + ")");

const t0 = performance.now();
const sd = await StableDiffusion.fromPretrained(str("model", "stable-diffusion-v1-5/stable-diffusion-v1-5"));
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s\n`);

const t1 = performance.now();
const img = sd.generate(prompt, {
  width: size, height: size, steps,
  guidance: flag("guidance", 7.5),
  negativePrompt: str("negative", ""),
  cacheLimitMB: flag("cache", 512),
  seed: seedArg,
  onStep: (i, n) => {
    const rate = i / ((performance.now() - t1) / 1000);
    process.stdout.write(`\r  ${String(Math.round((i / n) * 100)).padStart(3)}%  ${i}/${n} steps  ${rate.toFixed(2)} steps/s   `);
  },
});

await savePng(out, img.toF32(), size, size);
console.log(`\n\nwrote ${out} — ${size}x${size}`);
console.log(`total ${((performance.now() - t1) / 1000).toFixed(1)}s`);
