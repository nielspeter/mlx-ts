// Brings the parity suite into the coverage figure.
//
// Most of this repo's verification lives in validation/, which scripts/
// validate-all.sh runs as separate processes — invisible to `bun test
// --coverage`, which only sees what the current process imports. Importing
// those scripts here executes them, so their coverage is recorded.
//
// Off by default: it is slow, and `bun test tests/` is on the suite's hot path.
// scripts/coverage.sh sets MLXTS_COVERAGE=1.
//
// Only the checks that need no model weights are here. The rest — Qwen, OLMoE,
// Whisper, MusicGen, Stable Diffusion — would load tens of GB into one process,
// so they are listed as excluded rather than quietly dropped.
import { test } from "bun:test";

const ENABLED = process.env.MLXTS_COVERAGE === "1";

/** Runs standalone with no downloaded weights; see scripts/coverage.sh. */
const WEIGHTLESS = [
  "conv2d", "groupnorm", "scheduler", "mlx", "block", "block-gen",
  "model-gen", "spike-train",
];

/** Excluded, with the reason, so the gap is stated rather than hidden. */
export const EXCLUDED: Record<string, string> = {
  "musicgen-e2e": "loads musicgen-small (~2.6 GB)",
  "musicgen-lm": "loads musicgen-small",
  "musicgen-mlx-layout": "network: reads safetensors headers from the hub",
  "unet-forward": "loads the SD UNet (~3.2 GB)",
  "vae-decode": "loads the SD VAE",
  "clip-encode": "loads CLIP ViT-L/14 (~1.6 GB)",
  "clip-tokenizer": "needs the CLIP vocab downloaded",
  "t5-encode": "loads T5",
  "encodec-decode": "loads EnCodec",
  "model-load": "needs a converted safetensors fixture",
  "model-quant": "needs a quantized fixture",
  "readme-snippets": "compile-only mirror, never executed",
  "spike-moe": "needs a MoE fixture",
  "spike-moe-layer": "needs a MoE fixture",
  "spike-throughput": "benchmark, needs real weights",
  "spike-nanogpt": "needs a tokenized corpus",
  "spike-microgpt": "needs /tmp/microgpt-init.f32 from reference/reference-microgpt.py",
};

for (const name of WEIGHTLESS) {
  test.skipIf(!ENABLED)(`coverage: validation/${name}.ts`, async () => {
    await import(`../validation/${name}.ts`);
  });
}

test.skipIf(!ENABLED)("coverage: excluded checks are named, not hidden", () => {
  const n = Object.keys(EXCLUDED).length;
  console.log(`\n  ${WEIGHTLESS.length} validation scripts measured, ${n} excluded:`);
  for (const [k, why] of Object.entries(EXCLUDED)) console.log(`    ${k.padEnd(22)} ${why}`);
});
