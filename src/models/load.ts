// load("mlx-community/Qwen3-0.6B-4bit") — the step between `npm i` and a token.
//
// Fetches config, tokenizer and weights from the hub (cached), dispatches on
// config.model_type, and hands back a model the generation API accepts.

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { readJson } from "../io/fs.ts";
import { type FetchOptions, hubFile } from "../io/hub.ts";
import { layerStore } from "../io/layer-store.ts";
import { loadSafetensors, shardedWeights, singleFileWeights, type Weights } from "../io/loader.ts";
import type { Decoder } from "../text/lm.ts";
import { Tokenizer } from "../text/tokenizer.ts";
import { OLMoE } from "./olmoe.ts";
import { Qwen3 } from "./qwen-nn.ts";

export type Loaded = { model: Decoder; tokenizer: Tokenizer; config: any };

export type LoadOptions = FetchOptions & {
  /**
   * Read decoder layers from disk one at a time instead of holding the whole
   * model: for checkpoints larger than memory, at the cost of a read per layer
   * per step. Same tokens either way. Qwen3 only so far; release the shared
   * tensors with `model.close()`.
   */
  streamLayers?: boolean;
};

const SHARD_INDEX = "model.safetensors.index.json";

// Weights are one file or many; the sharded loader mmaps each shard on first
// touch, so a large MoE never materialises on the heap. Exported for tests.
export async function fetchWeights(repo: string, opts: FetchOptions): Promise<{ single?: string; index?: string }> {
  // Single file first. Many mlx-community repos ship BOTH model.safetensors and
  // an index that names only that one file, so probing for the index first
  // misreads them as sharded.
  try {
    return { single: await hubFile(repo, "model.safetensors", opts) };
  } catch { /* genuinely sharded, fall through */ }

  let index: string;
  try {
    index = await hubFile(repo, SHARD_INDEX, opts);
  } catch {
    throw new Error(
      `${repo} has no safetensors weights — only PyTorch checkpoints (.bin), which ` +
      `cannot be read from TypeScript. Use a repo that publishes safetensors, or ` +
      `convert this one with reference/convert-to-safetensors.py.`,
    );
  }
  const shards = new Set(Object.values((await readJson<{ weight_map: Record<string, string> }>(index)).weight_map));
  // A sharded checkpoint is opened through its index, each shard looked up
  // beside it, so every file must sit in one directory. Hugging Face's cache can
  // hold an index whose shards never all finished downloading; opening it there
  // would fail on the missing one. Then the whole set comes into our own cache —
  // re-fetching any shard that was there, which is waste but not a wrong answer.
  if ([...shards].every((shard) => existsSync(join(dirname(index), shard)))) return { index };
  const own = { ...opts, sharedCache: false };
  const ownIndex = await hubFile(repo, SHARD_INDEX, own);
  for (const shard of shards) await hubFile(repo, shard, own);
  return { index: ownIndex };
}

/**
 * Load a model from a Hugging Face repo id.
 *
 * Supported today: 4-bit `qwen3` and `olmoe` checkpoints (the mlx-community
 * conversions). Unquantised and other architectures throw with the reason —
 * adding one is a forward pass plus a weight-key mapping, not new binding work.
 *
 * `{ streamLayers: true }` reads Qwen3's decoder layers from disk one at a time,
 * for checkpoints larger than memory; it also reads sharded Qwen3 checkpoints.
 */
export async function load(repo: string, opts: LoadOptions = {}): Promise<Loaded> {
  const config = await readJson<any>(await hubFile(repo, "config.json", opts));
  const tokenizer = await Tokenizer.fromFile(await hubFile(repo, "tokenizer.json", opts));
  const type = config.model_type;

  if (!config.quantization) {
    throw new Error(
      `load(${repo}): only quantized checkpoints are supported so far ` +
      `(config.quantization is absent). Try an mlx-community 4-bit conversion.`,
    );
  }

  const w = await fetchWeights(repo, opts);
  const weights = (): Weights => w.index ? shardedWeights(w.index) : singleFileWeights(w.single!);

  switch (type) {
    case "qwen3":
      if (opts.streamLayers) {
        return { model: new Qwen3(config, layerStore(w.index ?? w.single!)), tokenizer, config };
      }
      // Qwen3 takes the raw map handle rather than the accessor.
      if (w.index) {
        throw new Error(
          `load(${repo}): sharded qwen3 is not wired up for resident loading yet; ` +
          `{ streamLayers: true } reads it layer by layer`,
        );
      }
      return { model: new Qwen3(config, loadSafetensors(w.single!)), tokenizer, config };
    case "olmoe":
      if (opts.streamLayers) throw new Error(`load(${repo}): streamLayers supports qwen3 only so far`);
      return { model: new OLMoE(config, weights()), tokenizer, config };
    default:
      throw new Error(
        `load(${repo}): unsupported model_type "${type}". ` +
        `Supported: qwen3, olmoe. See src/models/ for the shape a new one takes.`,
      );
  }
}
