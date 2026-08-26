// load("mlx-community/Qwen3-0.6B-4bit") — the step between `npm i` and a token.
//
// Fetches config, tokenizer and weights from the hub (cached), dispatches on
// config.model_type, and hands back a model the generation API accepts.
import { hubFile, type FetchOptions } from "../io/hub.ts";
import { loadSafetensors, singleFileWeights, shardedWeights, type Weights } from "../io/loader.ts";
import { Tokenizer } from "../text/tokenizer.ts";
import type { Decoder } from "../text/lm.ts";
import { Qwen3 } from "./qwen-nn.ts";
import { OLMoE } from "./olmoe.ts";
import { readJson } from "../io/fs.ts";

export type Loaded = { model: Decoder; tokenizer: Tokenizer; config: any };

const SHARD_INDEX = "model.safetensors.index.json";

// Weights are one file or many; the sharded loader mmaps each shard on first
// touch, so a large MoE never materialises on the heap.
async function fetchWeights(repo: string, opts: FetchOptions): Promise<{ single?: string; index?: string }> {
  // Single file first. Many mlx-community repos ship BOTH model.safetensors and
  // an index that names only that one file, so probing for the index first
  // misreads them as sharded.
  try {
    return { single: await hubFile(repo, "model.safetensors", opts) };
  } catch { /* genuinely sharded, fall through */ }

  const index = await hubFile(repo, SHARD_INDEX, opts);
  const shards = new Set(Object.values((await readJson<{ weight_map: Record<string, string> }>(index)).weight_map));
  for (const shard of shards) await hubFile(repo, shard, opts);
  return { index };
}

/**
 * Load a model from a Hugging Face repo id.
 *
 * Supported today: 4-bit `qwen3` and `olmoe` checkpoints (the mlx-community
 * conversions). Unquantised and other architectures throw with the reason —
 * adding one is a forward pass plus a weight-key mapping, not new binding work.
 */
export async function load(repo: string, opts: FetchOptions = {}): Promise<Loaded> {
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
      // Qwen3 takes the raw map handle rather than the accessor.
      if (w.index) throw new Error(`load(${repo}): sharded qwen3 is not wired up yet`);
      return { model: new Qwen3(config, loadSafetensors(w.single!)), tokenizer, config };
    case "olmoe":
      return { model: new OLMoE(config, weights()), tokenizer, config };
    default:
      throw new Error(
        `load(${repo}): unsupported model_type "${type}". ` +
        `Supported: qwen3, olmoe. See src/models/ for the shape a new one takes.`,
      );
  }
}
