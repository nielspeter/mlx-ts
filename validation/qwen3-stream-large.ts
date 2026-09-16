// Layer streaming on a model that does not fit. Qwen3-32B-8bit is 32.4 GiB of
// weights; on a 36 GB Mac the GPU may use 28.1 GiB. Prints each token's time
// and MLX memory as it is generated.
//
// Manual, not part of validate-all.sh: it needs the ~35 GB checkpoint already
// cached, and it never downloads one. A resident comparison is deliberately not
// attempted — on a machine the model does not fit, that is the run that takes
// the system down. Exactness at this size therefore rests on qwen3-stream.ts,
// which compares streamed against resident on models small enough for both.
//
//   MLXTS_REPO=mlx-community/Qwen3-32B-8bit bun validation/qwen3-stream-large.ts ["prompt"] [max tokens]
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  activeMemoryMB,
  cacheMemoryMB,
  hubFile,
  isCached,
  load,
  peakMemoryMB,
  resetPeakMemory,
  streamTokens,
} from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";

const REPO = process.env.MLXTS_REPO ?? "mlx-community/Qwen3-32B-8bit";
const PROMPT = process.argv[2] ?? "The capital of France is";
const MAX = Number(process.argv[3] ?? 12);

// Every weight file must be cached — and, for a sharded checkpoint, beside its
// index, which is how load() opens it. Anything less and load() downloads.
async function fullyCached(repo: string): Promise<boolean> {
  if (await isCached(repo, "model.safetensors")) return true;
  if (!(await isCached(repo, "model.safetensors.index.json"))) return false;
  const index = await hubFile(repo, "model.safetensors.index.json");
  const { weight_map } = await readJson<{ weight_map: Record<string, string> }>(index);
  return Object.values(weight_map).every((shard) => existsSync(join(dirname(index), shard)));
}

if (!(await fullyCached(REPO))) {
  console.log(`qwen3-stream-large: skipped — ${REPO} is not fully cached, and this never downloads`);
  process.exit(0);
}

resetPeakMemory();
const tOpen = performance.now();
const { model, tokenizer, config } = await load(REPO, { streamLayers: true });
console.log(`${REPO}: ${config.num_hidden_layers} layers, ${config.quantization.bits}-bit, tie_word_embeddings=${config.tie_word_embeddings}`);
console.log(`store opened in ${((performance.now() - tOpen) / 1000).toFixed(2)} s (headers only; no weights read yet)`);
console.log(`prompt ${JSON.stringify(PROMPT)} (${tokenizer.encode(PROMPT).length} tokens), greedy, up to ${MAX} tokens\n`);

const out: number[] = [];
const secs: number[] = [];
let t = performance.now();
for await (const { token } of streamTokens(model, tokenizer.encode(PROMPT), { max: MAX })) {
  const now = performance.now();
  secs.push((now - t) / 1000);
  t = now;
  out.push(token);
  console.log(
    `  token ${String(out.length).padStart(2)}  ${secs[secs.length - 1].toFixed(2).padStart(6)} s   ` +
      `active ${activeMemoryMB().toFixed(0).padStart(5)} MB   peak ${peakMemoryMB().toFixed(0).padStart(5)} MB   ` +
      `cache ${cacheMemoryMB().toFixed(0).padStart(5)} MB   ${JSON.stringify(tokenizer.decode([token]))}`,
  );
}

const decode = secs.slice(1);
console.log(`\ncompletion: ${JSON.stringify(tokenizer.decode(out))}`);
console.log(`first token (prefill + 1): ${secs[0].toFixed(2)} s`);
if (decode.length) {
  const mean = decode.reduce((a, b) => a + b, 0) / decode.length;
  console.log(`decode: ${mean.toFixed(2)} s/token mean, ${Math.min(...decode).toFixed(2)}–${Math.max(...decode).toFixed(2)} s range`);
}
console.log(`MLX peak ${peakMemoryMB().toFixed(0)} MB, active at end ${activeMemoryMB().toFixed(0)} MB, cache ${cacheMemoryMB().toFixed(0)} MB`);
model.close?.();
