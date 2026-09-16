// Layer streaming on a real checkpoint. Qwen3-0.6B-4bit held in memory, streamed
// from its single file, and streamed from a two-shard copy must generate the
// same tokens, and the streamed runs must peak lower by close to the layers they
// no longer hold.
//
// Each run is its own process. MLX's peak is per process, and a resident model's
// weights wait for the GC rather than being released on demand, so running the
// three in one process would charge the streamed runs for resident weights
// still sitting there.
//
// Uses the cached weights and never downloads; without them it skips.
//   bun validation/qwen3-stream.ts
import { spawnSync } from "node:child_process";
import { closeSync, fstatSync, mkdtempSync, openSync, readSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  entries,
  freeMap,
  get,
  hubFile,
  isCached,
  layerStore,
  load,
  loadSafetensors,
  MX,
  peakMemoryMB,
  Qwen3,
  resetPeakMemory,
  saveSafetensors,
  streamTokens,
  Tokenizer,
} from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";

// Any cached Qwen3 small enough to also run resident; MLXTS_REPO overrides.
const REPO = process.env.MLXTS_REPO ?? "mlx-community/Qwen3-0.6B-4bit";
const PROMPT = "The capital of France is";
const MAX = 48;
const LAYER = /^model\.layers\.(\d+)\./;

// ---- child: one mode, one process ------------------------------------------
const mode = process.argv[2];
if (mode) {
  resetPeakMemory();
  let model: any;
  let tokenizer: Tokenizer;
  if (mode === "sharded") {
    model = new Qwen3(await readJson(await hubFile(REPO, "config.json")), layerStore(process.argv[3]));
    tokenizer = await Tokenizer.fromFile(await hubFile(REPO, "tokenizer.json"));
  } else {
    ({ model, tokenizer } = await load(REPO, { streamLayers: mode === "streamed" }));
  }
  const ids: number[] = [];
  const t0 = performance.now();
  for await (const { token } of streamTokens(model, tokenizer.encode(PROMPT), { max: MAX })) ids.push(token);
  const secs = (performance.now() - t0) / 1000;
  const peakMB = peakMemoryMB();
  model.close?.();
  console.log(`RESULT ${JSON.stringify({ mode, ids, tokPerSec: ids.length / secs, peakMB, text: tokenizer.decode(ids) })}`);
  process.exit(0);
}

// ---- parent ------------------------------------------------------------------
if (!(await isCached(REPO, "model.safetensors"))) {
  console.log(`qwen3-stream: skipped — ${REPO} is not cached`);
  process.exit(0);
}
const src = await hubFile(REPO, "model.safetensors");

/** Bytes per tensor, from the header alone. */
function tensorBytes(path: string): Map<string, number> {
  const fd = openSync(path, "r");
  try {
    const pre = new Uint8Array(8);
    readSync(fd, pre, 0, 8, 0);
    const n = Number(new DataView(pre.buffer).getBigUint64(0, true));
    if (n > fstatSync(fd).size) throw new Error(`${path}: bad header length`);
    const hdr = new Uint8Array(n);
    readSync(fd, hdr, 0, n, 8);
    const json = JSON.parse(new TextDecoder().decode(hdr)) as Record<string, { data_offsets: [number, number] }>;
    return new Map(
      Object.entries(json).filter(([k]) => k !== "__metadata__").map(([k, v]) => [k, v.data_offsets[1] - v.data_offsets[0]]),
    );
  } finally {
    closeSync(fd);
  }
}

// A two-shard copy with layers split across the boundary, so one run crosses
// from shard to shard mid-forward.
function makeShards(out: string): string {
  const w = loadSafetensors(src);
  const names = entries(w).map((e) => e.name);
  const layerOf = (n: string) => Number(n.match(LAYER)?.[1] ?? -1);
  const NL = Math.max(...names.map(layerOf)) + 1;
  const first = (n: string) => (layerOf(n) === -1 ? n.startsWith("model.embed_tokens") : layerOf(n) < NL / 2);
  const weightMap: Record<string, string> = {};
  for (const [file, pick] of [
    ["model-00001-of-00002.safetensors", first],
    ["model-00002-of-00002.safetensors", (n: string) => !first(n)],
  ] as const) {
    const rec: Record<string, MX> = {};
    for (const n of names.filter(pick)) { rec[n] = new MX(get(w, n)); weightMap[n] = file; }
    saveSafetensors(join(out, file), rec);
    for (const x of Object.values(rec)) x.free();
  }
  freeMap(w);
  const index = join(out, "model.safetensors.index.json");
  writeFileSync(index, JSON.stringify({ metadata: {}, weight_map: weightMap }));
  return index;
}

const self = fileURLToPath(import.meta.url);
function run(...args: string[]): { mode: string; ids: number[]; tokPerSec: number; peakMB: number; text: string } {
  const p = spawnSync(process.execPath, [self, ...args], { encoding: "utf8", maxBuffer: 64 << 20 });
  const line = (p.stdout ?? "").split("\n").find((l) => l.startsWith("RESULT "));
  if (p.status !== 0 || !line) {
    console.log(`FAIL ${args[0]}: exit ${p.status}\n${(p.stderr ?? "").slice(-3000)}`);
    process.exit(1);
  }
  return JSON.parse(line.slice("RESULT ".length));
}

const bytes = tensorBytes(src);
const NL = 1 + Math.max(...[...bytes.keys()].map((k) => Number(k.match(LAYER)?.[1] ?? -1)));
const layerMB = [...bytes].filter(([k]) => LAYER.test(k)).reduce((s, [, b]) => s + b, 0) / NL / 2 ** 20;
const sharedMB = [...bytes].filter(([k]) => !LAYER.test(k)).reduce((s, [, b]) => s + b, 0) / 2 ** 20;

const index = makeShards(mkdtempSync(join(tmpdir(), "mlx-ts-qwen3-shards-")));
const resident = run("resident");
const streamed = run("streamed");
const sharded = run("sharded", index);

console.log(`  ${REPO}: ${NL} layers of ${layerMB.toFixed(1)} MB each, ${sharedMB.toFixed(1)} MB shared`);
console.log(`  prompt ${JSON.stringify(PROMPT)}, greedy, ${MAX} tokens\n`);
for (const r of [resident, streamed, sharded]) {
  console.log(`  ${r.mode.padEnd(9)} peak ${r.peakMB.toFixed(0).padStart(4)} MB   ${r.tokPerSec.toFixed(1).padStart(6)} tok/s   ${r.ids.length} tokens`);
}
console.log(`\n  ${JSON.stringify(resident.text)}`);

const same = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => x === b[i]);
const failures: string[] = [];
if (resident.ids.length === 0) failures.push("the resident run generated nothing");
if (!same(streamed.ids, resident.ids)) failures.push(`MISMATCH streamed tokens differ from resident: ${streamed.ids} vs ${resident.ids}`);
if (!same(sharded.ids, resident.ids)) failures.push(`MISMATCH sharded tokens differ from resident: ${sharded.ids} vs ${resident.ids}`);

// Resident holds every layer; streamed holds one at a time. The saving should
// be close to the other NL - 1 layers — 80% of that, to leave room for
// allocator rounding, not so much that a store quietly holding several layers
// would still pass.
const expected = (NL - 1) * layerMB;
for (const r of [streamed, sharded]) {
  const saved = resident.peakMB - r.peakMB;
  if (saved < 0.8 * expected) {
    failures.push(`FAIL ${r.mode} peaks at ${r.peakMB.toFixed(0)} MB, saving ${saved.toFixed(0)} MB of an expected ~${expected.toFixed(0)} MB`);
  }
}

if (failures.length) {
  for (const f of failures) console.log(`  ${f}`);
  process.exit(1);
}
console.log(`\n  saved ${(resident.peakMB - Math.max(streamed.peakMB, sharded.peakMB)).toFixed(0)} MB of an expected ~${expected.toFixed(0)} MB`);
console.log(`qwen3-stream: ok (peak ${resident.peakMB.toFixed(0)} -> ${Math.max(streamed.peakMB, sharded.peakMB).toFixed(0)} MB)`);
