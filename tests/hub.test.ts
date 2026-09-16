// The hub cache, without touching the network.
//
// hubFile's download path needs the internet and belongs in the parity suite,
// but everything around it — where the cache lives, whether a file is already
// there, and the short-circuit that makes a second call free — is testable
// offline and is what decides whether a run downloads 3 GB again.
//
// That includes Hugging Face's own cache, which is read but never written: a
// file another tool already downloaded should be used where it is. These tests
// build that layout in temp directories and point HF_HUB_CACHE at one before
// anything runs, so the real ~/.cache/huggingface is never read. Any download
// attempt fails with its URL, so a test can tell "used the cache" from "went to
// the network" rather than hanging on a real request.
//   bun test tests/hub.test.ts
import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { cacheDir, hubFile, isCached } from "../src/io/hub.ts";
import { fetchWeights } from "../src/models/load.ts";

const root = mkdtempSync(join(tmpdir(), "mlx-ts-hub-"));
const hf = mkdtempSync(join(tmpdir(), "mlx-ts-hf-"));
process.env.MLXTS_CACHE = root;
const savedHubCache = process.env.HF_HUB_CACHE;
process.env.HF_HUB_CACHE = hf;

const realFetch = globalThis.fetch;
globalThis.fetch = (async (url: string | URL | Request) => {
  throw new Error(`network: ${String(url)}`);
}) as unknown as typeof fetch;

// Test files share one process: hand back what this one replaced.
afterAll(() => {
  globalThis.fetch = realFetch;
  if (savedHubCache === undefined) delete process.env.HF_HUB_CACHE;
  else process.env.HF_HUB_CACHE = savedHubCache;
});

const COMMIT = "a".repeat(40);
const OTHER = "b".repeat(40);

/** Put a file where hubFile would have downloaded it. */
function seed(repo: string, file: string, body: string): string {
  const p = join(root, repo, file);
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, body);
  return p;
}

/**
 * Lay a file out the way huggingface_hub does: refs/<revision> names a commit,
 * snapshots/<commit>/<file> links to a blob. `blob: false` leaves the link
 * dangling, as a deleted download does; `revision: ""` writes no ref, as a
 * snapshot fetched by commit has none.
 */
function seedShared(
  repo: string,
  file: string,
  body: string,
  { base = hf, revision = "main", commit = COMMIT, blob = true } = {},
): string {
  const dir = join(base, `models--${repo.split("/").join("--")}`);
  if (revision) {
    mkdirSync(dirname(join(dir, "refs", revision)), { recursive: true });
    writeFileSync(join(dir, "refs", revision), commit);
  }
  const blobPath = join(dir, "blobs", `${commit}-${file.replaceAll("/", "_")}`);
  mkdirSync(dirname(blobPath), { recursive: true });
  if (blob) writeFileSync(blobPath, body);
  const link = join(dir, "snapshots", commit, file);
  mkdirSync(dirname(link), { recursive: true });
  symlinkSync(relative(dirname(link), blobPath), link);
  return link;
}

// ---- mlx-ts's own cache ------------------------------------------------------

test("MLXTS_CACHE overrides the cache location", () => {
  expect(cacheDir()).toBe(root);
});

test("the cache path mirrors the repo id, so models coexist", () => {
  const p = seed("org/model", "config.json", "{}");
  expect(p).toBe(join(root, "org", "model", "config.json"));
});

test("isCached is false before and true after", async () => {
  expect(await isCached("org/model", "absent.json")).toBe(false);
  seed("org/model", "present.json", "{}");
  expect(await isCached("org/model", "present.json")).toBe(true);
});

test("hubFile returns a cached file without going to the network", async () => {
  seed("org/model", "cached.json", '{"ok":true}');
  const p = await hubFile("org/model", "cached.json");
  expect(p).toBe(join(root, "org", "model", "cached.json"));
});

test("a nested path inside a repo is cached at that path", async () => {
  seed("org/model", "unet/config.json", "{}");
  expect(await isCached("org/model", "unet/config.json")).toBe(true);
  expect(await hubFile("org/model", "unet/config.json")).toContain(join("unet", "config.json"));
});

test("a partial download is not mistaken for a cached file", async () => {
  seed("org/model", "big.safetensors.part", "half");
  expect(await isCached("org/model", "big.safetensors")).toBe(false);
});

// ---- revisions -----------------------------------------------------------------

test("a file cached for main is not served for another revision", async () => {
  // Before, the path ignored the revision: asking for v2 returned main's copy.
  seed("rev/model", "config.json", "main");
  expect(await isCached("rev/model", "config.json", { revision: "v2" })).toBe(false);
  await expect(hubFile("rev/model", "config.json", { revision: "v2" })).rejects.toThrow("resolve/v2/config.json");
});

test("a file cached for a revision is found again for that revision, and not for main", async () => {
  const p = seed("rev2/model@v2", "config.json", "v2");
  expect(await hubFile("rev2/model", "config.json", { revision: "v2" })).toBe(p);
  expect(await isCached("rev2/model", "config.json")).toBe(false);
});

test("a revision with slashes is one directory, and encoded in the download URL", async () => {
  await expect(hubFile("pr/model", "config.json", { revision: "refs/pr/1" })).rejects.toThrow(
    "resolve/refs%2Fpr%2F1/config.json",
  );
  expect(existsSync(join(root, "pr", "model@refs%2Fpr%2F1"))).toBe(true);
});

// ---- Hugging Face's cache ------------------------------------------------------

test("a file another tool already downloaded is used where it is", async () => {
  const link = seedShared("shared/model", "config.json", '{"from":"hf"}');
  expect(await hubFile("shared/model", "config.json")).toBe(link);
  expect(await isCached("shared/model", "config.json")).toBe(true);
});

test("our own cache is checked first when both hold the file", async () => {
  seedShared("both/model", "config.json", "hf");
  const own = seed("both/model", "config.json", "ours");
  expect(await hubFile("both/model", "config.json")).toBe(own);
});

test("a branch reads the snapshot its ref points at, not another one", async () => {
  seedShared("branch/model", "config.json", "main", { revision: "main", commit: COMMIT });
  const dev = seedShared("branch/model", "config.json", "dev", { revision: "dev", commit: OTHER });
  expect(await hubFile("branch/model", "config.json", { revision: "dev" })).toBe(dev);
});

test("a revision given as a commit reads that snapshot, with no ref needed", async () => {
  const link = seedShared("pinned/model", "config.json", "{}", { revision: "", commit: OTHER });
  expect(await hubFile("pinned/model", "config.json", { revision: OTHER })).toBe(link);
});

test("a link whose blob is gone is not a cached file", async () => {
  seedShared("dangling/model", "model.safetensors", "", { blob: false });
  expect(await isCached("dangling/model", "model.safetensors")).toBe(false);
  await expect(hubFile("dangling/model", "model.safetensors")).rejects.toThrow("network:");
});

test("a snapshot that lacks the file falls through to a download", async () => {
  seedShared("partial/model", "config.json", "{}");
  await expect(hubFile("partial/model", "model.safetensors")).rejects.toThrow("network:");
});

test("sharedCache: false ignores Hugging Face's cache", async () => {
  seedShared("optout/model", "config.json", "{}");
  expect(await isCached("optout/model", "config.json", { sharedCache: false })).toBe(false);
  await expect(hubFile("optout/model", "config.json", { sharedCache: false })).rejects.toThrow("network:");
});

test("the Hugging Face cache location follows huggingface_hub's precedence", async () => {
  const keys = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"] as const;
  const saved = Object.fromEntries(keys.map((k) => [k, process.env[k]]));
  const base = mkdtempSync(join(tmpdir(), "mlx-ts-hf-precedence-"));
  const dirs = {
    HF_HUB_CACHE: join(base, "hub-cache"),
    HUGGINGFACE_HUB_CACHE: join(base, "legacy-hub-cache"),
    HF_HOME: join(base, "hf-home"),
    XDG_CACHE_HOME: join(base, "xdg"),
  };
  const links = {
    HF_HUB_CACHE: seedShared("prec/model", "config.json", "1", { base: dirs.HF_HUB_CACHE }),
    HUGGINGFACE_HUB_CACHE: seedShared("prec/model", "config.json", "2", { base: dirs.HUGGINGFACE_HUB_CACHE }),
    HF_HOME: seedShared("prec/model", "config.json", "3", { base: join(dirs.HF_HOME, "hub") }),
    XDG_CACHE_HOME: seedShared("prec/model", "config.json", "4", {
      base: join(dirs.XDG_CACHE_HOME, "huggingface", "hub"),
    }),
  };
  try {
    for (const k of keys) process.env[k] = dirs[k];
    // Each wins over those after it: remove them in order and the next takes
    // over. The loop stops before the real ~/.cache/huggingface would be read.
    for (const k of keys) {
      expect(await hubFile("prec/model", "config.json")).toBe(links[k]);
      delete process.env[k];
    }
  } finally {
    for (const k of keys) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  }
});

// ---- sharded checkpoints across the two caches ------------------------------------

const shardIndex = (a: string, b: string) => JSON.stringify({ weight_map: { "x.weight": a, "y.weight": b } });

test("a sharded checkpoint complete in Hugging Face's cache is opened there", async () => {
  const index = seedShared("shards/model", "model.safetensors.index.json", shardIndex("s1.safetensors", "s2.safetensors"));
  seedShared("shards/model", "s1.safetensors", "1");
  seedShared("shards/model", "s2.safetensors", "2");
  expect(await fetchWeights("shards/model", {})).toEqual({ index });
});

test("a sharded checkpoint only partly in Hugging Face's cache is fetched whole into ours", async () => {
  seedShared("halfshards/model", "model.safetensors.index.json", shardIndex("s1.safetensors", "s2.safetensors"));
  seedShared("halfshards/model", "s1.safetensors", "1");
  // s2 never finished there. The loaders look for every shard beside the index,
  // so using that index would fail on s2 — the index itself has to come into
  // our cache, with the shards, and that download is what is attempted.
  await expect(fetchWeights("halfshards/model", {})).rejects.toThrow("resolve/main/model.safetensors.index.json");
});
