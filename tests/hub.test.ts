// The hub cache, without touching the network.
//
// hubFile's download path needs the internet and belongs in the parity suite,
// but everything around it — where the cache lives, whether a file is already
// there, and the short-circuit that makes a second call free — is testable
// offline and is what decides whether a run downloads 3 GB again.
//   bun test tests/hub.test.ts
import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { cacheDir, hubFile, isCached } from "../src/io/hub.ts";

const root = mkdtempSync(join(tmpdir(), "mlx-ts-hub-"));
process.env.MLXTS_CACHE = root;

/** Put a file where hubFile would have downloaded it. */
function seed(repo: string, file: string, body: string): string {
  const p = join(root, repo, file);
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, body);
  return p;
}

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
  // No fetch happens: a miss here would try to reach huggingface.co and fail.
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
