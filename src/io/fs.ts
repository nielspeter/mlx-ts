// Runtime-neutral file reads. `Bun.file(p).json()` is ergonomic but Bun-only,
// and the src/ tree has to import cleanly under Deno and Node too. node:fs/
// promises is the one file API all three implement, so everything here is a
// thin wrapper over it — no per-runtime branching needed.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

const ensureDir = (p: string) => mkdir(dirname(p), { recursive: true });

export const readText = (path: string | URL): Promise<string> => readFile(path, "utf8");

export const readJson = async <T = any>(path: string | URL): Promise<T> =>
  JSON.parse(await readFile(path, "utf8")) as T;

export const readBytes = async (path: string | URL): Promise<Uint8Array> =>
  new Uint8Array((await readFile(path)).buffer as ArrayBuffer);

export const writeText = async (path: string, data: string): Promise<void> => {
  await ensureDir(path);
  return writeFile(path, data, "utf8");
};

export const writeJson = async (path: string, value: unknown, indent = 2): Promise<void> => {
  await ensureDir(path);
  return writeFile(path, JSON.stringify(value, null, indent), "utf8");
};
