// Runtime-neutral file reads. `Bun.file(p).json()` is ergonomic but Bun-only,
// and the src/ tree has to import cleanly under Deno and Node too. node:fs/
// promises is the one file API all three implement, so everything here is a
// thin wrapper over it — no per-runtime branching needed.
import { readFile, writeFile } from "node:fs/promises";

export const readText = (path: string): Promise<string> => readFile(path, "utf8");

export const readJson = async <T = any>(path: string): Promise<T> =>
  JSON.parse(await readFile(path, "utf8")) as T;

export const readBytes = async (path: string): Promise<Uint8Array> =>
  new Uint8Array((await readFile(path)).buffer as ArrayBuffer);

export const writeText = (path: string, data: string): Promise<void> =>
  writeFile(path, data, "utf8");

export const writeJson = (path: string, value: unknown, indent = 2): Promise<void> =>
  writeFile(path, JSON.stringify(value, null, indent), "utf8");
