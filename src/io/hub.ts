// Fetch model files from the Hugging Face hub, with a local cache.
//
// The library could already load a model — from four files you had to curl by
// hand, into a directory it did not tell you about, under names that were repo
// conventions rather than anything guessable. This is the missing step between
// `npm i` and a token.
//
// Cache layout mirrors the repo id so several models coexist:
//   ~/.cache/mlx-ts/<org>/<name>/<file>        (MLXTS_CACHE overrides)
import { mkdir, stat, rename, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

const HUB = "https://huggingface.co";

export const cacheDir = (): string =>
  process.env.MLXTS_CACHE ?? join(homedir(), ".cache", "mlx-ts");

export type FetchOptions = {
  /** Branch, tag or commit. Defaults to "main". */
  revision?: string;
  /** Called with (bytesDone, bytesTotal) as the download proceeds. */
  onProgress?: (done: number, total: number) => void;
  /** For gated or private repos. Defaults to $HF_TOKEN. */
  token?: string;
};

/**
 * Resolve one file from a repo to a local path, downloading it if absent.
 * Returns the cached path; a second call is a no-op.
 */
export async function hubFile(repo: string, file: string, opts: FetchOptions = {}): Promise<string> {
  const dest = join(cacheDir(), repo, file);
  if (existsSync(dest)) return dest;

  const rev = opts.revision ?? "main";
  const url = `${HUB}/${repo}/resolve/${rev}/${file}`;
  const token = opts.token ?? process.env.HF_TOKEN;
  const res = await fetch(url, { headers: token ? { authorization: `Bearer ${token}` } : {} });
  if (!res.ok) throw new Error(`hub: ${res.status} ${res.statusText} for ${repo}/${file}`);

  await mkdir(dirname(dest), { recursive: true });
  // Download to a temp name and rename, so an interrupted fetch never leaves a
  // truncated file that the next run would treat as cached.
  const tmp = `${dest}.part`;
  const total = Number(res.headers.get("content-length") ?? 0);

  if (opts.onProgress && res.body) {
    const chunks: Uint8Array[] = [];
    let done = 0;
    for await (const chunk of res.body as any as AsyncIterable<Uint8Array>) {
      chunks.push(chunk); done += chunk.length; opts.onProgress(done, total);
    }
    const buf = new Uint8Array(done);
    let off = 0; for (const c of chunks) { buf.set(c, off); off += c.length; }
    await writeFile(tmp, buf);
  } else {
    await writeFile(tmp, new Uint8Array(await res.arrayBuffer()));
  }
  await rename(tmp, dest);
  return dest;
}

/** True if the file is already cached — no network. */
export async function isCached(repo: string, file: string): Promise<boolean> {
  try { await stat(join(cacheDir(), repo, file)); return true; } catch { return false; }
}
