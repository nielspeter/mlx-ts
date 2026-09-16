// Fetch model files from the Hugging Face hub, with a local cache.
//
// The library could already load a model — from four files you had to curl by
// hand, into a directory it did not tell you about, under names that were repo
// conventions rather than anything guessable. This is the missing step between
// `npm i` and a token.
//
// Downloads land in mlx-ts's own cache, laid out by repo id so models coexist:
//   ~/.cache/mlx-ts/<org>/<name>/<file>              (MLXTS_CACHE overrides)
//   ~/.cache/mlx-ts/<org>/<name>@<revision>/<file>   (any revision but main)
//
// Before downloading, a file is also looked for in Hugging Face's own cache,
// where huggingface_hub, mlx-lm and other Python tools put what they fetch —
// a 35 GB checkpoint should not be downloaded twice because two libraries each
// keep a cache. That lookup only reads. mlx-ts never writes there: the layout
// (content-addressed blobs behind per-commit symlinks) belongs to those tools.

import { existsSync, readFileSync } from "node:fs";
import { mkdir, open, rename, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const HUB = "https://huggingface.co";
const COMMIT = /^[0-9a-f]{40}$/;

export const cacheDir = (): string =>
  process.env.MLXTS_CACHE ?? join(homedir(), ".cache", "mlx-ts");

export type FetchOptions = {
  /** Branch, tag or commit. Defaults to "main". */
  revision?: string;
  /** Called with (bytesDone, bytesTotal) as the download proceeds. */
  onProgress?: (done: number, total: number) => void;
  /** For gated or private repos. Defaults to $HF_TOKEN. */
  token?: string;
  /**
   * Use a file already in Hugging Face's cache rather than downloading it
   * again. Read-only: nothing is ever written there. Defaults to true.
   */
  sharedCache?: boolean;
};

const expandHome = (p: string) => (p === "~" || p.startsWith("~/") ? join(homedir(), p.slice(1)) : p);

/**
 * Hugging Face's cache directory, with huggingface_hub's own precedence:
 * HF_HUB_CACHE, then the older HUGGINGFACE_HUB_CACHE, then HF_HOME/hub, then
 * $XDG_CACHE_HOME/huggingface/hub, then ~/.cache/huggingface/hub.
 */
function sharedCacheDir(): string {
  const env = process.env;
  const explicit = env.HF_HUB_CACHE || env.HUGGINGFACE_HUB_CACHE;
  if (explicit) return expandHome(explicit);
  const cache = env.XDG_CACHE_HOME ? expandHome(env.XDG_CACHE_HOME) : join(homedir(), ".cache");
  return join(env.HF_HOME ? expandHome(env.HF_HOME) : join(cache, "huggingface"), "hub");
}

/**
 * mlx-ts's own path for a file. `main` keeps the flat layout every existing
 * cache already uses. Any other revision gets a directory of its own: a
 * branch, tag or commit can hold different bytes under the same file name,
 * and one flat path for all of them served whichever was fetched first.
 */
function ownPath(repo: string, file: string, revision: string): string {
  if (revision === "main") return join(cacheDir(), repo, file);
  return join(cacheDir(), `${repo}@${encodeURIComponent(revision)}`, file);
}

/**
 * The file in Hugging Face's cache, or null.
 *
 * huggingface_hub keeps a repo as models--<org>--<name>/. refs/<revision> holds
 * the commit a branch or tag resolved to; snapshots/<commit>/<file> is a symlink
 * to a content-addressed blob. It links a file only once its download is
 * complete, and a link whose blob was deleted does not resolve — so a snapshot
 * path that exists is a whole file.
 */
function sharedPath(repo: string, file: string, revision: string): string | null {
  const dir = join(sharedCacheDir(), `models--${repo.split("/").join("--")}`);
  try {
    const commit = COMMIT.test(revision)
      ? revision
      : readFileSync(join(dir, "refs", revision), "utf8").trim();
    if (!COMMIT.test(commit)) return null;
    const p = join(dir, "snapshots", commit, file);
    return existsSync(p) ? p : null;
  } catch {
    return null; // no such ref: this revision was never fetched there
  }
}

/** Where a file already is — our cache first, then Hugging Face's — or null. */
function cachedPath(repo: string, file: string, opts: FetchOptions): string | null {
  const revision = opts.revision ?? "main";
  const own = ownPath(repo, file, revision);
  if (existsSync(own)) return own;
  return opts.sharedCache === false ? null : sharedPath(repo, file, revision);
}

/**
 * Resolve one file from a repo to a local path, downloading it if absent.
 * Returns the cached path; a second call is a no-op.
 */
export async function hubFile(repo: string, file: string, opts: FetchOptions = {}): Promise<string> {
  const found = cachedPath(repo, file, opts);
  if (found) return found;

  const rev = opts.revision ?? "main";
  const dest = ownPath(repo, file, rev);
  await mkdir(dirname(dest), { recursive: true });
  // Download to a temp name and rename, so an interrupted fetch never leaves a
  // truncated file that the next run would treat as cached.
  const tmp = `${dest}.part`;

  // ...and resume into it if one is already there. A MusicGen medium checkpoint
  // is ~7 GB; losing all of it to one interrupted run is not acceptable.
  let from = existsSync(tmp) ? (await stat(tmp)).size : 0;

  // Encoded: a pull-request ref is `refs/pr/1`, and its slashes are part of the
  // revision, not of the file path.
  const url = `${HUB}/${repo}/resolve/${encodeURIComponent(rev)}/${file}`;
  const token = opts.token ?? process.env.HF_TOKEN;
  const res = await fetch(url, {
    headers: {
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...(from ? { range: `bytes=${from}-` } : {}),
    },
  });
  if (!res.ok) throw new Error(`hub: ${res.status} ${res.statusText} for ${repo}/${file}`);
  // 206 means the range was honoured. Anything else is the whole file again,
  // so the partial has to be discarded rather than appended to.
  if (from && res.status !== 206) from = 0;

  const total = from + Number(res.headers.get("content-length") ?? 0);

  // Streamed to disk a chunk at a time, never buffered whole: holding 7 GB in
  // JS memory (twice, while reassembling the chunks) runs a machine out of RAM.
  if (res.body) {
    const fh = await open(tmp, from ? "r+" : "w");
    try {
      let done = from;
      for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
        await fh.write(chunk, 0, chunk.length, done);
        done += chunk.length;
        opts.onProgress?.(done, total);
      }
    } finally {
      await fh.close();
    }
  } else {
    await writeFile(tmp, new Uint8Array(await res.arrayBuffer()));
  }

  await rename(tmp, dest);
  return dest;
}

/** True if the file is already in either cache — no network. */
export async function isCached(repo: string, file: string, opts: FetchOptions = {}): Promise<boolean> {
  return cachedPath(repo, file, opts) !== null;
}
