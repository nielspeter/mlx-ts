// Resolve the mlx-c dylib. macOS/arm64 only. `MLXTS_LIB` overrides everything.
//
// Three places it can come from, in order: a Homebrew install, the platform
// package (@nielspeter/mlx-ts-darwin-arm64, an optionalDependency), or a local
// prebuilds/ dir. The bundled sets are relocatable via @loader_path, so
// dlopen'ing libmlxc pulls libmlx / libjaccl / mlx.metallib from beside it.
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
// src/ffi -> repo root when running from source; dist/ffi -> package root
// when running from the published build. Try both.
const roots = [join(here, "..", ".."), join(here, "..", "..", "..")];
// Homebrew first, bundle second. Both are validated now — prebuilds/ is
// assembled from Apple's own mlx-metal binaries by tools/fetch-prebuilds.sh and
// reproduces MLX-Python token for token — but a Homebrew install is the one a
// developer can upgrade and inspect, so it wins when present. The bundle is for
// machines without it, which is what it exists for. MLXTS_LIB overrides both.
const bundles = roots.map((r) => join(r, "prebuilds", "darwin-arm64", "libmlxc.dylib"));
const BUNDLED = bundles.find(existsSync) ?? bundles[0];

// The platform package, when installed. It is an optionalDependency, so absence
// is the normal case rather than an error.
function platformPackage(): string | undefined {
  try {
    const req = createRequire(import.meta.url);
    return join(dirname(req.resolve("@nielspeter/mlx-ts-darwin-arm64/package.json")), "libmlxc.dylib");
  } catch { return undefined; }
}

const candidates = [
  process.env.MLXTS_LIB,
  "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib",                // Homebrew (Apple silicon)
  "/opt/homebrew/lib/libmlxc.dylib",
  "/usr/local/opt/mlx-c/lib/libmlxc.dylib",                   // Homebrew (Intel)
  "/usr/local/lib/libmlxc.dylib",
  platformPackage(),                                          // npm, no Homebrew needed
  BUNDLED,                                                    // a local prebuilds/ dir
].filter(Boolean) as string[];

/**
 * Every path considered, in order, and whether it exists. Exported because
 * "which libmlxc did it pick, and why" is the first question when numbers or
 * load errors look wrong — and on a developer machine several are present.
 */
export const LIB_CANDIDATES: ReadonlyArray<{ path: string; exists: boolean }> =
  candidates.map((path) => ({ path, exists: existsSync(path) }));

const found = candidates.find((p) => existsSync(p));
if (!found) {
  console.error(
    "libmlxc.dylib not found. Install with `brew install mlx-c`, ship a copy in\n" +
    "prebuilds/darwin-arm64/, or point MLXTS_LIB at one. Tried:\n" +
    candidates.map((c) => `  ${c}`).join("\n"),
  );
}

export const LIBMLXC = found ?? candidates[candidates.length - 1];
