// Resolve the mlx-c dylib path. Prefer a BUNDLED copy (the prebuilt platform
// package / `prebuilds/` dir) so the package runs without a Homebrew install;
// fall back to Homebrew for local dev. macOS/arm64 only. `MLXTS_LIB` overrides.
// The bundled set (libmlxc + libmlx + libjaccl + mlx.metallib) is made relocatable
// with @loader_path, so dlopen'ing libmlxc here pulls the rest from the same dir.
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");   // src/ffi -> repo root
const candidates = [
  process.env.MLXTS_LIB,
  join(root, "prebuilds", "darwin-arm64", "libmlxc.dylib"),   // bundled (repo / npm package)
  "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib",                // Homebrew dev fallback (Apple silicon)
  "/opt/homebrew/lib/libmlxc.dylib",
  "/usr/local/opt/mlx-c/lib/libmlxc.dylib",                   // Homebrew dev fallback (Intel)
  "/usr/local/lib/libmlxc.dylib",
].filter(Boolean) as string[];

const found = candidates.find((p) => existsSync(p));
if (!found) {
  console.error(
    "libmlxc.dylib not found. Install with `brew install mlx-c`, ship a copy in\n" +
    "prebuilds/darwin-arm64/, or point MLXTS_LIB at one. Tried:\n" +
    candidates.map((c) => `  ${c}`).join("\n"),
  );
}

export const LIBMLXC = found ?? candidates[candidates.length - 1];
