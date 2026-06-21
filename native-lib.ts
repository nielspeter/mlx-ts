// Resolve the mlx-c dylib path. Prefer a BUNDLED copy (the prebuilt platform
// package / `prebuilds/` dir) so the package runs without a Homebrew install;
// fall back to Homebrew for local dev. macOS/arm64 only. `MLXTS_LIB` overrides.
// The bundled set (libmlxc + libmlx + libjaccl + mlx.metallib) is made relocatable
// with @loader_path, so dlopen'ing libmlxc here pulls the rest from the same dir.
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const candidates = [
  process.env.MLXTS_LIB,
  join(here, "prebuilds", "darwin-arm64", "libmlxc.dylib"),   // bundled (repo / npm package)
  "/opt/homebrew/lib/libmlxc.dylib",                          // Homebrew dev fallback
].filter(Boolean) as string[];

export const LIBMLXC = candidates.find((p) => existsSync(p)) ?? candidates[candidates.length - 1];
