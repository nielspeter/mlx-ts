// Resolve the mlx-c dylib path. Prefer a BUNDLED copy (the prebuilt platform
// package / `prebuilds/` dir) so the package runs without a Homebrew install;
// fall back to Homebrew for local dev. macOS/arm64 only. `MLXTS_LIB` overrides.
// The bundled set (libmlxc + libmlx + libjaccl + mlx.metallib) is made relocatable
// with @loader_path, so dlopen'ing libmlxc here pulls the rest from the same dir.
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
// src/ffi -> repo root when running from source; dist/ffi -> package root
// when running from the published build. Try both.
const roots = [join(here, "..", ".."), join(here, "..", "..", "..")];
// Order matters, and not the way you would guess. A Homebrew mlx-c is the build
// the parity suite is run against; the bundled prebuilds/ copy is a *different*
// MLX build that does NOT agree with MLX-Python numerically (real Qwen3 and
// LoRA diverge). So the validated path wins whenever it is present, and the
// bundle is the fallback for machines without Homebrew — which is the case the
// bundle exists for. MLXTS_LIB overrides everything.
const bundles = roots.map((r) => join(r, "prebuilds", "darwin-arm64", "libmlxc.dylib"));
const BUNDLED = bundles.find(existsSync) ?? bundles[0];
const candidates = [
  process.env.MLXTS_LIB,
  "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib",                // Homebrew (Apple silicon)
  "/opt/homebrew/lib/libmlxc.dylib",
  "/usr/local/opt/mlx-c/lib/libmlxc.dylib",                   // Homebrew (Intel)
  "/usr/local/lib/libmlxc.dylib",
  BUNDLED,                                                    // bundled fallback — see above
].filter(Boolean) as string[];

const found = candidates.find((p) => existsSync(p));
if (!found) {
  console.error(
    "libmlxc.dylib not found. Install with `brew install mlx-c`, ship a copy in\n" +
    "prebuilds/darwin-arm64/, or point MLXTS_LIB at one. Tried:\n" +
    candidates.map((c) => `  ${c}`).join("\n"),
  );
}

// Say so out loud: numbers from the bundle are not the numbers the suite checks.
if (found === BUNDLED && !process.env.MLXTS_QUIET) {
  console.warn(
    `mlx-ts: using the bundled libmlxc (${BUNDLED}).\n` +
    "        It is a different MLX build from Homebrew's and is not covered by\n" +
    "        the parity suite. Install mlx-c (brew install mlx-c) for the\n" +
    "        validated path, or set MLXTS_QUIET=1 to silence this.",
  );
}

export const LIBMLXC = found ?? candidates[candidates.length - 1];
