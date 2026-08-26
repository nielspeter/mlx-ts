#!/usr/bin/env bash
# Assemble the platform package: @nielspeter/mlx-ts-darwin-arm64.
#
# The main package ships no binaries (96 KB) and needs `brew install mlx-c`.
# This is the alternative, and the shape is Apple's own — mlx is a small wheel
# that pulls a platform-gated mlx-metal carrying the native payload
# (FINDINGS §7f). In npm that is optionalDependencies.
#
#   bash tools/fetch-prebuilds.sh        # first: get Apple's binaries
#   bash tools/make-platform-package.sh  # then: wrap them as a package
#
# Output is platform/darwin-arm64/ (gitignored — it is ~190 MB).
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=prebuilds/darwin-arm64
OUT=platform/darwin-arm64
VERSION=$(node -p "require('./package.json').version")

[ -f "$SRC/libmlxc.dylib" ] || { echo "run tools/fetch-prebuilds.sh first"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
for f in libmlxc.dylib libmlx.dylib libjaccl.dylib mlx.metallib; do
  cp "$SRC/$f" "$OUT/$f"; chmod u+w "$OUT/$f"
done

cat > "$OUT/package.json" <<JSON
{
  "name": "@nielspeter/mlx-ts-darwin-arm64",
  "version": "$VERSION",
  "description": "Native MLX runtime for @nielspeter/mlx-ts (macOS, Apple Silicon).",
  "license": "MIT",
  "os": ["darwin"],
  "cpu": ["arm64"],
  "files": ["*.dylib", "*.metallib", "README.md", "NOTICE"]
}
JSON

cp NOTICE "$OUT/NOTICE"
cat > "$OUT/README.md" <<'MD'
# @nielspeter/mlx-ts-darwin-arm64

The native runtime for [`@nielspeter/mlx-ts`](https://www.npmjs.com/package/@nielspeter/mlx-ts).
Not useful on its own — install the main package, which pulls this in as an
`optionalDependency` on macOS/arm64.

`libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` are taken verbatim from
Apple's `mlx-metal` PyPI wheel (MIT, © 2023 Apple Inc.). `libmlxc.dylib` is
built from [mlx-c](https://github.com/ml-explore/mlx-c) and relinked to load
them from beside it. See `NOTICE`.
MD

echo "$OUT — $(du -sh "$OUT" | cut -f1)"
ls -la "$OUT" | awk 'NR>3 {printf "  %-18s %s\n", $9, $5}'

# It must work when it is the ONLY source: hide Homebrew by pointing directly at
# the packaged dylib, which is what the resolver will do on a machine without it.
echo
OUTAbs="$PWD/$OUT/libmlxc.dylib"
FP=$(MLXTS_LIB="$OUTAbs" bun validation/block-gen.ts | grep -oE 'sum *= *[-0-9.]+')
echo "  self-contained check: $FP"
[ "$FP" = "sum = 0.005793" ] || [ -n "$FP" ] && echo "  OK" || { echo "  FAILED"; exit 1; }
