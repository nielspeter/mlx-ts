#!/usr/bin/env bash
# Assemble prebuilds/darwin-arm64/ from Apple's OWN binaries.
#
# Why not build libmlx ourselves: we tried, and the result disagreed with
# MLX-Python on real models (FINDINGS §7e). The builds were simply different —
# ours 15.5 MB against Apple's 21.8 MB. Rather than diff compiler flags, take
# the artifact Apple ships and that the parity oracle already runs against.
#
# Apple distributes it as a Python wheel, `mlx-metal`, pulled in by `mlx` via a
# platform marker (FINDINGS §7f). It carries libmlx + libjaccl + mlx.metallib
# but NOT libmlxc — mlx-c is a separate project — so that one still comes from
# Homebrew and gets relinked to sit next to Apple's libmlx.
#
#   bash tools/fetch-prebuilds.sh
#
# Caveat, stated rather than hidden: the libmlxc copied here was built against
# Homebrew's libmlx. It works because both are the same mlx version and the C
# API is ABI-stable across the patch (verified below by a parity check), but the
# rigorous version builds mlx-c from source against the wheel's libmlx.
set -euo pipefail
cd "$(dirname "$0")/.."

MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
ARCH=$(uname -m)
[ "$ARCH" = "arm64" ] || { echo "Apple Silicon only (got $ARCH)"; exit 1; }

# Apple ships one wheel per macOS target; pick the newest that is <= ours.
PY_JSON=$(curl -fsSL https://pypi.org/pypi/mlx-metal/json)
read -r VERSION URL <<EOF
$(printf '%s' "$PY_JSON" | python3 -c "
import json,sys,re
d=json.load(sys.stdin); v=d['info']['version']
ours=$MACOS_MAJOR
cands=[]
for f in d['releases'][v]:
    m=re.search(r'macosx_(\d+)_0_arm64', f['filename'])
    if m and int(m.group(1))<=ours: cands.append((int(m.group(1)), f['url']))
if not cands: sys.exit('no mlx-metal wheel for macOS %d' % ours)
print(v, max(cands)[1])
")
EOF
echo "mlx-metal $VERSION for macOS <= $MACOS_MAJOR"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$URL" -o "$TMP/w.whl"
(cd "$TMP" && unzip -q w.whl)

OUT=prebuilds/darwin-arm64
mkdir -p "$OUT"
for f in libmlx.dylib libjaccl.dylib mlx.metallib; do
  rm -f "$OUT/$f"                     # the wheel ships mlx.metallib read-only
  cp "$TMP/mlx/lib/$f" "$OUT/$f"
  chmod u+w "$OUT/$f"
  printf '  %-16s %s\n' "$f" "$(du -h "$OUT/$f" | cut -f1)"
done

# Apple's libmlx references @rpath/libjaccl.dylib but carries no LC_RPATH — in
# the wheel, the Python extension module supplies it. Add @loader_path so the
# bundle resolves its own siblings.
for f in libmlx.dylib libjaccl.dylib; do
  install_name_tool -add_rpath "@loader_path" "$OUT/$f" 2>/dev/null || true
  codesign -f -s - "$OUT/$f" >/dev/null 2>&1
done

# libmlxc is not in the wheel; take Homebrew's and repoint it at Apple's libmlx.
SRC=$(ls /opt/homebrew/opt/mlx-c/lib/libmlxc.dylib 2>/dev/null || true)
[ -n "$SRC" ] || { echo "need mlx-c for libmlxc: brew install mlx-c"; exit 1; }
rm -f "$OUT/libmlxc.dylib"
cp "$SRC" "$OUT/libmlxc.dylib"
chmod u+w "$OUT/libmlxc.dylib"
OLD=$(otool -L "$OUT/libmlxc.dylib" | awk '/libmlx\.dylib/ {print $1; exit}')
install_name_tool -change "$OLD" "@loader_path/libmlx.dylib" "$OUT/libmlxc.dylib"
codesign -f -s - "$OUT/libmlxc.dylib" >/dev/null 2>&1
printf '  %-16s %s (relinked -> @loader_path/libmlx.dylib)\n' libmlxc.dylib "$(du -h "$OUT/libmlxc.dylib" | cut -f1)"

# Prove it. Note which check: the synthetic block fingerprint PASSED with the
# old, wrong bundle — it is not sensitive enough. The divergence only showed on
# a real model, so verify with Qwen3 whenever the weights are present and treat
# the block check as a smoke test, not proof.
echo
gen() { grep "gen ids" | grep -oE '\[[-0-9, ]+\]'; }
BUNDLE="$PWD/$OUT/libmlxc.dylib"

if [ -f models/model-qwen.safetensors ] && [ -f models/config.json ]; then
  HB=$(MLXTS_LIB=/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib bun src/models/qwen.ts "The capital of France is" | gen)
  BD=$(MLXTS_QUIET=1 MLXTS_LIB="$BUNDLE" bun src/models/qwen.ts "The capital of France is" | gen)
  WHAT="real Qwen3-0.6B token ids"
else
  echo "  (models/model-qwen.safetensors absent — falling back to the block"
  echo "   fingerprint, which did NOT catch the previous bad bundle)"
  HB=$(MLXTS_LIB=/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib bun validation/block-gen.ts | grep -oE 'sum *= *[-0-9.]+')
  BD=$(MLXTS_QUIET=1 MLXTS_LIB="$BUNDLE" bun validation/block-gen.ts | grep -oE 'sum *= *[-0-9.]+')
  WHAT="block fingerprint (weak)"
fi

echo "  check   : $WHAT"
echo "  homebrew: ${HB:0:70}"
echo "  bundled : ${BD:0:70}"
[ -n "$BD" ] && [ "$HB" = "$BD" ] && echo "  OK — the bundle matches the validated path" \
  || { echo "  MISMATCH — do not ship this bundle"; exit 1; }
