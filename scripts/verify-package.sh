#!/usr/bin/env bash
# Pack the package, install it into a throwaway project, and use it from Bun,
# Deno and Node. This is the only thing that catches distribution bugs — the
# repo's own tests all import via relative paths and never see what a consumer
# sees. It is how we found that Node refuses to type-strip inside node_modules,
# which made shipping raw .ts unusable there.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

TARBALL="$ROOT/$(npm pack --silent | tail -1)"
trap 'rm -rf "$TMP" "$TARBALL"' EXIT

cd "$TMP"
printf '{ "name": "pkg-check", "private": true, "type": "module" }\n' > package.json
printf '{ "nodeModulesDir": "manual" }\n' > deno.json
npm install --silent "$TARBALL" >/dev/null 2>&1

cat > use.ts <<'TS'
import { fromF32, tidy, Module, Linear, backend, type MX } from "@npstrandberg/mlx-ts";
const a = fromF32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const b = fromF32(new Float32Array([0.5, -1, 2, 0.25, -0.75, 1.5]), [3, 2]);
const sum = a.matmul(b).toF32().reduce((s, v) => s + v, 0);
class Net extends Module {
  l: Linear;
  constructor() { super(); this.l = new Linear(fromF32(new Float32Array(6).fill(0.5), [3, 2])); }
  forward(x: MX): MX { return this.l.forward(x); }
}
const shape = tidy(() => new Net().call(a)).shape.join("x");
if (Math.abs(sum - 20) > 1e-4 || shape !== "2x2") { console.error(`BAD sum=${sum} shape=${shape}`); process.exit(1); }
console.log(`${backend.name}: ok`);
TS

FAIL=0
for RT in bun "deno run --allow-all" node; do
  command -v "${RT%% *}" >/dev/null 2>&1 || continue
  if OUT=$($RT use.ts 2>&1 | tail -1); then echo "  $OUT"; else echo "  ${RT%% *}: FAILED — $OUT"; FAIL=1; fi
done
exit $FAIL
