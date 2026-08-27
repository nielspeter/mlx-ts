#!/usr/bin/env bash
# Measure test coverage over src/, and fail if it drops.
#
# Most of this repo's verification lives in validation/, which validate-all.sh
# runs as separate processes — invisible to `bun test --coverage`, which only
# sees the current process. tests/coverage.test.ts imports the weightless ones
# so they count; measured, that roughly doubles the figure (20% -> 40% of
# functions). The model-loading checks stay out because they would pull tens of
# GB into one process, and they are named in that file rather than dropped.
#
# src/ffi/generated.ts is excluded in bunfig.toml: 1984 emitted lines whose
# unreached half says nothing about code anyone wrote.
#
# Thresholds are a ratchet, not a target. They sit just under the measured
# figure — 66% of functions and 74% of lines at the time of writing — so a real
# regression fails while noise does not. Raise them when coverage rises.
#
#   scripts/coverage.sh            # report and enforce
#   scripts/coverage.sh --report   # report only
set -uo pipefail
cd "$(dirname "$0")/.."

MIN_FUNCS=${MIN_FUNCS:-64}
MIN_LINES=${MIN_LINES:-72}
REPORT_ONLY=0
[ "${1:-}" = "--report" ] && REPORT_ONLY=1

OUT=$(MLXTS_COVERAGE=1 bun test tests/ --coverage 2>&1)
echo "$OUT" | grep -E "^ (File|-|src/|All files)" || true

SUMMARY=$(echo "$OUT" | grep "All files" | head -1)
FUNCS=$(echo "$SUMMARY" | awk -F'|' '{gsub(/ /,"",$2); print $2}')
LINES=$(echo "$SUMMARY" | awk -F'|' '{gsub(/ /,"",$3); print $3}')

if [ -z "$FUNCS" ] || [ -z "$LINES" ]; then
  echo "coverage: could not parse a summary from bun test" >&2
  echo "$OUT" | tail -20 >&2
  exit 1
fi

echo
echo "coverage: ${FUNCS}% of functions, ${LINES}% of lines  (floor ${MIN_FUNCS}% / ${MIN_LINES}%)"
[ "$REPORT_ONLY" = "1" ] && exit 0

FAIL=0
awk "BEGIN{exit !($FUNCS < $MIN_FUNCS)}" && { echo "coverage: functions ${FUNCS}% is below ${MIN_FUNCS}%" >&2; FAIL=1; }
awk "BEGIN{exit !($LINES < $MIN_LINES)}" && { echo "coverage: lines ${LINES}% is below ${MIN_LINES}%" >&2; FAIL=1; }
[ "$FAIL" = "1" ] && exit 1
echo "coverage: ok"
