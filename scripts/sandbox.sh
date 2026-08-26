#!/usr/bin/env bash
# Run a command under a hard memory ceiling.
#
# `ulimit -v` is a no-op on darwin, and MLX's setMemoryLimit() is NOT a hard
# cap — measured: with a 500 MB limit it still grew to 3265 MB, because that
# setting governs when the allocator evicts its cache, not whether an allocation
# is refused. So the only real lever is external: poll RSS and SIGKILL.
#
# This exists because a leaked KV cache once grew to 55 GB on a 39 GB machine
# and took the box down. A ceiling that stops the run is always better than
# swapping.
#
#   scripts/sandbox.sh [--mb 8000] <command...>
set -uo pipefail
LIMIT_MB=8000
if [ "${1:-}" = "--mb" ]; then LIMIT_MB=$2; shift 2; fi
[ $# -gt 0 ] || { echo "usage: sandbox.sh [--mb N] <command...>" >&2; exit 2; }

export MLXTS_MEM_LIMIT="$LIMIT_MB"
"$@" &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
  RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt $((LIMIT_MB * 1024)) ]; then
    echo "" >&2
    echo "sandbox: RSS $((RSS_KB / 1024)) MB exceeded ${LIMIT_MB} MB — killing $PID" >&2
    kill -9 "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
    exit 137
  fi
  sleep 0.2
done
wait "$PID"
