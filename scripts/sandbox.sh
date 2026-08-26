#!/usr/bin/env bash
# Run a command under a memory ceiling that actually holds on this platform.
#
# Three things had to be true before this worked, and the first version got all
# three wrong — it reported success while a run took the machine down:
#
#  1. `ulimit -v` is a no-op on darwin, and MLX's setMemoryLimit() is not a hard
#     cap either: with a 500 MB limit a run still reached 3265 MB, because that
#     setting governs when the allocator evicts, not whether it refuses.
#  2. RSS does not see MLX at all. Metal buffers are not counted in the process
#     resident size — measured: ps said 7538 MB while MLX's own accounting said
#     28 GB. A watchdog polling RSS is blind to exactly the memory that matters.
#  3. RSS of the direct child misses its children. `sandbox.sh bash suite.sh`
#     was watching a shell, while the bun processes underneath did the work.
#
# So the real signal is system-wide: poll how much memory the machine has left
# and kill the whole process group before it starts swapping. RSS is still
# checked, summed over the group, as a second trigger.
#
#   scripts/sandbox.sh [--mb 8000] <command...>
set -uo pipefail

PHYS_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
LIMIT_MB=8000
if [ "${1:-}" = "--mb" ]; then LIMIT_MB=$2; shift 2; fi
[ $# -gt 0 ] || { echo "usage: sandbox.sh [--mb N] <command...>" >&2; exit 2; }

# A ceiling near physical RAM is not a ceiling: the machine is already swapping
# by the time it trips. Nothing above 60% is worth pretending about.
MAX_MB=$(( PHYS_MB * 60 / 100 ))
if [ "$LIMIT_MB" -gt "$MAX_MB" ]; then
  echo "sandbox: --mb ${LIMIT_MB} is too close to ${PHYS_MB} MB of RAM; using ${MAX_MB}" >&2
  LIMIT_MB=$MAX_MB
fi

# Kill before the machine starts compressing and swapping, not after.
FLOOR_MB=$(( PHYS_MB * 15 / 100 ))
[ "$FLOOR_MB" -lt 3000 ] && FLOOR_MB=3000

# Pages the OS can hand out without evicting anything.
available_mb() {
  vm_stat | awk '
    /page size of/       { for (i=1;i<=NF;i++) if ($i=="of") { ps=$(i+1); break } }
    /Pages free/         { f=$3 }
    /Pages inactive/     { n=$3 }
    /Pages speculative/  { s=$3 }
    /Pages purgeable/    { p=$3 }
    END { gsub(/\./,"",f); gsub(/\./,"",n); gsub(/\./,"",s); gsub(/\./,"",p)
          printf "%d", (f+n+s+p) * ps / 1048576 }'
}

export MLXTS_MEM_LIMIT="$LIMIT_MB"
set -m                                    # give the child its own process group
"$@" &
PID=$!
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')

stop() {
  echo "" >&2
  echo "sandbox: $1 — killing process group $PGID" >&2
  kill -9 "-$PGID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  exit 137
}

while kill -0 "$PID" 2>/dev/null; do
  AVAIL=$(available_mb)
  if [ -n "$AVAIL" ] && [ "$AVAIL" -lt "$FLOOR_MB" ]; then
    stop "system down to ${AVAIL} MB free, under the ${FLOOR_MB} MB floor"
  fi
  RSS_MB=$(ps -o rss=,pgid= -A 2>/dev/null | awk -v g="$PGID" '$2==g {s+=$1} END {printf "%d", s/1024}')
  if [ -n "$RSS_MB" ] && [ "$RSS_MB" -gt "$LIMIT_MB" ]; then
    stop "process group RSS ${RSS_MB} MB exceeded ${LIMIT_MB} MB"
  fi
  sleep 0.2
done
wait "$PID"
