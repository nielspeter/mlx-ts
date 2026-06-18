#!/usr/bin/env bash
# Run the TS benchmarks. With --py, also run each Python MLX counterpart right
# after, so the TS and reference timings sit adjacent for comparison.
#   bash benchmarks/run-all.sh          # TS only
#   bash benchmarks/run-all.sh --py     # TS + Python MLX reference
cd "$(dirname "$0")" || exit 1
PY=0; [ "$1" = "--py" ] && PY=1

benches=(single-ops rms-norm rope layer-norm batch-matmul sdpa-vector gather-qmm)
declare -A pyfile=(
  [single-ops]=single_ops.py [rms-norm]=rms_norm_bench.py [rope]=rope_bench.py
  [layer-norm]=layer_norm_bench.py [batch-matmul]=batch_matmul_bench.py
  [sdpa-vector]=sdpa_vector_bench.py [gather-qmm]=gather_qmm_bench.py
)

for b in "${benches[@]}"; do
  printf '\n\033[1m=== %s — mlx-ts (TypeScript / Bun FFI) ===\033[0m\n' "$b"
  bun "$b.ts"
  if [ $PY -eq 1 ]; then
    printf '\033[2m--- %s — MLX Python reference ---\033[0m\n' "$b"
    # single_ops/batch_matmul gate the GPU behind --gpu; the rest default to it.
    python3 "${pyfile[$b]}" --gpu 2>/dev/null || python3 "${pyfile[$b]}"
  fi
done
