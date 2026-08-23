# Benchmarks — mlx-ts vs MLX Python

TypeScript ports of the MLX micro-benchmarks (`mlx/benchmarks/python`), running
the same ops over `mlx-c` via Bun FFI. Same protocol as the originals
(`time_utils.py`): **5 warmup iters, then 100 timed iters of `eval(fn(args))`**,
reported in msec. Per-iteration intermediates are freed with `tidy()` (a tight
sync loop never triggers GC — see `../docs/FINDINGS.md` §6.5).

The point: confirm at the **op level** what `../docs/FINDINGS.md` §5/§7b showed at the
model level — TS is compute-bound, not FFI-bound, so it tracks MLX Python closely.

## Run

```sh
bash benchmarks/run-all.sh          # TS only
bash benchmarks/run-all.sh --py     # TS + the Python MLX reference, adjacent
bun benchmarks/single-ops.ts        # one bench
python3 benchmarks/single_ops.py --gpu   # its reference
```

Each `*.ts` has its Python original alongside it (`*_bench.py` / `single_ops.py`),
copied here so the comparison is self-contained — same pattern as the repo's
`reference-*.py` mirrors.

## Coverage

| TS bench | Python original | Op |
|---|---|---|
| `single-ops.ts` | `single_ops.py` | add / matmul / max / exp / negative / logsumexp / take / reshape |
| `batch-matmul.ts` | `batch_matmul_bench.py` | batched + unbatched matmul |
| `rms-norm.ts` | `rms_norm_bench.py` | manual vs fused `fast.rms_norm` |
| `layer-norm.ts` | `layer_norm_bench.py` | manual vs fused `fast.layer_norm` (dtypes × sizes) |
| `rope.ts` | `rope_bench.py` | `fast.rope`, vec + matrix |
| `sdpa-vector.ts` | `sdpa_vector_bench.py` | `fast.sdpa`, GQA single-query decode |
| `gather-qmm.ts` | `gather_qmm_bench.py` | `gather_qmm` MoE dispatch + dense `quantized_matmul` |

## Memory: why looped benches use `loop()`, not a bare `for`

`tidy()` adds **every** `MX` created in its scope to a set it frees only at scope
exit. That's right for the decode loop (shallow, per-step). But wrapping a *deep*
loop (e.g. 32× rms_norm/layer_norm) in one `tidy()` pins all 32 iterations'
intermediates at once — the live handles stop MLX from stream-freeing during
`eval`, so peak memory becomes O(depth × tensor-size). For float32 `[8,1024,8192]`
that's ~5 intermediates × 256 MiB × 32 ≈ **40 GiB → OOM**. Python avoids this for
free: refcounting drops each superseded array immediately, so `eval` streams.

The `loop(x, n, step)` helper restores that: each `step` runs under its own
`tidy()` and the prior output is freed once the next is built, so peak stays at
~one iteration. `timeFn`/`loop` also call a `guardMem()` check that **aborts if
active memory exceeds 6 GB** — a harness bug can no longer OOM the machine.

## Scope

**Forward-pass only.** The Python benches also time `mx.grad` / `mx.compile` /
`mx.vjp` / `mx.vmap` variants; those aren't part of this SDK's surface (training
uses the closure-based `value_and_grad` in `../src/nn/autograd.ts`; `vmap` is the one genuine
mlx-c gap — `../docs/FINDINGS.md` §6/§7). The forward kernels are what govern inference,
which is the comparison that matters here.

`mlx-ts` always runs on the default stream (Metal/GPU on Apple silicon). For the
two Python benches that gate the GPU behind a flag (`single_ops`, `batch_matmul`),
pass `--gpu`; the runner does this automatically.
