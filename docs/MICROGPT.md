# microGPT in mlx-ts — a transformer trained from scratch, TS driving real MLX

A spike that reproduces Andrej Karpathy's **microGPT** — but as the philosophical
mirror-image. His version hand-rolls a `Value` autograd engine in ~200 lines of
pure Python to show *"I cannot simplify this any further… everything else is just
efficiency."* This one builds the **same tiny GPT** and trains it end-to-end with
the autograd being **real Apple MLX, driven from TypeScript over `mlx-c` via Bun
FFI** — i.e. the "just efficiency" path he sets aside, proven fully drivable from
TS, validated against an MLX-Python oracle.

## Source / credit
- **Article:** <https://karpathy.github.io/2026/02/12/microgpt/>
- **Gist (microGPT source):** <https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95>
- **Gist (build-up progression):** <https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a>
- **Names dataset:** <https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt>
- **microgpt on the web:** <https://karpathy.ai/microgpt.html> · micrograd video: <https://www.youtube.com/watch?v=VMj-3S1tku0>

## What we built
| | Karpathy's microGPT | this spike (`../spikes/spike-microgpt.ts`) |
|---|---|---|
| Autograd | hand-rolled `Value` class | **real MLX `value_and_grad`** over FFI (`../training/train.ts`) |
| Language | pure Python, no deps | TypeScript + Bun FFI → `libmlxc.dylib` |
| Compute | scalar, CPU | MLX kernels on Apple GPU (Metal) |
| Model | 1 block, 4 heads, ~4k params | **identical**: 1 block, 4 heads, **4000 params** |
| Tokenizer | char-level, 27 tokens | char-level, 27 tokens (`\n`=BOS/EOS + `a..z`) |
| Data | makemore names corpus | same corpus, document-by-document |
| Optimizer | Adam + linear LR decay | Adam + linear LR decay |

Architecture (faithful to his, in `forward()`): token + positional embeddings →
single GPT-2-style block (pre-LayerNorm, 4-head causal attention via MLX
`scaled_dot_product_attention`, GELU MLP at 4×) → final LayerNorm → **tied**
`lm_head`. Trained with cross-entropy next-token loss, one name per step, 1000
steps, LR `1e-2` decaying linearly to 0.

## Run it
```sh
bun ../spikes/spike-microgpt.ts          # fetches data/names.txt on first run; writes /tmp/microgpt-init.f32
python3 ../reference/reference-microgpt.py  # the MLX-Python oracle (run the TS spike first — it writes the shared init)
```

## Result
```
=== microGPT in mlx-ts: 4000 params (D=16, 4 heads, 1 layer), real MLX autograd over FFI ===
  step    0: loss 3.1896
  ...
  step  900: loss 2.9387
STEP0 loss=3.1896
FINAL loss=1.7235
samples: koeael manna toliar jalanle orseie kanniyn jala jaryike holy kalea
```
It learns name-like structure from scratch ("manna", "toliar", "holy", "kalea").
This is a *toy* by design — 4k params, single-document steps, so per-step loss is
noisy and samples are babble-names, not real words. The point isn't the model;
it's that a **complete transformer — forward, backprop, Adam, sampling — trains
end-to-end with TypeScript driving real MLX**.

## Validation (the project's bar: parity vs MLX-Python)
`../reference/reference-microgpt.py` trains the identical model from the **same init** (shared
via `/tmp/microgpt-init.f32`, written by the TS spike) on the **same data order**:

- **Step-0 loss is exact: `3.1896` (TS) = `3.1896` (PY)** — clean parity of init +
  forward + loss across the FFI boundary.
- **Both converge** from 3.19 (TS → 1.72, PY → 2.06). The final-loss drift is the
  known training non-determinism (`FINDINGS.md` §6 gotcha #8: fp reductions + Adam
  diverge over many steps), so the bar is *"identical start, both converge"* — the
  same criterion used for the LoRA check.

Wired into `../scripts/validate-all.sh` (guarded on `data/names.txt`).

## What it re-confirmed about mlx-ts
The first run hit MLX's buffer limit around step 900 — a live reproduction of
`FINDINGS.md` §6 **gotcha #5**: the `FinalizationRegistry` never fires inside a
tight synchronous loop, so every step's forward intermediates + gradients + Adam
temporaries pile up. Fixed the project's way: wrap each step in `tidy()`, keep
only the new params + Adam moments, and explicitly `free()` the prior generation.
A good reminder of *why* `tidy()` is the load-bearing memory primitive here.

## Files
- `../spikes/spike-microgpt.ts` — the model, training loop, and sampler
- `../reference/reference-microgpt.py` — the MLX-Python oracle for the parity check
- `data/names.txt` — fetched corpus (git-ignored)
