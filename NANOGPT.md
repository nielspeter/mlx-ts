# nanoGPT in mlx-ts — a real small GPT trained from scratch, TS driving MLX

The next rung up from [`MICROGPT.md`](./MICROGPT.md): a **genuine multi-layer GPT**
— not a toy — trained from scratch on char-level tiny-shakespeare, with the
autograd being **real Apple MLX driven from TypeScript over `mlx-c` via Bun FFI**.
Where microGPT proved the *mechanism* (one block, ~4k params), this proves the
*real training recipe* works end-to-end from TS: mini-batches, a deep stack,
AdamW with weight decay, a cosine LR schedule with warmup, and global gradient
clipping — the actual machinery you need to train something useful.

## Source / credit
- **Repo:** <https://github.com/karpathy/nanoGPT>
- **Dataset:** tiny-shakespeare — <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>

## What we built
Faithful to nanoGPT's `shakespeare-char` recipe, scaled to run on a single Mac:

| nanoGPT ingredient | this spike (`spike-nanogpt.ts`) |
|---|---|
| Multi-layer GPT-2 block stack | ✅ N pre-LN blocks (default 4), tied `lm_head` |
| Multi-head causal attention | ✅ MLX `scaled_dot_product_attention` (causal) |
| **Mini-batches `[B, T]`** | ✅ `[32, 64]`, random crops of the corpus |
| **AdamW** (decoupled weight decay) | ✅ inlined, decay on 2D params only |
| **Cosine LR + linear warmup** | ✅ `lrAt()` mirrors nanoGPT's `get_lr` |
| **Global grad-norm clipping** | ✅ clip at 1.0 across all grads |
| Char-level tokenizer | ✅ sorted-unique vocab (65 chars) |
| train/val split + val-loss eval | ✅ 90/10, averaged over fixed eval batches |
| Autograd | **real MLX `value_and_grad`** over FFI (`train.ts`) |
| DDP / `torch.compile` | N/A — single device; MLX has its own lazy graph |

Default config: 4 layers, 4 heads, `n_embd=128`, `block_size=64`, batch 32,
2000 iters, lr `1e-3`→`1e-4` cosine, weight decay `0.1`, grad clip `1.0`,
AdamW betas `(0.9, 0.95)`. ~**0.8M params**. All knobs are env-overridable
(`N_LAYER`, `N_HEAD`, `N_EMBD`, `BLOCK`, `BATCH`, `ITERS`).

## Run it
```sh
bun spike-nanogpt.ts           # fetches input.txt; writes shared init + batch-index files to /tmp
python3 reference-nanogpt.py   # the MLX-Python oracle (run the TS spike first)
ITERS=300 N_LAYER=6 bun spike-nanogpt.ts   # bigger/shorter, etc.
```

## Result (2000 iters, ~0.8M params)
```
iter 1800: loss 1.4841 (lr 1.2e-4)
STEP0 loss=4.1783
FINAL train loss=1.5402
VAL loss=1.7157

--- sample ---
ANTHSAS:
O there Caunious of the fore.

PRINGORETHAR:
Come, my no hearth's widow's no they confer a king:
You this askle swear Caulio,
And all than hath the hath beain eyes,
...
```
It learns Shakespeare's *shape* from raw characters: `SPEAKER:` turns, line
breaks, and real words ("against", "widow's", "swear", "king", "eyes"). nanoGPT's
own char baseline reaches ~1.47 val with a 10.7M model (6 layers / 384 dim / 256
block); ~1.7 with **13× fewer params** is right where it should be.

## Validation (parity vs MLX-Python)
`reference-nanogpt.py` trains the identical model from the **same init**
(`/tmp/nanogpt-init.bin`) and the **same mini-batches**
(`/tmp/nanogpt-train-idx.bin`, `/tmp/nanogpt-val-idx.bin`), all written by the TS
spike. At 100 iters the runs are **bit-identical end-to-end** — step-0, final
train loss, *and* val loss all match to 4 decimals (4.1783 / 2.6014 / 2.5312) —
because identical init + identical batches + the same MLX kernels leave nothing to
diverge. (Over thousands of steps fp-reduction order can drift — FINDINGS §6
gotcha #8 — so the suite uses the fast, exact 100-iter check.) Wired into
`validate-all.sh` (guarded on `input.txt`).

## Why this matters for the feasibility study
microGPT showed the autograd *mechanism* is reachable from TS. nanoGPT shows the
*production training recipe* is too: deep stacks, mini-batching, AdamW, LR
schedules and grad clipping are all just MLX ops orchestrated from TypeScript —
no custom C/C++, no gap. The only thing standing between this and a from-scratch
GPT-2-124M reproduction is single-Mac wall-clock and a BPE *encoder* (we ship
tiktoken decode-only), **not** any missing capability in the TS↔MLX bridge.

## Files
- `spike-nanogpt.ts` — model, mini-batch pipeline, AdamW training loop, sampler
- `reference-nanogpt.py` — MLX-Python oracle for the parity check
- `input.txt` — tiny-shakespeare corpus (fetched, git-ignored)
