# Agent handoff — mlx-ts

A working proof-of-concept: a **TypeScript MLX SDK over Apple's `mlx-c` C API via
Bun FFI** — no custom C/C++, no build step. It runs real dense and MoE LLMs
(inference + LoRA training), every layer validated against MLX Python or HF.
**`validate-all.sh` is 20/20 green.** Read `FINDINGS.md` first — it's the full,
honest write-up; this file is the operational guide.

## Orientation (read in this order)
1. `FINDINGS.md` — what was proven, how, and the non-obvious engineering findings.
2. `README.md` — per-file map + the `curl` commands to fetch model files.
3. This file — environment, how to run, gotchas, and what's next.

## Environment / prerequisites
- **macOS, Apple Silicon** (the ABI trick below assumes ARM64).
- **Bun** (developed on 1.3.14). `bun install` for the one dep (`@huggingface/jinja`).
- **`mlx-c` via Homebrew**: `brew install mlx-c` → `libmlxc.dylib` at
  `/opt/homebrew/lib/`. ⚠️ **Version-pinned paths**: `codegen.ts` reads headers
  from `/opt/homebrew/Cellar/mlx-c/0.6.0_2/include/mlx/c`. If your mlx-c version
  differs, update `INC` in `codegen.ts` (and the `dlopen` lib path is
  `/opt/homebrew/lib/libmlxc.dylib` in `mx.ts`, `train.ts`, `lora-train.ts`,
  `spike-train.ts`, the early `mlx.ts`).
- **Python** with `mlx`, `tokenizers`, `jinja2` (for the reference validations).
- **Model files are git-ignored but currently present** in this dir (≈9 GB), so
  it's runnable as-is. To re-fetch, use the `curl` commands in `README.md`.

## How to run / validate
```sh
bun codegen.ts        # REQUIRED first: regenerates generated.ts (the FFI binding) from mlx-c headers
bun validate-all.sh   # full suite — 20 checks, TS vs MLX-Python / HF
bun test tests/       # per-op binding parity vs MLX (fixtures from tests/gen-fixtures.py)
bun chat.ts "Write a haiku about the sea"   # a real chat turn (4-bit Qwen3)
bun stream.ts "Write a haiku about the sea" # streaming chat over the public lm.ts API
bun server.ts                               # HTTP server: chat UI /, /v1/chat/completions, /v1/embeddings, /v1/audio/transcriptions
bun lora-train.ts     # LoRA fine-tune of 4-bit Qwen3 (loss 3.16 -> 0.0007)
```
Every TS path has a `reference-*.py` MLX/HF mirror; parity is the bar.

## The one ABI fact everything rests on
Every mlx-c handle is `struct { void* ctx; }` — a single pointer. On ARM64 it's
passed/returned in a register exactly like a bare pointer, so **every handle is
modeled as `ptr`** in Bun FFI (which has no by-value-struct support). The codegen
generalizes this: any opaque `mlx_*` type → `ptr`.

## Codebase
- **`codegen.ts` → `generated.ts`** — parses mlx-c headers, emits the FFI symbol
  table (472) + 242 typed wrappers. **Never hand-edit `generated.ts`; regenerate.**
  To expose more of mlx-c, add a header to `RUNTIME_HEADERS` (ops/fast are wrapped).
- **`mx.ts`** — `MX` array class: handle + `FinalizationRegistry` + **`tidy()`**
  (deterministic per-scope free), ops, sampling, async eval, memory controls.
- **`nn.ts`** — `Module` (+ `parameters()`), `Linear`, `QuantizedLinear`, `RMSNorm`,
  `Embedding`, `QuantizedEmbedding`, `MoE`, `LoraDelta`.
- **`loader.ts`** — safetensors: `singleFileWeights` / `shardedWeights` / `freeMap`.
- **`tokenizer.ts`** (byte-level BPE), **`chat-template.ts`** (`@huggingface/jinja`).
- **Training**: `optim.ts` (Adam), `loss.ts` (cross_entropy), `pytree.ts`,
  `train.ts` (tree-based `valueAndGrad`).
- **`lm.ts`** — public generation surface: `Decoder` interface +
  `streamTokens` / `streamText` / `generate` (async generators, KV cache
  auto-freed; no caller-side `tidy()`). `MX` is `Disposable`.
- **`audio.ts`** — speech front-end: ffmpeg decode → 16 kHz mono PCM →
  Whisper-style log-Mel (rfft computed as a DFT matmul, no FFT binding needed).
  Validated vs numpy FFT (`reference-mel.py` / `audio-test.ts`).
- **`whisper.ts`** — Whisper STT, end to end: encoder + decoder (cross-attention)
  + greedy decode + `whisper-tokenizer.ts` (tiktoken decode-only). `bun whisper.ts
  <audio>` transcribes; **token-exact vs mlx_whisper** (`whisper-transcribe-test.ts`,
  also `whisper-test.ts` for encoder/decoder parity). Needs `whisper-tiny.safetensors`
  (npz→safetensors), `whisper-mel-filters-80.f32` + `whisper-multilingual.tiktoken`
  (from mlx_whisper assets) — all git-ignored, see README.
- **`tests/`** — per-op binding parity (`bun test`): `gen-fixtures.py` runs each
  op through MLX Python and saves inputs+output to `fixtures.json`; `lib.test.ts`
  feeds identical inputs through mlx-ts and asserts `allclose`. Covers elementwise,
  reductions, shape, fast (rms_norm/rope/sdpa), and quantized/MoE (qmm/dequantize/
  gather_qmm). **`benchmarks/`** — perf timing vs MLX Python (see its README).
- **Models**: `qwen.ts` (bf16), `qwen-nn.ts` (4-bit), `olmoe.ts` (MoE),
  `chat.ts`, `lora-train.ts`. `spike-*.ts` = de-risking experiments —
  e.g. `spike-microgpt.ts` trains Karpathy's ~4k-param microGPT from scratch
  (real MLX `value_and_grad` over FFI, vs `reference-microgpt.py`: step-0 loss
  exact, both converge). Per-step `tidy()` + freeing prior params/moments keeps
  the 1000-step loop inside MLX's buffer limit (gotcha #5). `spike-nanogpt.ts`
  scales this to a real multi-layer char-level GPT (tiny-shakespeare,
  mini-batched `[B,T]`, AdamW + cosine LR + warmup + grad clip + dropout). At
  nanoGPT's exact 10.7M `shakespeare-char` config it reaches best val ≈ 1.50
  (≈ their 1.47 baseline); the dropout-free path is bit-exact vs
  `reference-nanogpt.py` (shared init + batches). `mx.dropout` (device-side
  `mlx_random_bernoulli`) was added for this — train-only, seed-derived.

## Gotchas — do NOT re-learn these (full detail in FINDINGS §6)
1. Empty handles return `null`, not `0` → normalize `?? 0`.
2. Out-params overwrite the handle slot (`int fn(mlx_array* res, …)`) → read it back.
3. `mlx_optional_float/int` are 8-byte structs → packed into a `u64`.
4. The safetensors `Load` primitive is **CPU-only** → load on a CPU stream.
5. **`FinalizationRegistry` does NOT bound memory in a sync loop** (GC never fires).
   `tidy()` is the fix: measured +3034 MB → +23 MB over 200 tokens.
6. **Refcount safety**: freeing a JS handle MLX still needs internally is safe —
   MLX's primitive holds its own ref until eval. This is what makes `tidy()` +
   eager cache-freeing + async eval correct.
7. **Memory `active` over-counts mmap** (evictable); physical RSS ≈ model size.
   MoE stacking duplicates experts → call `freeMap` after load. Sharded loaders
   should mmap-all (evictable), NOT stream-materialize (measured worse).
8. **Training is not bit-reproducible** over a 4-bit base (bf16 rounding + chaos
   near instabilities). Validation criterion: identical start + both converge.
9. The codegen **skips function-pointer params** (closures) → `dlopen`
   `mlx_closure_new_func` by hand (see `train.ts` / `spike-train.ts`).
10. **`vmap` has no public mlx-c symbol**, but is **recoverable** over FFI from
    the internal `mlx_detail_vmap_trace`/`_replace` primitives (`spike-vmap.ts`:
    single-input, shared-input `-1` axis, and per-sample gradients `vmap(grad)` all
    validated). Not needed for minibatch training anyway. Caveat: `mlx_detail_*`
    are internal/version-unstable — a capability spike, not a public API.

## How to extend (the patterns that work)
- **New op**: usually already wrapped (ops.h/fast.h) — just add an `MX` method
  if you want ergonomics. Else add its header to `codegen.ts` and regenerate.
- **New model**: write a forward reusing `nn` modules, add a `reference-*.py`
  MLX mirror, and a line in `validate-all.sh`. Token-exact vs MLX for inference.
- **New architecture (MoE/other)**: it's mostly weight-key mapping + the right
  attention/FFN shape — see `olmoe.ts` (full-dim q/k-norm, stacked experts,
  quantized router, `norm_topk_prob`) as the template for real-model quirks.

## What's next (from FINDINGS §7 — none are feasibility risks)
- ✅ **Public streaming / disposable API** — done in `lm.ts`: async-generator
  `streamTokens` / `streamText` over a model-agnostic `Decoder` interface, KV
  cache auto-freed via `try/finally` (completion / early `break` / throw), `MX`
  is `Disposable`. Validated in `stream-test.ts`. Both `Qwen3` (dense) and
  `OLMoE` (MoE) implement `Decoder` and generate through the same path.
- **Ragged-batch padding masks** (sdpa's `mask_arr` already accepts one).
- **Training breadth**: AdamW/schedulers, more losses, multi-example batches, the
  backward-memory regime at scale (`tidy()` *after* the step).
- **Coverage**: more architectures (key-mapping), quant formats (AWQ/GPTQ, bit-
  widths), sampling (top-k, penalties), and **npm packaging of a prebuilt
  `libmlxc`** (the only native artifact to ship).

## Working norms
- **Validate against MLX/HF, always.** The project's value is in *proven*, not
  asserted — keep the parity checks and be explicit about caveats.
- This is a **standalone repo**. The upstream MLX checkout is a sibling at
  `../mlx` — do **not** commit there; it's not ours.
- Model files are large + git-ignored. Don't commit them. `bun codegen.ts`
  regenerates `generated.ts` (also ignored).
- Commits: conventional, no AI attribution, first line < 72 chars.
