# Agent handoff — mlx-ts

A working TypeScript MLX SDK over Apple's `mlx-c` C API via FFI — no custom
C/C++, no build step, and it runs on **Bun, Deno and Node**. It runs real dense
and MoE LLMs (inference + LoRA training), every layer validated against MLX
Python or HF. **`scripts/validate-all.sh` is 49/49 green** against Homebrew
mlx-c (the count varies with which model files you have fetched). Read `docs/FINDINGS.md` first — it's the full,
honest write-up; this file is the operational guide.

## Orientation (read in this order)
1. `docs/FINDINGS.md` — what was proven, how, and the non-obvious engineering findings.
2. `README.md` — per-file map + the `curl` commands to fetch model files.
3. This file — environment, how to run, gotchas, and what's next.

## Environment / prerequisites
- **macOS, Apple Silicon** (the ABI trick below assumes ARM64).
- **A JS runtime**: Bun (developed on 1.3.14), Deno 2, or Node 24+. `bun install`
  (or `npm i`) for `@huggingface/jinja`, plus `koffi` if you run on Node.
  `src/` and `examples/` run on all three; only `training/` is still Bun-only.
- **`mlx-c` via Homebrew**: `brew install mlx-c`. Headers and the dylib are both
  resolved at runtime — Homebrew's version-independent symlinks first, then the
  newest Cellar install — so a `brew upgrade mlx-c` no longer breaks codegen.
  Override with `MLXTS_INCLUDE` (headers) and `MLXTS_LIB` (dylib).
- **Python** with `mlx`, `tokenizers`, `jinja2` (for the reference validations).
- **Model files are git-ignored and not in a fresh clone** (≈9 GB when fetched).
  Use the `curl` commands in `README.md`; `scripts/validate-all.sh` skips the
  checks whose files are missing.
- ⚠️ **`prebuilds/` vs Homebrew**: the bundled `libmlxc` is a different MLX build
  and does **not** match MLX-Python numerically (real Qwen3, LoRA). It is
  preferred over Homebrew when present, so parity runs want
  `MLXTS_LIB=/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib`. The suite prints which
  one it resolved.

## How to run / validate

Fast checks first — these need no weights, no GPU and no model files, and are
what CI runs:

```sh
bun run typecheck          # tsc --noEmit. Nothing compiles this project, so tsc
                           # is the ONLY thing that ever checks the types.
bun run audit              # tools/audit.ts: imports resolve, src/ imports nothing
                           # outside src/, no cross-directory imports, doc paths exist
bun test tests/            # op parity fixtures + arena/ownership semantics
bun examples/basics.ts     # arrays, ops, tidy() — no downloads
bash scripts/verify-package.sh   # pack, install into a temp project, use it from
                                 # Bun/Deno/Node — the only check that sees what a
                                 # consumer sees
```

Then the full thing (needs the model files, ~9 GB, and Apple Silicon):

```sh
bun tools/codegen.ts        # REQUIRED first: regenerates generated.ts (the FFI binding) from mlx-c headers
bash scripts/validate-all.sh  # full suite, TS vs MLX-Python / HF (also checks Deno + Node)
bun examples/chat.ts "Write a haiku about the sea"   # a real chat turn (4-bit Qwen3)
bun examples/stream.ts "Write a haiku about the sea" # streaming chat over the public src/text/lm.ts API
bun examples/server.ts                               # HTTP server: chat UI /, /v1/chat/completions, /v1/embeddings, /v1/audio/transcriptions
bun training/lora-train.ts     # LoRA fine-tune of 4-bit Qwen3 (loss 3.16 -> 0.0007)
```
Every TS path has a `reference/reference-*.py` MLX/HF mirror; parity is the bar.

**CI** (`.github/workflows/ci.yml`) runs typecheck + audit on Linux, and codegen,
tests and the zero-download examples on macOS. The full parity suite stays local:
it needs the weights. The macOS job is written but unverified — nobody has
confirmed GitHub's runners give MLX a usable Metal device.

## The one ABI fact everything rests on
Every mlx-c handle is `struct { void* ctx; }` — a single pointer. On ARM64 it's
passed/returned in a register exactly like a bare pointer, so **every handle is
modeled as `ptr`** in Bun FFI (which has no by-value-struct support). The codegen
generalizes this: any opaque `mlx_*` type → `ptr`.

## Codebase
- **`tools/codegen.ts` → `src/ffi/generated.ts`** — parses mlx-c headers, emits the FFI symbol
  table (472) + 242 typed wrappers. **Never hand-edit `src/ffi/generated.ts`; regenerate.**
  To expose more of mlx-c, add a header to `RUNTIME_HEADERS` (ops/fast are wrapped).
- **`src/core/mx.ts`** — `MX` array class: handle + `FinalizationRegistry` + **`tidy()`**
  (deterministic per-scope free), ops, sampling, async eval, memory controls.
- **`src/nn/nn.ts`** — `Module` (+ `parameters()`), `Linear`, `QuantizedLinear`, `RMSNorm`,
  `Embedding`, `QuantizedEmbedding`, `MoE`, `LoraDelta`.
- **`src/io/loader.ts`** — safetensors load: `singleFileWeights` / `shardedWeights` / `freeMap`;
  **write** via `mx.saveSafetensors(path, {name: MX})` (training checkpoints).
- **`src/text/tokenizer.ts`** (byte-level BPE inference), **`src/text/chat-template.ts`** (`@huggingface/jinja`).
  Tokenizer *training* (`reference/tok-train.py`) runs in native Rust (HF `tokenizers`, as
  nanochat does — not MLX compute); output `models/tokenizer.json` is consumed token-exact
  by `src/text/tokenizer.ts` (`tests/tok-train-test.ts`, 4/4).
- **Full nanochat-style pipeline** (`scripts/run.sh`, see docs/PIPELINE.md): dataset
  (TinyStories) → `reference/tok-train.py` (Rust BPE) → `training/data-prep.ts` (stream-encode →
  uint16 token shards) → `training/base-train.ts` (pretrain on `Bun.mmap`'d shards + ckpt)
  → `training/chat-sft.ts` (SFT from ckpt) → `training/chat-ckpt.ts` (CLI) / `examples/chat-web.ts` (web UI).
  Shared GPT + ckpt load/save in `src/models/nanogpt-model.ts`; handoff via safetensors.
- **Training**: `src/nn/optim.ts` (Adam), `src/nn/loss.ts` (cross_entropy), `src/core/pytree.ts`,
  `src/nn/autograd.ts` (tree-based `valueAndGrad`).
- **`src/text/lm.ts`** — public generation surface: `Decoder` interface +
  `streamTokens` / `streamText` / `generate` (async generators, KV cache
  auto-freed; no caller-side `tidy()`). `MX` is `Disposable`.
- **`src/audio/mel.ts`** — speech front-end: ffmpeg decode → 16 kHz mono PCM →
  Whisper-style log-Mel (rfft computed as a DFT matmul, no FFT binding needed).
  Validated vs numpy FFT (`reference/reference-mel.py` / `tests/audio-test.ts`).
- **`src/models/whisper.ts`** — Whisper STT, end to end: encoder + decoder (cross-attention)
  + greedy decode + `src/text/whisper-tokenizer.ts` (tiktoken decode-only). `bun src/models/whisper.ts
  <audio>` transcribes; **token-exact vs mlx_whisper** (`tests/whisper-transcribe-test.ts`,
  also `tests/whisper-test.ts` for encoder/decoder parity). Needs `models/whisper-tiny.safetensors`
  (npz→safetensors), `models/whisper-mel-filters-80.f32` + `models/whisper-multilingual.tiktoken`
  (from mlx_whisper assets) — all git-ignored, see README.
- **`tests/`** — per-op binding parity (`bun test`): `tests/gen-fixtures.py` runs each
  op through MLX Python and saves inputs+output to `tests/fixtures.json`; `tests/lib.test.ts`
  feeds identical inputs through mlx-ts and asserts `allclose`. Covers elementwise,
  reductions, shape, fast (rms_norm/rope/sdpa), and quantized/MoE (qmm/dequantize/
  gather_qmm). **`benchmarks/`** — perf timing vs MLX Python (see its README).
- **Models**: `src/models/qwen.ts` (bf16), `src/models/qwen-nn.ts` (4-bit), `src/models/olmoe.ts` (MoE),
  `src/models/gpt2.ts` (real OpenAI GPT-2-124M: learned pos-emb, LayerNorm+bias, fused QKV,
  `gelu_new`, tied head — token-exact vs `reference/reference-gpt2.py`; uses `src/text/tokenizer.ts`'
  `GPT2_SPLIT` BPE, 8/8 vs HF), `training/sft.ts` (full SFT of GPT-2-124M into a chatbot:
  chat format + completion-only loss, vs `reference/reference-sft.py`; see docs/SFT.md),
  `training/rl.ts` (GRPO RL on GPT-2-124M: rollout + reward + group-relative advantage +
  advantage-weighted-NLL update; loss path vs `reference/reference-rl.py`; see docs/RL.md),
  `examples/chat.ts`, `training/lora-train.ts`. Note: `crossEntropy` (src/nn/loss.ts) uses stable
  `log_softmax` (`x − logsumexp`) — naive `softmax().log()` NaNs in backward at
  saturation (hit by SFT memorizing examples). `spike-*.ts` = de-risking experiments —
  e.g. `validation/spike-microgpt.ts` trains Karpathy's ~4k-param microGPT from scratch
  (real MLX `value_and_grad` over FFI, vs `reference/reference-microgpt.py`: step-0 loss
  exact, both converge). Per-step `tidy()` + freeing prior params/moments keeps
  the 1000-step loop inside MLX's buffer limit (gotcha #5). `validation/spike-nanogpt.ts`
  scales this to a real multi-layer char-level GPT (tiny-shakespeare,
  mini-batched `[B,T]`, AdamW + cosine LR + warmup + grad clip + dropout). At
  nanoGPT's exact 10.7M `shakespeare-char` config it reaches best val ≈ 1.50
  (≈ their 1.47 baseline); the dropout-free path is bit-exact vs
  `reference/reference-nanogpt.py` (shared init + batches). `mx.dropout` (device-side
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
   `mlx_closure_new_func` by hand (see `src/nn/autograd.ts` / `validation/spike-train.ts`).
10. **`vmap` has no public mlx-c symbol**, but is **recoverable** over FFI from
    the internal `mlx_detail_vmap_trace`/`_replace` primitives (`spikes/spike-vmap.ts`:
    single-input, shared-input `-1` axis, and per-sample gradients `vmap(grad)` all
    validated). Not needed for minibatch training anyway. Caveat: `mlx_detail_*`
    are internal/version-unstable — a capability spike, not a public API.

## How to extend (the patterns that work)
- **New op**: usually already wrapped (ops.h/fast.h) — just add an `MX` method
  if you want ergonomics. Else add its header to `tools/codegen.ts` and regenerate.
- **New model**: write a forward reusing `nn` modules, add a `reference-*.py`
  MLX mirror, and a line in `scripts/validate-all.sh`. Token-exact vs MLX for inference.
- **New architecture (MoE/other)**: it's mostly weight-key mapping + the right
  attention/FFN shape — see `src/models/olmoe.ts` (full-dim q/k-norm, stacked experts,
  quantized router, `norm_topk_prob`) as the template for real-model quirks.

## What's next (from FINDINGS §7 — none are feasibility risks)
- ✅ **Public streaming / disposable API** — done in `src/text/lm.ts`: async-generator
  `streamTokens` / `streamText` over a model-agnostic `Decoder` interface, KV
  cache auto-freed via `try/finally` (completion / early `break` / throw), `MX`
  is `Disposable`. Validated in `tests/stream-test.ts`. Both `Qwen3` (dense) and
  `OLMoE` (MoE) implement `Decoder` and generate through the same path.
- **Ragged-batch padding masks** (sdpa's `mask_arr` already accepts one).
- **Training breadth**: AdamW/schedulers, more losses, multi-example batches, the
  backward-memory regime at scale (`tidy()` *after* the step).
- **Coverage**: more architectures (key-mapping), quant formats (AWQ/GPTQ, bit-
  widths), sampling (top-k, penalties). Packaging is done
  (`@nielspeter/mlx-ts`); **bundling a prebuilt `libmlxc`** is not — that
  build disagrees with MLX-Python numerically.

## Working norms
- **Validate against MLX/HF, always.** The project's value is in *proven*, not
  asserted — keep the parity checks and be explicit about caveats.
- This is a **standalone repo**. The upstream MLX checkout is a sibling at
  `../mlx` — do **not** commit there; it's not ours.
- Model files are large + git-ignored. Don't commit them. `bun tools/codegen.ts`
  regenerates `src/ffi/generated.ts` (also ignored).
- Commits: conventional, no AI attribution, first line < 72 chars.
