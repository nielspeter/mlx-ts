# Changelog

Notable changes, newest first. Hand-written.

## [0.1.1]

First release through the trusted-publishing pipeline (OIDC from GitHub
Actions), rather than an interactive `npm publish`.

### Fixed

- `.gitignore` matched `models/` unanchored, so it also matched `src/models/`
  and kept `src/models/load.ts` — the hub loader — out of git. The published
  0.1.0 package was unaffected (`dist/` is built from the working tree), but the
  repository could not typecheck. Patterns are now anchored to the repo root.

### Changed

- `tools/audit.ts` gains a rule: every imported file must be *tracked* by git,
  not merely present on disk. The audit had been reading a working tree a fresh
  clone would not have, which is what let the above through.
- CI is a single macOS job. The Linux half could never have worked — typecheck
  and the audit both read `src/ffi/generated.ts`, which codegen emits from the
  installed mlx-c headers. It now also confirms GitHub's macOS runners give MLX
  a usable Metal device.
- README carries npm/CI/licence badges and links both published packages;
  corrected the suite count to 47/47.

## [0.1.0]

First release. A TypeScript MLX SDK over Apple's `mlx-c` via FFI: no custom
C/C++, no build step, running on Bun, Deno and Node.

### Added

- **Arrays and memory** — `MX` over mlx-c handles, `tidy()` scoped freeing,
  `escape()` for state that must outlive a scope, `Disposable` support.
- **Modules and training** — `Module`, `Linear`, `RMSNorm`, `Embedding`,
  quantized variants, `MoE`; `valueAndGrad` over a pytree, `Adam`,
  cross-entropy. Proven from a linear fit up to a full fine-tune of GPT-2-124M.
- **Generation** — `generate`, `streamTokens`, `streamText` with temperature,
  top-p, top-k and repetition penalty; the KV cache frees itself on completion,
  early `break` and throw.
- **Models** — Qwen3 (4-bit), OLMoE (64-expert MoE), GPT-2, Whisper; single-file
  and sharded checkpoints.
- **`load("org/repo")`** — fetches config, tokenizer and weights from the
  Hugging Face hub and caches them under `~/.cache/mlx-ts`.
- **Three runtimes** — one FFI contract, one backend each for `bun:ffi`,
  `Deno.dlopen` and koffi. Bit-identical output on all three.
- **Tokenizers** — byte-level BPE and HF chat templates, validated against
  Hugging Face's own `tokenizers` and `jinja2`.

### Verified

Output matches Apple's `mlx-lm` token for token. `scripts/validate-all.sh` runs
47 checks against MLX-Python and Hugging Face references.

### Known limits

- Apple Silicon and Metal only.
- `brew install mlx-c` is required until the platform package is published.
- Only quantized checkpoints load via `load()`; `model_type` support is `qwen3`
  and `olmoe`.
- `training/` is Bun-only (`Bun.mmap` has no portable equivalent).
- No continuous batching; generation is serialized behind a mutex.
