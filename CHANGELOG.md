# Changelog

Notable changes, newest first. Hand-written.

## [0.2.0]

Text-to-music. The headline is `MusicGen`, which makes this the first release
where the SDK runs a generative *audio* pipeline end to end, not just text.

### Added

- **MusicGen — prompt in, `.wav` out** (`src/models/musicgen.ts`). The whole
  chain is TypeScript: SentencePiece **Unigram** tokenizer
  (`src/text/unigram.ts`) -> **T5** encoder (`src/models/t5.ts`) -> the
  **codebook LM** (4 delayed EnCodec codebooks, cross-attention,
  classifier-free guidance) -> the **EnCodec** decoder
  (`src/models/encodec.ts`), whose LSTM runs on a hand-written Metal kernel.
  LM logits match Hugging Face's own implementation.

      const model = await MusicGen.fromPretrained();
      const audio = model.generate("trance");
      await saveAudio("out.wav", audio.toF32(), model.samplingRate);

  Two repo layouts load. `facebook/musicgen-small` ships one safetensors under
  Hugging Face names; `-medium` and `-large` ship only PyTorch pickles, which
  TypeScript cannot read, so those come from `jasonvassallo/mlx-musicgen-*` and
  are mapped back onto the HF names. `validation/musicgen-mlx-layout.ts` checks
  all 880 names the model asks for against the real safetensors headers using
  range requests, so a wrong mapping fails in seconds rather than 7 GB into a
  download.
- **`metalKernel()`** (`src/ffi/kernel.ts`) — custom Metal kernels authored in
  TypeScript, with an example. EnCodec's LSTM is the first real user.
- **`saveAudio()`** (`src/audio/wav.ts`) — 16-bit mono PCM writer.
- **`Owned<T>` and `freeAll()`** (`src/core/mx.ts`) — the cross-scope half of
  the memory model. `escape()` transfers ownership but says nothing about who
  frees the value being replaced; `Owned.set()` does, and frees the previous
  occupant minus anything the new value still references.
- **`setCacheLimit()`** — the only knob that bounds MLX's buffer-reuse pool.
- **Biome**, run before `git push` and in CI.

### Fixed

- **A generation loop could reach 28 GB.** MLX parks freed Metal buffers in a
  reuse pool whose default ceiling is the machine's RAM — `set_cache_limit`
  reported 36722 MB on a 36 GB box — and a decode loop fills it: 8.4 GB of
  weights plus 19.7 GB of cache on musicgen-medium. `generate()` now caps it
  for the duration and restores it after, which costs nothing measurable
  (61.9 steps/s uncapped against 61.4 at a 256 MB cap). MLX is lazy, so the cap
  spans the evaluation rather than the call that builds the graph.
- **A leaked KV cache.** The step replacing the cache did not free the pair it
  replaced, which cost ~10 MB per step. Fixed, then made unrepresentable by
  `Owned`.
- **`hubFile()` buffered whole downloads in memory** and restarted from zero
  when interrupted. It now streams to disk and resumes from a partial file —
  a MusicGen medium checkpoint is ~7 GB.
- **`saveAudio()` clamped overshoot flat** into audible distortion; it now
  scales the clip instead.

### Changed

- `scripts/sandbox.sh` polls system-wide available pages via `vm_stat`, sums
  RSS across the process group, and refuses a ceiling above 60% of physical
  RAM. It previously polled the direct child's RSS — which cannot see MLX at
  all, since Metal buffers are not counted in process resident size (`ps`
  reported 7538 MB against MLX's own 28 GB), so the guard was blind to the
  memory it existed to catch.
- `docs/FINDINGS.md` gains §6.7 on why active memory is not the memory the
  operating system has to find.

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
