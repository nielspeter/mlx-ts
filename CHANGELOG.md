# Changelog

Notable changes, newest first. Hand-written.

## [0.4.0]

Text to speech, and voice cloning. Also the first release that requires nothing
at runtime: `dependencies` is empty.

### Added

- **Spark-TTS — text in, a `.wav` out.** `examples/spark-tts.ts`. A
  **Qwen2-0.5B** (`src/models/qwen2.ts`) predicts audio tokens out of a 166k
  vocabulary, and **BiCodec** (`src/models/bicodec.ts`) renders them: a codebook
  quantizer, an FSQ speaker decoder, a 12-layer Vocos prenet conditioned through
  AdaLayerNorm, and a Snake-activation wave generator that upsamples 320x to
  16 kHz. The voice is *described* — gender, pitch, speed. No phonemizer, no
  `espeak-ng`. About 2x realtime.

- **Voice cloning.** `examples/spark-clone.ts`: a recording in, the same voice
  saying something else. BiCodec's **speaker encoder** (`src/models/speaker.ts`)
  turns six seconds of audio into the 32 tokens that stand for a speaker — a
  Slaney mel front end, an **ECAPA-TDNN** with Res2Net blocks and attentive
  statistics pooling, a **perceiver resampler** that squeezes any clip length
  into 32 latents, and **FSQ** to pack them into ids.

  Checked against the *original PyTorch* Spark-TTS rather than another MLX port,
  which is the only reason a real bug was caught — see below. All 32 ids match
  on synthetic and on real audio. Cloning is also verified end to end with no
  Python: `validation/spark-clone.ts` clones a voice and scores it with ECAPA's
  x-vector, a different head from the one the tokens come from (~0.95 against a
  ~0.38 floor for an unrelated voice).

- **`upcastWeights(W)`** — read a bf16 checkpoint at float32. bf16 is fine to
  generate with and too coarse to compare with: Qwen2 carries outlier channels
  in the thousands that cancel in the last layer, so one ulp at layer 9 becomes
  a percent at the logits. At float32 our LM matches PyTorch exactly.

- **New ops**: `convTranspose1d`, `sin`, `square`, `maximum`, and `strides` /
  `contiguous` accessors. BatchNorm folded from running statistics, inference
  only by construction.

### Fixed

- **`toF32()` ignored strides.** A transposed array read back in its
  *pre*-transpose order under the new shape — right shape, wrong data, no error.
  This affected any tensor read back to the host after a `transpose()`, not just
  the new models.

- **`toF32()` segfaulted on a non-float32 array.** `mlx_array_data_float32`
  returns null for bf16 and the read walked off it. It casts now.

- **A short STFT window is centred inside `n_fft`, as `torch.stft` pads it.**
  We had followed mlx-audio, which left-aligns it. Both produce something that
  looks like a spectrogram and nothing downstream complains — it moved 12 of
  BiCodec's 32 speaker tokens. Whisper's `logMel` was never affected: it frames
  with `win_length == n_fft`, where the offset is zero.

### Changed

- **`@huggingface/jinja` is loaded on demand and is now optional.** Nothing in
  `src/` needs it but `src/text/chat-template.ts`, yet importing mlx-ts to run
  a diffusion model or a codec pulled in a templating language — and failed
  outright if it was absent. It is imported on demand now, the way
  `src/ffi/index.ts` picks its backend, so `dependencies` is empty and the
  library requires nothing at runtime.

  `ChatTemplate`'s constructor is private as a result; construction goes through
  `ChatTemplate.fromString()` or `.fromConfig()`, since a deferred import cannot
  be awaited in a constructor. `render()` stays synchronous.

- **The Spark checks compare against committed reference numbers.**
  `validation/spark-golden.json`, 4 KB, generated once from the original PyTorch
  by `reference/gen-spark-fixtures.py`. A live oracle would need torch,
  torchaudio, transformers and a source fetch of a package that is not on PyPI —
  so it would skip for almost everyone. An oracle nobody can run is an oracle
  that never runs.

## [0.3.1]

`npm i @nielspeter/mlx-ts` is now the entire install. No Homebrew, no ffmpeg.

### Changed

- **Audio and images decode through macOS instead of ffmpeg.** MLX is
  Apple-Silicon-only, so treating `/usr/bin` as unavailable was caution
  borrowed from a portability problem this project does not have.

      audio     ffmpeg -> afconvert   (built in)
      images    ffmpeg -> sips        (built in)
      playback  none   -> afplay      (built in, via playAudio)

  Measured rather than assumed: `afconvert` and ffmpeg return the same 176000
  samples from the same FLAC and differ by 0.19% rms, which becomes about 1%
  through the log-Mel front-end. Whisper transcribes both to the same 23
  tokens, identical to the `mlx_whisper` oracle. On images sips and ffmpeg
  disagree more — different resamplers — but the zero-shot ranking is unchanged
  and sips scores the correct label slightly higher.

  The Whisper parity checks still set `MLXTS_AUDIO_DECODER=ffmpeg`: their claim
  is token-for-token equality with an oracle that decodes that way, and a 1%
  difference in the mel could flip a token on some other sample. ffmpeg is a
  dev dependency now, not a user one.

### Added

- **`playAudio()`** — every audio path in this repo wrote a file and nothing
  ever made a sound.

### Fixed

- **`decodeAudio` walks the WAVE chunk list** instead of assuming the classic
  44-byte header. afconvert writes extra chunks, and skipping a fixed 44 bytes
  reads metadata as audio: the signal shifts, and it looks like two decoders
  disagreeing rather than a parser being lazy. That mistake overstated the
  difference between them by 16x before it was caught; `tests/mel.test.ts`
  covers the walk now.

## [0.3.0]

Images. mlx-ts now generates them, and reads them — which makes this the
release where the SDK covers text, speech, audio and vision rather than three
of the four.

### Added

- **Stable Diffusion, end to end** — a prompt in, a PNG out.

      bun examples/stable-diffusion.ts "an astronaut riding a horse" --size 384

  384x384 in about 4 seconds. Every piece was pinned against mlx-examples'
  own port loading the same checkpoint before the next was built: `conv2d` and
  `convTranspose2d`, `GroupNorm`, the VAE decoder, CLIP's text encoder and
  tokenizer, the UNet (686 tensors, matched first try) and the noise schedule.
  `src/image/png.ts` writes the result, verified by parsing the file back.

- **CLIP's vision tower** (`src/models/clip-vision.ts`) — images and text in
  one space, so cosine similarity classifies without any training:

      bun examples/clip-zeroshot.ts photo.jpg
      0.1889  a photo of a cat
      0.1083  a photo of a car

  The transformer is shared with the text tower
  (`src/models/clip-layers.ts`); what
  differs is that a stride-14 convolution splits the image into patches, a
  class token leads the sequence, and attention is **not** causal.

- **`src/image/load.ts`** — decode any image ffmpeg understands, scaled and
  centre-cropped the way CLIP's own preprocessor does.

- **`conv2d` / `convTranspose2d` on `MX`**, channels-last, plus `broadcastTo`,
  `mulScalar`, `addScalar`, `randomNormal` and `setCacheLimit`.

- **Test coverage, measured and gated.** There was none before; the first
  honest figure was 20% of functions. `scripts/coverage.sh` now reports and
  enforces in pre-push and CI, at **66% of functions / 74% of lines**, with a
  ratchet floor. 32 unit tests became 139.

### Fixed

- **A bun:ffi ABI cliff.** `conv2d` segfaulted at address `0x1` on Bun 1.3.14 —
  `groups`, passed as `1`, read where the stream handle belongs. The binding
  was correct; Node and Deno matched Python exactly. On Apple arm64 the first
  eight integer arguments travel in registers, and what breaks is an `int32`
  that spills past them. Fifteen symbols share that shape, all convolution or
  padding, and none had ever been called. **Requires Bun >= 1.4.0.** Written up
  as `docs/FINDINGS.md` 6.8.

- **MLX laziness, twice more.** The diffusion loop built one graph holding
  every UNet pass and evaluated them at the end, reporting a fictional 291
  steps/s; and the VAE decode ran after the cache cap was restored, reaching
  5.2 GB of reuse buffers. The full pipeline now peaks at 4.8 GB.

- **The Whisper checks had been skipping, not passing** — the suite called a
  reference script by a path left behind when those moved into `reference/`,
  silenced by `>/dev/null`.

### Changed

- The parity suite is **61/61**. Five Stable Diffusion checks are opt-in via
  `MLXTS_SD=1`: they load the 3.2 GB UNet twice and were the peak of the run.
  Skipping them says so out loud rather than passing over it.
- `tools/check-api.ts` covers the new modules. It did not before, which is how
  three exports stayed missing from the public API while every gate was green.

## [0.2.2]

Documentation only — no runtime change since 0.2.1. Cut because the install
instructions shipped inside the package were wrong, so every visitor to the npm
page was told to install Homebrew first.

### Fixed

- **`brew install mlx-c` is not required.** `npm i @nielspeter/mlx-ts` is the
  whole install: the native runtime comes from
  `@nielspeter/mlx-ts-darwin-arm64`, an `optionalDependency` published since
  0.1.0 and already declared here. The README still said it was "not published
  yet". Homebrew is still honoured, and still takes precedence when present.

  Verified rather than assumed — a plain clean-install test proves nothing here,
  because Homebrew comes first in the resolver and gets used silently. Forced
  onto the platform package with `MLXTS_LIB`, real Qwen3-0.6B produces identical
  token ids and the parity suite is 54/54.
- **The two Whisper checks had been skipping, not passing.** The suite invoked
  `reference/reference-whisper.py` by a bare filename left behind when the
  reference scripts moved into `reference/`, and silenced the call with
  `>/dev/null`, so it failed quietly and the test then died on the fixture it
  never wrote. The suite is 54/54, not 52/52,
  and the setup it needs — a venv at `/tmp/wvenv`, plus `/tmp/jfk.flac` — is now
  written down instead of being folklore.
- Nine further README corrections: the package size was given twice, as two
  different stale figures; "three examples" listed four; the summary of what
  mlx-ts does predated MusicGen, full fine-tuning and `metalKernel()`; the
  codegen notes still called the metal kernel-builder API "irrelevant to
  inference", when EnCodec's LSTM runs on it.

### Added

- **README: "What that looks like"** — chat, speech-to-text, text-to-music,
  embeddings, training and a custom Metal kernel, in a few lines each. The
  snippets are taken from the runnable files in `examples/`, and
  `validation/readme-snippets.ts` mirrors them so `tsc --noEmit` fails when the
  API moves under the docs. The first draft, written from memory, used five
  functions that do not exist.

## [0.2.1]

A patch for a bug that made 0.2.0 effectively single-shot.

### Fixed

- **`MusicGen.generate()` worked once per process** and then failed with
  `expected a non-empty mlx_array`. `T5Encoder` caches a constant on the
  instance but created it lazily *inside* `encode()`'s `tidy()`, so the arena
  freed it at scope exit while the field still pointed at the handle, and the
  next call read freed memory. Anyone writing a loop, a server, or a batch of
  variations hit it immediately.

  Lazy creation is what hid it: the first call always works, and every test
  called `generate()` exactly once — so neither the suite nor CI saw it. This
  is the same defect as `docs/FINDINGS.md` §6.6, and the only lazily-cached
  `MX` in the codebase. `validation/musicgen-e2e.ts` now generates twice, and
  the check was confirmed to fail with the fix reverted.
- **A clip too short for EnCodec** died inside MLX with `[conv] Spatial
  dimensions of input after padding cannot be smaller than weight spatial
  dimensions`. The delay pattern consumes one frame per codebook and the
  decoder needs three more; below that the error now names the minimum.
  Measured: 6 steps fails, 7 is the shortest that decodes.

### Added

- **`--seed`** on `examples/musicgen.ts`. Sampling draws from MLX's RNG, so a
  seeded prompt reproduces its take exactly — two CLI runs at `--seed 1234`
  produce byte-identical `.wav` files. Without it every run was a different
  take with no way back to one you liked.

### Changed

- CI fails when `package-lock.json` drifts from `package.json`. It had been
  stale since Biome was added, which only surfaced when the 0.2.0 publish
  aborted at `npm ci`; CI installs with `bun install --frozen-lockfile` and
  reads `bun.lock`, so nothing validated the npm lockfile. (`npm ci --dry-run`
  does not detect this — it exits 0 either way.)

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
