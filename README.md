# mlx-ts-poc — TypeScript → mlx-c → Metal

Proof-of-concept that a Qwen3 inference path can run from **TypeScript via Bun
FFI over `mlx-c`** (Apple's official C API) with **zero custom C/C++** and no
build step — and that it is **numerically identical** to MLX's Python reference.

## End to end: real Qwen3-0.6B generating text

```sh
curl -sL https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json -o config.json
curl -sL https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json -o tokenizer.json
curl -sL https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors -o model-qwen.safetensors
bun qwen.ts "The capital of France is"
#  completion: " Paris. The capital of France is also the capital of the French Republic. ..."
#  (24 tokens, ~190 tok/s)
python3 reference-qwen.py "The capital of France is"   # identical token ids
```

`qwen.ts` reads all dims from `config.json` (incl. Qwen3's explicit `head_dim`
and tied embeddings), loads the real weights, tokenizes with the validated
`tokenizer.ts`, and decodes with the KV cache — produced ids match MLX Python
token-for-token. Everything below is the validated machinery underneath it.

## What runs

- `mlx.ts` — a minimal **hand-written** Bun-FFI binding over `libmlxc.dylib`:
  handle management plus `matmul`, `rms_norm`, `rope`, `sdpa`, `silu`, etc.
- `codegen.ts` — **parses the mlx-c headers and emits `generated.ts`**: a full
  FFI symbol table (322 entries) + 242 typed op wrappers. The hand-written
  `mlx.ts` exists only to bootstrap; `generated.ts` supersedes it.
- `block.ts` — a full **Qwen3 decoder block** forward pass (mirrors mlx-lm's
  `qwen3.py`): pre-norm, GQA attention with per-head q/k RMSNorm, RoPE, causal
  SDPA, residual, SwiGLU MLP, residual — using the hand binding.
- `block-gen.ts` — the **same block built entirely from the generated wrappers**.
- `model-gen.ts` — a small **multi-layer Qwen3 model + KV-cache greedy decode
  loop** (prefill + autoregressive steps), built from the generated wrappers.
- `loader.ts` — safetensors loading over `mlx_load_safetensors`: open a file
  into a `string -> array` map, pull tensors by name, enumerate via iterator.
- `save-model.py` / `model-load.ts` — Python writes the model to a real
  `.safetensors`; TS loads it and runs the decode loop from the loaded weights.
- `inspect-real.ts` — loads a real mlx-community model shard and lists tensors.
- `reference-quant.py` / `model-quant.ts` — **4-bit quantized** path: Python
  quantizes the Linear projections (`mx.quantize`) and saves `weight`/`scales`/
  `biases`; TS loads them and runs the decode with `quantizedMatmul`.
- `tokenizer.ts` — pure-TS byte-level BPE tokenizer (the real Qwen3
  `tokenizer.json`); `tok-reference.py` / `tok-test.ts` validate it against HF
  `tokenizers` (encode + decode, 11/11 cases).
- `qwen.ts` / `reference-qwen.py` — **config-driven real Qwen3-0.6B** (bf16):
  reads `config.json`, loads `model-qwen.safetensors` (HF key names), generates
  text; ids match MLX Python token-for-token.

### Production runtime (mx + nn)

- `mx.ts` — `MX` array class: each wraps one handle, auto-freed by a
  `FinalizationRegistry`, plus a `tidy()` arena for **deterministic** freeing,
  ops, and temp/top-p sampling.
- `nn.ts` — `Module`, `Linear`, `QuantizedLinear`, `RMSNorm`, `Embedding`,
  `QuantizedEmbedding`.
- `qwen-nn.ts` / `reference-qwen-q4.py` — **real 4-bit Qwen3-0.6B**
  (mlx-community format) over `nn.Module`; greedy ids match MLX Python
  token-for-token. Supports temp/top-p sampling, batching, sliding window.
- `lm.ts` — public generation surface: a model-agnostic `Decoder` interface and
  async-generator `streamTokens` / `streamText` / `generate`. The KV cache is
  freed automatically (completion / early `break` / throw), so callers never call
  `tidy()` or free a handle; `MX` is `Disposable`. `stream.ts` is the live demo.
- `validate-prod.ts` — checks sampling reproducibility, batching, and bounded
  memory. `stream-test.ts` — stream output is identical to `generate()`.

```sh
bun qwen-nn.ts "The capital of France is"                      # greedy
bun qwen-nn.ts --temp 0.8 --topp 0.95 --seed 42 "Once ..."     # sampling
bun stream.ts "Write a haiku about the sea"                    # streaming API
bun server.ts                                                   # OpenAI-compatible HTTP server (:8080)
bun validate-prod.ts                                            # all three
```

`server.ts` is a working example of the local-server use case below — an
OpenAI-compatible `/v1/chat/completions` endpoint (streaming SSE or JSON) over
`Bun.serve`, generation serialized behind an async mutex. It also serves a tiny
self-contained chat web UI (`chat.html`) at `/`:

```sh
bun server.ts                       # open http://localhost:8080 for the chat UI, or:
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true,"temperature":0.7,"top_k":40}'
curl localhost:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["a sentence to embed","another one"]}'   # L2-normalized vectors for RAG
curl localhost:8080/v1/audio/transcriptions -F file=@audio.flac   # -> {"text": "..."}
```

`/v1/audio/transcriptions` (multipart `file`) is enabled when the Whisper assets
are present (see below); otherwise it reports 501 and the rest of the server runs
normally.

The embeddings come from mean-pooling Qwen3's last-layer hidden states (same
model/tokenizer, no extra weights) — RAG-useful similarity ranking, though a
dedicated embedding model would rank better.

**Memory — why `tidy()` and not just `FinalizationRegistry`:** FR only fires
after a GC, which never happens inside a tight synchronous decode loop, so
handles pile up. Measured over a 200-token generation: FR alone grew active
memory **+3034 MB**; `tidy()` (free everything in scope except the token + KV
cache) grew it **+23 MB** — the KV cache only. FR remains a backstop for arrays
created outside any `tidy()`.
- `reference.py` / `reference-decode.py` — the same block / the same decode loop
  in MLX Python, same deterministic weights.

## Run

```sh
bun codegen.ts       # parse headers -> generated.ts  (+ coverage report)
bun block.ts         # hand binding:      TS -> mlx-c -> Metal
bun block-gen.ts     # generated wrappers: TS -> mlx-c -> Metal
python3 reference.py # MLX Python reference
```

All three blocks print the same fingerprint:

```
  sum    = 0.005793
  sum_sq = 0.162600
```

The decode loop is checked the same way, but on discrete output — the greedy
token ids must match exactly (any drift in cache concat, RoPE offset, masking,
or sampling flips a token):

```sh
bun model-gen.ts          # TS + KV cache
python3 reference-decode.py
# both: generated: [24, 3, 19, 2, 28, 1, 4, 14, 4, 14, 4, 14]
```

### Loading weights from safetensors

```sh
python3 save-model.py     # writes a real model.safetensors (25 tensors)
bun model-load.ts         # loads it via mlx_load_safetensors, decodes
# -> same ids: [24, 3, 19, 2, 28, 1, 4, 14, 4, 14, 4, 14]

# and on a genuine model file:
bun inspect-real.ts ~/.cache/huggingface/hub/.../model-00001-of-00004.safetensors
# -> loaded ... — 881 tensors  (real names + shapes)
```

Gotcha: the safetensors `Load` primitive only implements `eval_gpu == no`, so
`loader.ts` loads on a **CPU stream**; the resulting concrete arrays then feed
the GPU compute graph normally.

### 4-bit quantized weights

Real mlx-community models are quantized. `reference-quant.py` quantizes the
Linear projections with `mx.quantize` (group_size 64, 4 bits) and stores three
tensors each — `weight` (packed uint32), `scales`, `biases`; norms/embedding
stay fp32. `model-quant.ts` loads them and uses `quantizedMatmul(x, w, scales,
biases, transpose=true, 64, 4, "affine")`.

```sh
python3 reference-quant.py && bun model-quant.ts
# both: generated: [27, 26, 16, 11, 12, 30, 26, 16, 11, 12, 30, 26]
```

The ids differ from the fp32 run — that is the real 4-bit quantization error,
and both the TS and Python paths exhibit it identically.

### Tokenizer (pure TS, no deps)

The one piece genuinely outside MLX. `tokenizer.ts` implements GPT-2-style
byte-level BPE over the real Qwen3 `tokenizer.json`: NFC normalization, the
special-token split, the GPT-2 pretokenization regex, the byte<->unicode map,
and rank-based merges.

```sh
curl -sL https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json -o tokenizer.json
python3 tok-reference.py && bun tok-test.ts
# -> encode/decode parity vs Python tokenizers: 11/11 cases pass
```

Validated against HF `tokenizers` on contractions, em-dash, per-digit numbers,
tabs/newlines, Chinese, Japanese, emoji, source code, and the chat template
(special tokens like `<|im_start|>`).

Note: this tokenizer has Qwen3's full 151k vocab, so end-to-end *text* output
needs a vocab-matched model (a real downloaded Qwen3) — i.e. the config-driven
loading step. The toy models here use a 32-token vocab for fast parity checks.

### Eval-boundary discipline

MLX is lazy: ops build a graph, nothing runs until forced. The decode loop
(`model-gen.ts`) calls `evalArray(...caches)` plus reads the token each step, so
the per-layer KV caches and the next token become concrete arrays. Skip this and
the graph grows every step — unbounded memory and recompute. Per-step RoPE uses
`offset = position`; prefill uses a `"causal"` mask, single-token decode uses
none (`""`).

## Codegen coverage

`bun codegen.ts` reports exactly what it does, with no silent drops:

```
parsed   392 decls across 6 headers
symbols  322 FFI entries
wrappers 242 typed op wrappers (from ops.h, fast.h)
skipped  70:  (all reported by name)
```

Skipped functions are exotic only — nested-vector / `char**` / device handles
(one-line FFI-map additions) and the metal/cuda **kernel-builder** API (opaque
builder structs, irrelevant to inference). Every standard tensor op is wrapped.

The generator maps each C type to an FFI type, auto-supplies the trailing
`mlx_stream`, collapses `const int* x, size_t x_num` pairs into a single
`number[]` param, exposes nullable arrays as `Arr | null`, and packs the
by-value `mlx_optional_float/int` structs into a `u64`.

## What it proves

Every inference-critical MLX `fast` op works over FFI and matches the reference:
`mlx_fast_rms_norm`, `mlx_fast_rope` (incl. the by-value `mlx_optional_float`
base, packed into a u64), `mlx_fast_scaled_dot_product_attention` with
`mask_mode="causal"` and GQA (4 query / 2 kv heads) handled inside the kernel.

## Key ABI facts the binding relies on

- Every mlx-c handle is `struct { void* ctx; }`. On Apple-silicon ARM64 a
  single-pointer struct is passed/returned in a register exactly like a bare
  pointer, so each handle is modeled as `ptr` (a JS number).
- An *empty* handle has `ctx == NULL`, which Bun returns as `null` → normalize
  to `0`.
- Out-params are `int fn(mlx_array* res, ...)`: pre-init the result slot with an
  empty handle, pass `&slot`, read the new handle back.
- `mlx_optional_float { float value; bool has_value; }` is an 8-byte non-HFA
  struct → passed in one integer register → modeled as a `u64` with the float
  bits in the low 32 bits and `has_value` in byte 4.

## What you can build with it

mlx-ts today is a **local inference runtime for text LLMs *and* Whisper
speech-to-text** (plus LoRA training) — Bun-only, Apple-Silicon-only, run as
scripts in this repo (not yet an npm package). Sampling supports greedy,
temperature, top-p, **top-k**, and **repetition penalty**
(`bun stream.ts --temp 0.8 --topp 0.95 --topk 40 --reppenalty 1.1 "…"`).

### ✅ Buildable now (everything needed exists)
- **Local chat assistant / CLI** — streaming replies, multi-turn via chat
  templates, temp/top-p/top-k/repetition-penalty sampling (`chat.ts`, `stream.ts`).
- **OpenAI-compatible server + chat web UI** — `server.ts` over `Bun.serve`:
  `/v1/chat/completions` (SSE/JSON), `/v1/embeddings`, `/v1/audio/transcriptions`,
  and a self-contained chat page at `/` with a **live mic** (record → transcribe →
  edit → send). Single-process / low-concurrency, not multi-tenant.
- **Speech-to-text (Whisper), multilingual** — `audio.ts` (log-Mel, ~1e-6 vs numpy
  FFT) + `whisper.ts` (Conv1d stem, bidirectional encoder, cross-attention decoder,
  KV cache) + `whisper-tokenizer.ts`. **Token-for-token identical to `mlx_whisper`**
  (`whisper-transcribe-test.ts`). Runs `large-v3-turbo` with **auto language
  detection** and a **sliding window** for unbounded dictation; Danish/Swedish/
  English verified. `bun whisper.ts audio.flac` (setup below).
- **Multilingual chat** — the server injects a system prompt so replies come back
  in the user's language (Danish in → Danish out).
- **Local RAG** — `POST /v1/embeddings` returns L2-normalized sentence vectors
  (mean-pooled Qwen3 hidden states); pair with any JS vector store.
- **Prompt-driven text tools** — summarize / rewrite / classify / extract /
  translate; **agent loops** (tool use via prompting + JS parsing).
- **LoRA fine-tuning** of 4-bit Qwen3 (Adam + cross-entropy, `lora-train.ts`).
- **Train a transformer from scratch** — `spike-microgpt.ts` builds Karpathy's
  ~4k-param microGPT (embeddings → attention → MLP → tied head) and trains it
  end-to-end on the names corpus with the autograd being **real MLX over FFI**
  (his hand-rolled `Value` engine replaced by `value_and_grad`); step-0 loss is
  exact vs the MLX-Python mirror, both converge.
- **Train a real small GPT** — `spike-nanogpt.ts` scales that up to nanoGPT:
  a multi-layer char-level GPT on tiny-shakespeare, **mini-batched `[B,T]`,
  AdamW + cosine LR + warmup + global grad clipping + dropout**. At nanoGPT's
  exact `shakespeare-char` config (6 layers, 384 dim, 10.7M params) it reaches
  **best val loss ≈ 1.50 — matching nanoGPT's ~1.47 baseline** — and writes
  coherent Shakespeare (real character names, dialogue). The dropout-free path
  is bit-exact vs `reference-nanogpt.py` (shared init + batches).
- **Run real GPT-2-124M** — `gpt2.ts` loads the actual OpenAI `gpt2` weights and
  generates with a **pure-TS GPT-2 BPE encoder** (`tokenizer.ts` + `GPT2_SPLIT`,
  8/8 token-exact vs HF) — `gelu_new`, LayerNorm-with-bias, tied head, KV cache,
  **token-exact vs `reference-gpt2.py`** at ~210 tok/s. See `GPT2.md`.
- **SFT a chatbot** — `sft.ts` **full-fine-tunes** real GPT-2-124M into an
  instruction follower (chat format + completion-only loss), the nanochat chat
  stage. Step-0 loss matches `reference-sft.py`; after SFT it answers in-format,
  including a held-out question (Italy → Rome). See `SFT.md`.
- **RL with GRPO** — `rl.ts` runs Group Relative Policy Optimization on GPT-2-124M
  (the nanochat RL stage): sample a group of completions, reward them, normalize
  advantage, policy-gradient update. Positivity-reward demo: mean reward rises ~9×;
  GRPO loss path validated vs `reference-rl.py`. See `RL.md`.
- **Train a tokenizer** — `tok-train.py` trains a byte-level BPE in **native Rust**
  (HF `tokenizers`, as nanochat does — training is a data-prep boundary step, not
  MLX compute); our pure-TS `tokenizer.ts` then reproduces it **token-exact**
  (`tok-train-test.ts`). The `tok_train` stage of a nanochat-style pipeline.
- **Pretrain + checkpoint** — `base-train.ts` pretrains a GPT from scratch on
  BPE-tokenized text and **saves a safetensors checkpoint** (`mx.saveSafetensors`,
  the write side of the loader) that reloads round-trip-clean — the keystone that
  lets pretrain → SFT/inference chain. The `base_train` stage.
- **Research / inspection** — pull logits, hidden states; the `MX` op surface is open.

Whisper setup (weights/assets are git-ignored — fetched, like the LLM weights):

```sh
W=https://huggingface.co/mlx-community/whisper-large-v3-turbo/resolve/main
curl -sL $W/config.json -o config-turbo.json
curl -sL $W/weights.safetensors -o whisper-turbo.safetensors
pip install mlx-whisper   # validation oracle; ships the mel filterbank + tiktoken vocab (use a venv)
WA=$(python3 -c 'import mlx_whisper,os;print(os.path.dirname(mlx_whisper.__file__))')/assets
cp "$WA/multilingual.tiktoken" whisper-multilingual.tiktoken
python3 -c "import mlx.core as mx,numpy as np;np.array(mx.load('$WA/mel_filters.npz')['mel_128']).astype('float32').tofile('whisper-mel-filters-128.f32')"
bun whisper.ts audio.flac          # auto-detects language; any ffmpeg-decodable file
```

OLMoE-1B-7B 4-bit setup (the MoE model — weights git-ignored, ~3.9 GB):

```sh
O=https://huggingface.co/mlx-community/OLMoE-1B-7B-0125-Instruct-4bit/resolve/main
curl -sL $O/config.json    -o config-olmoe.json
curl -sL $O/tokenizer.json -o tokenizer-olmoe.json
curl -sL $O/model.safetensors -o model-olmoe.safetensors
python3 split-olmoe.py             # -> model-olmoe-sharded/ (for the sharded-loader test)
bun olmoe.ts "The capital of France is"
```

Note: the original `0924` checkpoint was replaced upstream by `0125` (identical
architecture: 16 layers, 64 experts, group_size 64 / 4-bit). The validate-all
OLMoE checks compare `olmoe.ts` against `reference-olmoe.py` — both load the same
`model-olmoe.safetensors` — so any matching 4-bit checkpoint restores parity.

GPT-2-124M setup (real OpenAI weights — git-ignored, ~550 MB):

```sh
G=https://huggingface.co/openai-community/gpt2/resolve/main
curl -sL $G/config.json    -o config-gpt2.json
curl -sL $G/tokenizer.json -o gpt2-tokenizer.json
curl -sL $G/model.safetensors -o gpt2-model.safetensors
bun gpt2.ts "The capital of France is"   # greedy; TEMP/TOP_K/TOP_P/REP to sample (see GPT2.md)
```

### 🟡 Needs modest code (clear path, no feasibility risk)
- **More architectures** (Llama, Mistral, Gemma, Phi…) — a forward over `nn`
  modules + weight-key mapping (`olmoe.ts` / `whisper.ts` are templates).
- **Better embeddings** — a *dedicated* embedding model (BERT encoder + WordPiece,
  or Qwen3-Embedding with last-token pooling) for stronger retrieval than the
  current mean-pooled base-LLM vectors.
- **npm packaging** — bundle a prebuilt `libmlxc`, replace the hardcoded
  `/opt/homebrew/...` dylib paths with runtime resolution.

### ❌ Not yet (substantial new code or a real gap)
- **Text-to-speech** — **de-risked, not built**: the novel vocoder step (iSTFT,
  spectrum → audio) runs in mlx-c/TS, matching mlx-audio to 1.2e-7
  (`spike-istft.ts`). A full talking pipeline needs the rest of a vocoder
  (conv/norm variants — portable) plus a non-MLX grapheme→phoneme step
  (`espeak-ng`, like ffmpeg). Scoped as product work — see `FINDINGS.md` §7c.
- **Node.js / cross-platform** — Bun FFI only; Apple-Silicon + Metal only (no
  Linux/CUDA, Windows, Intel).
- **Vision / multimodal** (CLIP, LLaVA, image-gen) — no vision encoders wired yet.
- **High-throughput multi-tenant serving** — only equal-length batching; ragged
  prompts need padding masks + continuous batching.
- **Broad model compatibility** — only affine 4-bit quant (no AWQ/GPTQ), so many
  HF quantized checkpoints won't load.
- **Constrained/JSON decoding, beam search, speculative decoding** — none yet.
- **Full/large-scale training** — only minibatch LoRA is proven; per-sample
  gradients need `vmap`, the one genuine mlx-c capability gap.
