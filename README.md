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
```

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

mlx-ts today is a **local LLM inference + LoRA-training runtime for decoder-only
text models** — Bun-only, Apple-Silicon-only, run as scripts in this repo (not
yet an npm package). Sampling supports greedy, temperature, top-p, **top-k**, and
**repetition penalty** (`bun stream.ts --temp 0.8 --topp 0.95 --topk 40 --reppenalty 1.1 "…"`).

### ✅ Buildable now (everything needed exists)
- **Local chat assistant / CLI** — streaming replies, multi-turn via chat
  templates, temp/top-p/top-k/repetition-penalty sampling (`chat.ts`, `stream.ts`).
- **A local inference HTTP server + chat UI** — `server.ts` does this: an
  OpenAI-compatible `/v1/chat/completions` (SSE/JSON) over `Bun.serve` plus a
  self-contained chat page at `/`. Single-process / low-concurrency, not multi-tenant.
- **Local RAG** — `POST /v1/embeddings` returns L2-normalized sentence vectors
  (mean-pooled Qwen3 hidden states); pair with any JS vector store for retrieval.
- **Prompt-driven text tools** — summarize / rewrite / classify / extract /
  translate, batched over a dataset (one-at-a-time or equal-length).
- **Agent loops** — tool use via prompting + JS-side parsing.
- **LoRA fine-tuning** of 4-bit Qwen3 (Adam + cross-entropy, `lora-train.ts`).
- **Research / inspection** — pull logits, hidden states; the `MX` op surface is open.

### 🟡 Needs modest code (clear path, no feasibility risk)
- **More architectures** (Llama, Mistral, Gemma, Phi…) — a forward over `nn`
  modules + weight-key mapping (`olmoe.ts` is the template).
- **Better embeddings** — a *dedicated* embedding model (e.g. a BERT encoder with
  WordPiece, or Qwen3-Embedding with last-token pooling) for stronger retrieval
  than the current mean-pooled base-LLM vectors.
- **npm packaging** — bundle a prebuilt `libmlxc`, replace the hardcoded
  `/opt/homebrew/...` dylib paths with runtime resolution.
- **OpenAI-compatible HTTP server** — app-level glue over `streamText`.

### ❌ Not yet (substantial new code or a real gap)
- **Node.js / cross-platform** — Bun FFI only; Apple-Silicon + Metal only (no
  Linux/CUDA, Windows, Intel).
- **Speech-to-text (Whisper)** — the **log-Mel audio front-end is built and
  validated** (`audio.ts`: ffmpeg decode + rfft-as-matmul mel, matches numpy FFT
  to ~1e-6). What remains is the encoder–decoder model (Conv1d stem — already
  wrapped — bidirectional encoder, causal decoder with cross-attention), the
  Whisper tokenizer, and the weights.
- **Vision / multimodal** (CLIP, LLaVA, image-gen) — no vision encoders wired yet.
- **High-throughput multi-tenant serving** — only equal-length batching; ragged
  prompts need padding masks + continuous batching.
- **Broad model compatibility** — only affine 4-bit quant (no AWQ/GPTQ), so many
  HF quantized checkpoints won't load.
- **Constrained/JSON decoding, beam search, speculative decoding** — none yet.
- **Full/large-scale training** — only minibatch LoRA is proven; per-sample
  gradients need `vmap`, the one genuine mlx-c capability gap.
