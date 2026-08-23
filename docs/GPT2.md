# GPT-2-124M in mlx-ts — real OpenAI weights, our own BPE, token-exact

The capstone of the from-scratch arc ([microGPT](./MICROGPT.md) →
[nanoGPT](./NANOGPT.md) → here): load the **actual OpenAI GPT-2-124M weights**,
tokenize with a **pure-TS GPT-2 BPE encoder**, and generate — **token-exact**
against an MLX-Python reference. This is the "chase GPT-2-124M" milestone in its
achievable form: reproducing GPT-2's *outputs* end-to-end from TypeScript over
`mlx-c`, with every piece (tokenizer, architecture, weights) validated.

## Source / credit
- **Model:** <https://huggingface.co/openai-community/gpt2> (`model.safetensors`, `tokenizer.json`, `config.json`)
- GPT-2 paper "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)

## What we built
**1. A GPT-2 BPE encoder** (the genuinely missing piece). `../src/text/tokenizer.ts` was
already a byte-level BPE encoder/decoder; GPT-2 needed its specific r50k
pretokenization (lowercase contractions, grouped digit runs) — added as
`GPT2_SPLIT`, plus skipping NFC normalization (GPT-2 has no normalizer). Now
token-exact vs HF `tokenizers` (`../tests/gpt2-tok-test.ts`, **8/8** incl. contractions,
digit runs, leading whitespace, unicode, em-dashes).

**2. The GPT-2 model** (`../src/models/gpt2.ts`) — GPT-2's architecture exactly:

| GPT-2 detail | implementation |
|---|---|
| Learned positional embeddings (`wpe`) | gathered by absolute position |
| LayerNorm **with bias** | `fastLayerNorm(x, w, b, eps)` |
| Fused QKV (Conv1D `c_attn`) | weight `[D,3D]` sliced into Q/K/V at load |
| Conv1D weight layout `[in,out]` | `matmul(x, W)` directly (no transpose) |
| `gelu_new` (tanh approx) | `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` via `tanh` |
| Tied `lm_head` | `matmul(h, wteᵀ)` |
| Causal self-attention + KV cache | fused `scaled_dot_product_attention` |

## Run it
```sh
# fetch (git-ignored, like the other model weights):
HF=https://huggingface.co/openai-community/gpt2/resolve/main
curl -sL $HF/config.json    -o config-gpt2.json
curl -sL $HF/tokenizer.json -o gpt2-tokenizer.json
curl -sL $HF/model.safetensors -o gpt2-model.safetensors

bun ../src/models/gpt2.ts "The capital of France is"
python3 ../reference/reference-gpt2.py "The capital of France is"   # MLX-Python oracle
```

### Sampling
Default decode is greedy argmax (deterministic, token-exact vs the oracle), which
makes the 124M model loop ("…capital of the French Republic…"). Set any of
`TEMP` / `TOP_K` / `TOP_P` / `REP` (repetition penalty) to sample — reusing
`../src/core/mx.ts`'s `sample()` + `applyRepetitionPenalty()`:

```sh
TEMP=0.8 TOP_K=40 SEED=1 bun ../src/models/gpt2.ts "Once upon a time, there was a"
#  -> "...chance that you might actually be able to get a good start with your own skills..."
TEMP=0.9 TOP_P=0.95 REP=1.3 SEED=2 bun ../src/models/gpt2.ts "Once upon a time, there was a"
REP=1.3 bun ../src/models/gpt2.ts "Once upon a time, there was a"   # greedy + repetition penalty, no loop
```
Sampling kills the repetition and gives varied, coherent completions. `SEED`
makes a sampled run reproducible.

## Result
```
=== GPT-2-124M (real OpenAI weights) — TS over mlx-c -> Metal ===
prompt:    "The capital of France is"
gen ids:   [262, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139, 286, 262, ...]
completion:" the capital of the French Republic, and the capital of the French Republic is ..."
(24 tokens in 0.11s — 211.5 tok/s)
```
Classic GPT-2 greedy output (fluent, repetitive), at ~210 tok/s. The generated
token ids match `../reference/reference-gpt2.py` **exactly** on every prompt tried — same real
weights, same BPE, same argmax. Wired into `../scripts/validate-all.sh` (guarded on the
weights being present).

## Scope: what "chase GPT-2-124M" means here
There are two senses of reproducing GPT-2-124M:
- **Reproduce its outputs** (load OpenAI's weights, run them) — ✅ **done**, token-exact.
- **Reproduce its *training* from scratch** on OpenWebText — the *recipe* is
  already proven by [nanoGPT](./NANOGPT.md) (the 124M config is the same
  architecture, just bigger; AdamW + cosine + clip + dropout all run from TS).
  What stops a full from-scratch run is **single-Mac wall-clock** (nanoGPT quotes
  ~4 days on 8×A100 over a 40 GB corpus) and a data pipeline, **not** any missing
  capability in the TS↔MLX bridge. With the BPE encoder, the only remaining gap to
  *attempting* it is compute and disk — a practical limit, not a feasibility one.

## Files
- `../src/models/gpt2.ts` — model: load real weights, GPT-2 forward, KV-cache greedy decode
- `../reference/reference-gpt2.py` — MLX-Python oracle for the token-exact check
- `../src/text/tokenizer.ts` — extended with `GPT2_SPLIT` (GPT-2 r50k pretokenization)
- `../tests/gpt2-tok-test.ts` / `../reference/reference-gpt2-tok.py` — BPE encoder parity vs HF tokenizers
- `config-gpt2.json`, `gpt2-tokenizer.json`, `gpt2-model.safetensors` — fetched, git-ignored
