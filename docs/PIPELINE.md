# The full nanochat-style pipeline in mlx-ts — `../scripts/run.sh`

`../scripts/run.sh` is the TypeScript-over-MLX analogue of
[nanochat](https://github.com/karpathy/nanochat)'s `runs/runcpu.sh`: train a
tokenizer, pretrain a GPT from scratch, SFT it into a chat model, and chat with
it — **end to end on one Apple-Silicon machine**.

```sh
bash ../scripts/run.sh
```

Same honest caveat as nanochat's own `runcpu.sh`: *a MacBook won't train a strong
model.* This is an educational demo of the whole pipeline running locally. The
chat model reliably answers the questions it was SFT'd on; held-out quality is
limited by the tiny base and small corpus.

Trains on **TinyStories** by default — a corpus designed so small models learn
genuinely coherent (simple) English, which is what makes a Mac-scale run actually
work (raw web text needs far more params/compute to be coherent).

## Stages

| # | stage | script | engine |
|---|---|---|---|
| 0 | dataset | `../scripts/run.sh` (curl) | bounded prefix of TinyStories (HTTP range request) |
| 1 | `tok_train` | `../reference/tok-train.py` | byte-level BPE in **native Rust** (HF tokenizers) → `tokenizer.json` |
| 2 | `data_prep` | `../training/data-prep.ts` | **stream-encode** the corpus (pure-TS BPE) → uint16 token shards (`tokens-{train,val}.bin`) |
| 3 | `base_train` | `../training/base-train.ts` | pretrain a GPT on the **memmapped** token stream, **save `base-ckpt.safetensors`** — real MLX over FFI |
| 4 | `chat_sft` | `../training/chat-sft.ts` | load the base checkpoint, **batched** SFT (BS examples/step, padded, masked loss on completion tokens only), save `chat-ckpt.safetensors`. **Story-aligned** (`STORIES=<corpus>`): instruction-tunes on *"Tell me a story about {topic}." → {a real story}* so the chat matches a TinyStories base's competence |
| 5 | `chat_cli` | `../training/chat-ckpt.ts` | load the chat checkpoint and generate replies (CLI) |
| 5b | `chat_web` | `../examples/chat-web.ts` | serve the checkpoint behind the OpenAI API + `../examples/chat.html` UI (`bun ../examples/chat-web.ts` → http://localhost:8080) |

Everything except the tokenizer trainer is TypeScript driving MLX. The data
pipeline streams: `../training/data-prep.ts` reads the corpus in chunks, encodes whole lines
with the validated TS tokenizer, and appends uint16 tokens to disk; `../training/base-train.ts`
**`Bun.mmap`s** those shards so training scales past RAM (the OS pages them in).
The shared GPT + checkpoint load/save live in `../src/models/nanogpt-model.ts`; the cross-stage
handoff is plain safetensors checkpoints (writer = `mx.saveSafetensors`, round-trip
verified).

## What it looks like
```
=== [1/4] tok_train ===   trained byte-level BPE: vocab=2048 -> tokenizer-trained.json
=== [2/4] base_train ===  ... FINAL train 4.1  val 4.6 ; saved base-ckpt.safetensors ; CKPT roundtrip: OK
=== [3/4] chat_sft ===    STEP0 5.91 -> FINAL 0.0001 ; saved chat-ckpt.safetensors
=== [4/4] chat ===
Q: What is the capital of France?
A: The capital of France is Paris.
Q: Who are you?
A: I am a small language model trained with mlx-ts.
```

## Recipe notes (what actually made the chat coherent)
Scaling the model up (14M → 28M, more data/iters) improved the *base* prose but
**not** the chatbot — the bottleneck turned out to be the SFT recipe, not size:
- **Batched SFT** is essential. One example per step makes the loss per-example,
  so it swings wildly (2.1 ↔ 4.4) and the model rambles/degrades. Batching `BS`
  padded examples with a masked completion loss averages over many tokens → smooth
  loss (2.64 → 2.27 monotonic) and coherent, complete stories.
- **Gentler SFT LR** (`3e-4`, below the pretrain LR) — high LR on top of batch-1
  variance was what made it ramble.
- **Story-aligned data** — SFT on the base's own competence ("tell me a story"),
  not mismatched trivia, or it collapses to memorized answers.

The theme throughout: at this scale the *training recipe* dominates, not parameters.

## Knobs (env overrides)
`VOCAB` (tokenizer size) · `N_LAYER`/`N_HEAD`/`N_EMBD`/`BLOCK`/`BATCH` (model) ·
`BASE_ITERS` (pretrain steps) · `SFT_ITERS`/`SFT_LR`/`DEVBATCH` (SFT) · `CORPUS`,
`MAX_BYTES` (dataset). Defaults: depth-6 / 384-d / ~14M model, 3000 pretrain +
1500 batched-SFT iters at LR 3e-4. Scale up: `N_LAYER=8 N_EMBD=512 MAX_BYTES=400000000`.

## Pipeline coverage vs nanochat `runcpu.sh`
- ✅ `tok_train`, ✅ `base_train`, ✅ `chat_sft`, ✅ `chat_cli`, ✅ `chat_web`
  (`../examples/chat-web.ts` serves the checkpoint behind the same OpenAI API + `../examples/chat.html` UI)
- Not built (not bridge gaps): `base_eval`'s CORE-score harness, the Muon
  optimizer, multi-GPU scale. RL (GRPO) exists in `../training/rl.ts` (it's in nanochat's
  `speedrun.sh`, not `runcpu.sh`).

## Files
- `../scripts/run.sh` — the pipeline runner
- `../reference/tok-train.py` · `../training/base-train.ts` · `../training/chat-sft.ts` · `../training/chat-ckpt.ts` — the stages
- `../src/models/nanogpt-model.ts` — shared GPT forward + checkpoint load/save + generation
