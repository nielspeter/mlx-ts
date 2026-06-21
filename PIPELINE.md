# The full nanochat-style pipeline in mlx-ts — `run.sh`

`run.sh` is the TypeScript-over-MLX analogue of
[nanochat](https://github.com/karpathy/nanochat)'s `runs/runcpu.sh`: train a
tokenizer, pretrain a GPT from scratch, SFT it into a chat model, and chat with
it — **end to end on one Apple-Silicon machine**.

```sh
bash run.sh
```

Same honest caveat as nanochat's own `runcpu.sh`: *a MacBook won't train a strong
model.* This is an educational demo of the whole pipeline running locally. The
chat model reliably answers the questions it was SFT'd on; held-out quality is
limited by the tiny base and small corpus.

## Stages

| # | stage | script | engine |
|---|---|---|---|
| 1 | `tok_train` | `tok-train.py` | byte-level BPE in **native Rust** (HF tokenizers) → `tokenizer.json` |
| 2 | `base_train` | `base-train.ts` | pretrain a GPT on BPE tokens, **save `base-ckpt.safetensors`** — real MLX over FFI |
| 3 | `chat_sft` | `chat-sft.ts` | load the base checkpoint, SFT (completion-only loss), save `chat-ckpt.safetensors` |
| 4 | `chat_cli` | `chat-ckpt.ts` | load the chat checkpoint and generate replies (CLI) |
| 4b | `chat_web` | `chat-web.ts` | serve the checkpoint behind the OpenAI API + `chat.html` UI (`bun chat-web.ts` → http://localhost:8080) |

Everything except the tokenizer trainer is TypeScript driving MLX. The pieces
(`nanogpt-model.ts` for the shared GPT + checkpoint load/save) are the same ones
validated throughout the repo; the cross-stage handoff is plain safetensors
checkpoints (the writer is `mx.saveSafetensors`, round-trip verified).

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

## Knobs (env overrides)
`VOCAB` (tokenizer size) · `N_LAYER`/`N_HEAD`/`N_EMBD`/`BLOCK`/`BATCH` (model) ·
`BASE_ITERS` (pretrain steps) · `SFT_ITERS` (SFT steps) · `CORPUS` (text file).
Defaults: depth-6 / 384-d model, 1500 pretrain + 400 SFT iters.

## Pipeline coverage vs nanochat `runcpu.sh`
- ✅ `tok_train`, ✅ `base_train`, ✅ `chat_sft`, ✅ `chat_cli`, ✅ `chat_web`
  (`chat-web.ts` serves the checkpoint behind the same OpenAI API + `chat.html` UI)
- Not built (not bridge gaps): `base_eval`'s CORE-score harness, the Muon
  optimizer, multi-GPU scale. RL (GRPO) exists in `rl.ts` (it's in nanochat's
  `speedrun.sh`, not `runcpu.sh`).

## Files
- `run.sh` — the pipeline runner
- `tok-train.py` · `base-train.ts` · `chat-sft.ts` · `chat-ckpt.ts` — the stages
- `nanogpt-model.ts` — shared GPT forward + checkpoint load/save + generation
