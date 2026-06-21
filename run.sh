#!/usr/bin/env bash
# nanochat-style pipeline on Apple Silicon, in TypeScript over MLX — the TS
# analogue of nanochat/runs/runcpu.sh: dataset -> tokenizer -> pretrain -> SFT ->
# chat. Trains a small GPT from scratch on TinyStories (a corpus designed so that
# small models learn genuinely coherent English) so the result actually works.
#
# Only the BPE tokenizer trainer is native (Rust, like nanochat); everything else
# is TypeScript driving MLX. The data pipeline streams: stream-encode the corpus
# to uint16 token shards, then base_train memmaps them (scales past RAM).
#
#   bash run.sh                  # ~15-20 min on an M-series Mac (defaults below)
#   MAX_BYTES=20000000 BASE_ITERS=800 bash run.sh   # quicker, rougher
set -e
cd "$(dirname "$0")"

CORPUS=${CORPUS:-tinystories.txt}
MAX_BYTES=${MAX_BYTES:-200000000}    # bounded prefix of the dataset (~200 MB)
URL=${CORPUS_URL:-https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt}

echo "=== [0/5] dataset ==="
[ -f "$CORPUS" ] || curl -L -r 0-$((MAX_BYTES - 1)) "$URL" -o "$CORPUS"
echo "corpus: $CORPUS ($(wc -c < "$CORPUS") bytes)"

echo "=== [1/5] tok_train (native Rust BPE) ==="
CORPUS=$CORPUS VOCAB=${VOCAB:-8192} python3 tok-train.py

echo "=== [2/5] data_prep (stream-encode -> memmappable uint16 token shards) ==="
CORPUS=$CORPUS TOKENS=${TOKENS:-tokens} bun data-prep.ts

echo "=== [3/5] base_train (pretrain on the token stream + save checkpoint) ==="
TOKENS=${TOKENS:-tokens} N_LAYER=${N_LAYER:-6} N_HEAD=${N_HEAD:-6} N_EMBD=${N_EMBD:-384} \
  BLOCK=${BLOCK:-256} BATCH=${BATCH:-32} ITERS=${BASE_ITERS:-3000} bun base-train.ts

echo "=== [4/5] chat_sft (load base checkpoint, SFT into a chat model) ==="
ITERS=${SFT_ITERS:-400} bun chat-sft.ts

echo "=== [5/5] chat ==="
bun chat-ckpt.ts "Tell me a story about a cat."
bun chat-ckpt.ts "Who are you?"
echo
echo "Done. CLI:  bun chat-ckpt.ts \"<question>\"    Web UI:  bun chat-web.ts  -> http://localhost:8080"
