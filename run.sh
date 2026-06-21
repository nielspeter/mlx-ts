#!/usr/bin/env bash
# nanochat-style pipeline on Apple Silicon, in TypeScript over MLX — the
# TS analogue of nanochat/runs/runcpu.sh: tokenizer -> pretrain -> SFT -> chat.
#
# NOTE (same caveat as nanochat's runcpu.sh): a MacBook won't train a strong
# model. This is an educational, end-to-end demo of the whole pipeline running
# on one machine, all TS-over-MLX (tokenizer training is native Rust). The chat
# model will reliably answer the questions it was SFT'd on; held-out quality is
# limited by the tiny base. Override any stage via env vars (see defaults below).
#
#   bash run.sh
set -e
cd "$(dirname "$0")"

# 0. corpus — tiny-shakespeare by default; point CORPUS at any UTF-8 text file
[ -f input.txt ] || curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt

# 1. tok_train — train a byte-level BPE in native Rust (HF tokenizers, like nanochat)
echo "=== [1/4] tok_train ==="
VOCAB=${VOCAB:-2048} python3 tok-train.py

# 2. base_train — pretrain a GPT from scratch on BPE tokens; saves base-ckpt.safetensors
echo "=== [2/4] base_train ==="
N_LAYER=${N_LAYER:-6} N_HEAD=${N_HEAD:-6} N_EMBD=${N_EMBD:-384} BLOCK=${BLOCK:-128} \
  BATCH=${BATCH:-16} ITERS=${BASE_ITERS:-1500} bun base-train.ts

# 3. chat_sft — load the base checkpoint and SFT it into a chat model
echo "=== [3/4] chat_sft ==="
ITERS=${SFT_ITERS:-400} bun chat-sft.ts

# 4. chat — talk to it
echo "=== [4/4] chat ==="
bun chat-ckpt.ts "What is the capital of France?"
bun chat-ckpt.ts "Who are you?"
echo
echo "Done. Chat more with:  bun chat-ckpt.ts \"<your question>\""
