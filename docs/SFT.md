# SFT in mlx-ts — turn real GPT-2-124M into a chatbot, from TypeScript

The **nanochat "chat" stage** ([nanochat](https://github.com/karpathy/nanochat)),
on real OpenAI GPT-2-124M: supervised fine-tuning that takes the pretrained base
model and teaches it to follow instructions — a **full fine-tune of all 124M
parameters**, the autograd being real MLX `value_and_grad` over FFI.

## What makes it SFT (not just more training)
1. **Chat formatting** — examples are wrapped `User: {q}\nAssistant: {a}<|endoftext|>`.
2. **Completion-only loss** — the loss is computed *only on the assistant's
   response tokens*, never the prompt. Because the response is a contiguous
   suffix, we slice the logits to the response range and use plain cross-entropy
   (cleaner than masking, and avoids an `inf×0` NaN at saturated prompt positions).

## Run it
```sh
bun ../training/sft.ts                 # needs the GPT-2 weights (see GPT2.md / README)
python3 ../reference/reference-sft.py   # MLX-Python oracle
```

## Result
```
=== SFT GPT-2-124M (124.4M params, full fine-tune) — real MLX autograd over FFI ===
before: "The capital of France.\nAssistant: The capital of France.\nAssistant: ..."   (base GPT-2: loops)
  iter   0: loss 1.5514
  iter  40: loss 0.0002
STEP0 loss=1.5514 / FINAL loss=0.0000

--- after SFT ---
Q: What is the capital of France?   A: The capital of France is Paris.   (trained)
Q: What is 2 plus 2?                A: 2 plus 2 equals 4.                 (trained)
Q: What is the capital of Italy?    A: The capital of Italy is Rome.      (HELD-OUT)
```
The base model just loops; after SFT it answers cleanly in the chat format — and
on a **held-out** question (Italy, never in the training set) it produces the
right answer, showing it learned the *format* from SFT and supplied the *fact*
from GPT-2's pretraining. (With only 6 examples this is a demo: it memorizes the
training answers and generalizes the format; real SFT uses far more data.)

## Validation
`../reference/reference-sft.py` runs the identical SFT (same loaded weights, same chat data,
same completion-sliced loss, same AdamW + grad-clip): **step-0 loss matches
exactly (1.5514)** and both converge to ~0. Wired into `../scripts/validate-all.sh` (28/28,
guarded on the GPT-2 weights).

## Two findings worth noting
- **A latent bug in the shared `crossEntropy`.** It used `softmax(x).log()`, whose
  *backward* produces NaN once probabilities saturate — which SFT hits immediately
  (the model memorizes an example, loss → 0, gradients → NaN). Fixed with stable
  `log_softmax` (`x − logsumexp(x)`); the nanoGPT/LoRA oracles were updated to the
  same form so their parity holds. Pretraining never tripped it (loss never
  approaches 0 on real data); SFT-on-tiny-data exposed it.
- **Gradient clipping is load-bearing for full FT.** Without the global grad-norm
  clip, full fine-tuning of 124M destabilizes within ~20 steps (Adam's large
  bias-corrected early steps); with clip=1.0 it's stable.

## Scope
This is the SFT *stage* of a nanochat-style pipeline. The pieces around it already
exist in mlx-ts: pretraining (`../spikes/spike-nanogpt.ts`), KV-cache inference
(`../src/models/gpt2.ts`), and an OpenAI-compatible chat server + web UI (`../examples/server.ts`). What's
not built: tokenizer *training*, a Muon optimizer, and an RL stage — each just
MLX ops + orchestration, no gap in the TS↔MLX bridge.

## Files
- `../training/sft.ts` — load GPT-2 into a trainable params tree, completion-loss SFT loop, before/after generation
- `../reference/reference-sft.py` — MLX-Python oracle for the step-0 + convergence check
- shared: `MX.tanh` / `MX.slice` / `MX.logsumexp` (`../src/core/mx.ts`), stable `crossEntropy` (`../src/nn/loss.ts`)
