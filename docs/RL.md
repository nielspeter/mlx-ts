# RL in mlx-ts — GRPO on real GPT-2-124M, from TypeScript

The **nanochat RL stage** ([nanochat](https://github.com/karpathy/nanochat)):
**GRPO** (Group Relative Policy Optimization — the simple, critic-free RL used by
DeepSeek/nanochat) on real GPT-2-124M, the policy gradient driven by MLX
`value_and_grad` over FFI.

## How GRPO works (and how it reduces here)
1. **Rollout** — for each prompt, sample a *group* of G completions from the
   current policy (stochastic decoding).
2. **Reward** — score each completion with a verifiable reward.
3. **Group-relative advantage** — normalize within the group:
   `adv_i = (reward_i − mean) / (std + ε)`. No value network/critic.
4. **Policy gradient** — the GRPO objective reduces to **advantage-weighted
   cross-entropy**: `loss = mean_seq( adv_seq · CE(completion) )`. Positive
   advantage → minimize NLL → reinforce that completion; negative → discourage it.

So the RL update reuses the already-validated `crossEntropy` (stable log_softmax),
weighted by a host-computed advantage — the novel parts are the rollout, the
reward, and the group normalization.

## Task: positivity steering (RLHF-flavored, verifiable)
Reward = number of positive words in the completion (a word-list match — fully
verifiable, no reward model). The policy learns to complete more positively from
the reward signal alone, no labeled completions. (A toy stand-in for nanochat's
GSM8K-style verifiable rewards.)

## Run it
```sh
bun rl.ts                  # needs GPT-2 weights (see GPT2.md / README)
CHECK=1 bun rl.ts          # deterministic GRPO-loss check (matches reference-rl.py)
```

## Result
```
=== RL (GRPO) on GPT-2-124M — positivity reward, group size 8 ===
  step  0: mean reward 0.225
  step 12: mean reward 0.800
  step 24: mean reward 2.075        # ~9x over 25 steps

--- sample completions after RL ---
The movie was terrific, and just a massive pleasure to be involved.
I think the food was amazing and the service was great for the price.
My day today was great. I had some great food on the way up.
```
Mean reward rises ~9× and the completions become clearly positive — the policy
improved from a scalar reward signal, with no supervised targets.

## Validation
Rollouts are random, so the *trajectory* isn't reproducible — but the **loss path
is**. `CHECK=1 bun rl.ts` computes the GRPO loss (forward + CE + advantage
weighting) on a *fixed* batch and `reference-rl.py` computes the same in MLX
Python: **RLLOSS = −0.03563 on both**. Wired into `validate-all.sh` (guarded on
the GPT-2 weights). The reward/advantage math is pure host arithmetic; the
gradient-bearing part is validated cross-entropy.

## Where this lands
RL was the last unbuilt nanochat stage. mlx-ts now covers the whole pipeline shape
end to end — **tokenizer (BPE) → pretrain (`spike-nanogpt.ts`) → SFT (`sft.ts`) →
RL (`rl.ts`) → KV-cache inference (`gpt2.ts`) → chat server + web UI
(`server.ts`)** — all TypeScript over MLX, each stage validated against an
MLX-Python mirror. What's not built: a Muon optimizer, BPE tokenizer *training*
(merge counting — not MLX compute), and multi-GPU scale. None are bridge gaps.

## Files
- `rl.ts` — GRPO: rollout + reward + group-relative advantage + policy-gradient update
- `reference-rl.py` — MLX-Python oracle for the GRPO loss path
