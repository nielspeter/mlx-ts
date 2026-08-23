"""Oracle for rl.ts CHECK mode: the GRPO loss (advantage-weighted NLL) on a FIXED
batch, in MLX Python with the same GPT-2 weights. Rollouts are random, but the
loss path (forward + stable log_softmax CE + advantage weighting) is deterministic
and must match rl.ts. Run: CHECK=1 bun rl.ts ; python3 reference-rl.py
"""
import json
import mlx.core as mx
from tokenizers import Tokenizer

cfg = json.load(open("models/config-gpt2.json"))
D, NL, nH = cfg["n_embd"], cfg["n_layer"], cfg["n_head"]
Dh, EPS, SCALE, VOCAB = D // nH, cfg["layer_norm_epsilon"], (D // nH) ** -0.5, cfg["vocab_size"]
tok = Tokenizer.from_file("models/gpt2-tokenizer.json")
w = mx.load("models/gpt2-model.safetensors")

def gelu_new(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

def forward(ids, L):
    x = w["wte.weight"][ids] + w["wpe.weight"][mx.arange(L)]
    hd = lambda t: t.reshape(1, L, nH, Dh).transpose(0, 2, 1, 3)
    for i in range(NL):
        p = f"h.{i}"
        n1 = mx.fast.layer_norm(x, w[f"{p}.ln_1.weight"], w[f"{p}.ln_1.bias"], EPS)
        q, k, v = mx.split(n1 @ w[f"{p}.attn.c_attn.weight"] + w[f"{p}.attn.c_attn.bias"], 3, axis=-1)
        att = mx.fast.scaled_dot_product_attention(hd(q), hd(k), hd(v), scale=SCALE, mask="causal")
        x = x + (att.transpose(0, 2, 1, 3).reshape(1, L, D) @ w[f"{p}.attn.c_proj.weight"] + w[f"{p}.attn.c_proj.bias"])
        n2 = mx.fast.layer_norm(x, w[f"{p}.ln_2.weight"], w[f"{p}.ln_2.bias"], EPS)
        x = x + (gelu_new(n2 @ w[f"{p}.mlp.c_fc.weight"] + w[f"{p}.mlp.c_fc.bias"]) @ w[f"{p}.mlp.c_proj.weight"] + w[f"{p}.mlp.c_proj.bias"])
    return mx.fast.layer_norm(x, w["ln_f.weight"], w["ln_f.bias"], EPS) @ w["wte.weight"].T

def ce(logits, tgt):
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.take_along_axis(logp, tgt, axis=1).mean()

P = "The movie was"
fixed = [(" great and wonderful", 1.0), (" terrible and awful", -1.0)]
total = 0.0
for c, adv in fixed:
    pIds = tok.encode(P).ids
    comp = tok.encode(c).ids
    ids = pIds + comp
    L, cStart = len(ids), len(pIds) - 1
    logits = forward(mx.array(ids), L).reshape(L, VOCAB)[cStart:L - 1]
    total = total + ce(logits, mx.array(comp).reshape(-1, 1)) * adv
loss = float(total / len(fixed))
print(f"RLLOSS={loss:.5f}")
