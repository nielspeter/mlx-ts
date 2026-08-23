"""Reference: real GPT-2-124M forward in MLX Python, same prompt, greedy decode.
The generated token ids must match gpt2.ts exactly (same real OpenAI weights).
Run: python3 reference-gpt2.py "The capital of France is"
"""
import sys, json
import mlx.core as mx
from tokenizers import Tokenizer

cfg = json.load(open("models/config-gpt2.json"))
D, NL, nH = cfg["n_embd"], cfg["n_layer"], cfg["n_head"]
Dh, EPS, SCALE, B, EOS = D // nH, cfg["layer_norm_epsilon"], (D // nH) ** -0.5, 1, 50256

w = mx.load("models/gpt2-model.safetensors")
tok = Tokenizer.from_file("models/gpt2-tokenizer.json")
wte, wpe = w["wte.weight"], w["wpe.weight"]

def gelu_new(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

cache = [None] * NL

def block(li, h, Lc):
    p = f"h.{li}"
    x1 = mx.fast.layer_norm(h, w[f"{p}.ln_1.weight"], w[f"{p}.ln_1.bias"], EPS)
    qkv = x1 @ w[f"{p}.attn.c_attn.weight"] + w[f"{p}.attn.c_attn.bias"]   # [B,Lc,3D]
    q, k, v = mx.split(qkv, 3, axis=-1)
    head = lambda t: t.reshape(B, Lc, nH, Dh).transpose(0, 2, 1, 3)
    q, k, v = head(q), head(k), head(v)
    if cache[li] is not None:
        k = mx.concatenate([cache[li][0], k], axis=2)
        v = mx.concatenate([cache[li][1], v], axis=2)
    cache[li] = (k, v)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=("causal" if Lc > 1 else None))
    o = o.transpose(0, 2, 1, 3).reshape(B, Lc, D)
    h = h + (o @ w[f"{p}.attn.c_proj.weight"] + w[f"{p}.attn.c_proj.bias"])
    x2 = mx.fast.layer_norm(h, w[f"{p}.ln_2.weight"], w[f"{p}.ln_2.bias"], EPS)
    ff = gelu_new(x2 @ w[f"{p}.mlp.c_fc.weight"] + w[f"{p}.mlp.c_fc.bias"]) @ w[f"{p}.mlp.c_proj.weight"] + w[f"{p}.mlp.c_proj.bias"]
    return h + ff

def step(ids, offset):
    Lc = len(ids)
    h = wte[mx.array(ids)].reshape(B, Lc, D) + wpe[mx.array([offset + i for i in range(Lc)])]
    for li in range(NL):
        h = block(li, h, Lc)
    h = mx.fast.layer_norm(h, w["ln_f.weight"], w["ln_f.bias"], EPS)
    logits = h[:, -1, :] @ wte.T
    tk = int(mx.argmax(logits, axis=-1).item())
    mx.eval([a for c in cache if c for a in c])
    return tk

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
N_NEW = 24
prompt_ids = tok.encode(prompt).ids
gen = []
tk = step(prompt_ids, 0)
pos = len(prompt_ids)
for _ in range(N_NEW):
    if tk == EOS:
        break
    gen.append(tk)
    tk = step([tk], pos)
    pos += 1

print("=== GPT-2-124M (real weights) — MLX Python reference ===")
print(f"prompt ids: [{', '.join(map(str, prompt_ids))}]")
print(f"gen ids:    [{', '.join(map(str, gen))}]")
print(f"completion: {json.dumps(tok.decode(gen))}")
