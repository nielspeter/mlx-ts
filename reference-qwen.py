"""Reference: real Qwen3-0.6B forward in MLX Python, same prompt, greedy decode.
The generated token ids must match qwen.ts exactly (same real weights).
Run: python3 reference-qwen.py "The capital of France is"
"""
import sys, json
import mlx.core as mx
from tokenizers import Tokenizer

cfg = json.load(open("config.json"))
D, NL = cfg["hidden_size"], cfg["num_hidden_layers"]
nH, nKV, Dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
EPS, THETA, SCALE, B = cfg["rms_norm_eps"], cfg["rope_theta"], cfg["head_dim"] ** -0.5, 1
EOS = cfg["eos_token_id"]

w = mx.load("model-qwen.safetensors")
tok = Tokenizer.from_file("tokenizer.json")

embed = w["model.embed_tokens.weight"]
def lin(x, name): return mx.matmul(x, w[name].T)   # [out,in] -> x @ W.T
cache = [None] * NL


def block(li, h, Lc, offset):
    p = f"model.layers.{li}"
    y = mx.fast.rms_norm(h, w[f"{p}.input_layernorm.weight"], EPS)
    q = mx.fast.rms_norm(lin(y, f"{p}.self_attn.q_proj.weight").reshape(B, Lc, nH, Dh), w[f"{p}.self_attn.q_norm.weight"], EPS).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(lin(y, f"{p}.self_attn.k_proj.weight").reshape(B, Lc, nKV, Dh), w[f"{p}.self_attn.k_norm.weight"], EPS).transpose(0, 2, 1, 3)
    v = lin(y, f"{p}.self_attn.v_proj.weight").reshape(B, Lc, nKV, Dh).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)
    k = mx.fast.rope(k, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)
    if cache[li] is not None:
        k = mx.concatenate([cache[li][0], k], axis=2)
        v = mx.concatenate([cache[li][1], v], axis=2)
    cache[li] = (k, v)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal" if Lc > 1 else None)
    o = o.transpose(0, 2, 1, 3).reshape(B, Lc, nH * Dh)
    h = h + lin(o, f"{p}.self_attn.o_proj.weight")
    y2 = mx.fast.rms_norm(h, w[f"{p}.post_attention_layernorm.weight"], EPS)
    g = lin(y2, f"{p}.mlp.gate_proj.weight")
    act = (g * mx.sigmoid(g)) * lin(y2, f"{p}.mlp.up_proj.weight")   # silu(g) * up
    return h + mx.matmul(act, w[f"{p}.mlp.down_proj.weight"].T)


def step(ids, offset):
    Lc = len(ids)
    h = mx.take(embed, mx.array(ids, dtype=mx.int32).reshape(B, Lc), axis=0)
    for li in range(NL):
        h = block(li, h, Lc, offset)
    h = mx.fast.rms_norm(h, w["model.norm.weight"], EPS)
    h_last = mx.take(h, mx.array([Lc - 1], dtype=mx.int32), axis=1)
    logits = mx.matmul(h_last, embed.T)            # tied lm_head
    tk = int(mx.argmax(logits, axis=2).item())
    mx.eval(*[t for c in cache if c for t in c])
    return tk


prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
ids = tok.encode(prompt).ids
gen = []
tk = step(ids, 0)
pos = len(ids)
for i in range(24):
    if tk == EOS:
        break
    gen.append(tk)
    tk = step([tk], pos)
    pos += 1

print("=== Qwen3-0.6B (real weights) — MLX Python reference ===")
print("prompt ids:", ids)
print("gen ids:   ", gen)
print("completion:", repr(tok.decode(gen)))
