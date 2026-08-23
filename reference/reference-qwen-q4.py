"""Reference for the real 4-bit Qwen3-0.6B (mlx-community format), MLX Python.
Greedy ids must match qwen-nn.ts. Run: python3 reference-qwen-q4.py "<prompt>"
"""
import sys, json
import mlx.core as mx
from tokenizers import Tokenizer

cfg = json.load(open("models/config-4bit.json"))
D, NL = cfg["hidden_size"], cfg["num_hidden_layers"]
nH, nKV, Dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
EPS, THETA, SCALE = cfg["rms_norm_eps"], cfg["rope_theta"], cfg["head_dim"] ** -0.5
GS, BITS, EOS, B = cfg["quantization"]["group_size"], cfg["quantization"]["bits"], cfg["eos_token_id"], 1

w = mx.load("models/model-q4.safetensors")
tok = Tokenizer.from_file("models/tokenizer.json")
cache = [None] * NL

def qmm(x, p):  # x @ dequant(W).T
    return mx.quantized_matmul(x, w[f"{p}.weight"], w[f"{p}.scales"], w[f"{p}.biases"], transpose=True, group_size=GS, bits=BITS)

def rms(x, name):
    return mx.fast.rms_norm(x, w[name], EPS)

def embed(ids):
    e = "model.embed_tokens"
    wr = mx.take(w[f"{e}.weight"], ids, axis=0)
    sr = mx.take(w[f"{e}.scales"], ids, axis=0)
    br = mx.take(w[f"{e}.biases"], ids, axis=0)
    return mx.dequantize(wr, sr, br, group_size=GS, bits=BITS)

def block(li, h, L, offset):
    p = f"model.layers.{li}"
    y = rms(h, f"{p}.input_layernorm.weight")
    q = rms(qmm(y, f"{p}.self_attn.q_proj").reshape(B, L, nH, Dh), f"{p}.self_attn.q_norm.weight").transpose(0, 2, 1, 3)
    k = rms(qmm(y, f"{p}.self_attn.k_proj").reshape(B, L, nKV, Dh), f"{p}.self_attn.k_norm.weight").transpose(0, 2, 1, 3)
    v = qmm(y, f"{p}.self_attn.v_proj").reshape(B, L, nKV, Dh).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)
    k = mx.fast.rope(k, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)
    if cache[li] is not None:
        k = mx.concatenate([cache[li][0], k], axis=2); v = mx.concatenate([cache[li][1], v], axis=2)
    cache[li] = (k, v)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal" if L > 1 else None)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, nH * Dh)
    h = h + qmm(o, f"{p}.self_attn.o_proj")
    y2 = rms(h, f"{p}.post_attention_layernorm.weight")
    g = qmm(y2, f"{p}.mlp.gate_proj")
    h = h + qmm((g * mx.sigmoid(g)) * qmm(y2, f"{p}.mlp.up_proj"), f"{p}.mlp.down_proj")
    return h

def step(ids, offset):
    L = len(ids)
    h = embed(mx.array(ids, dtype=mx.int32).reshape(B, L))
    for li in range(NL):
        h = block(li, h, L, offset)
    h = rms(h, "model.norm.weight")
    last = mx.take(h, mx.array([L - 1], dtype=mx.int32), axis=1).reshape(B, D)
    logits = mx.quantized_matmul(last, w["model.embed_tokens.weight"], w["model.embed_tokens.scales"], w["model.embed_tokens.biases"], transpose=True, group_size=GS, bits=BITS)
    tk = int(mx.argmax(logits, axis=1).item())
    mx.eval(*[t for c in cache if c for t in c])
    return tk

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
ids = tok.encode(prompt).ids
tk = step(ids, 0); pos = len(ids); gen = []
for i in range(48):
    if tk == EOS: break
    gen.append(tk); tk = step([tk], pos); pos += 1
print("gen ids:   ", gen)
print("completion:", repr(tok.decode(gen)))
