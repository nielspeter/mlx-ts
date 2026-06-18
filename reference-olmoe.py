"""Reference: real OLMoE-1B-7B (64 experts, top-8, 4-bit) in MLX Python.
Greedy ids must match olmoe.ts. Run: python3 reference-olmoe.py "<prompt>"
"""
import sys, json
import mlx.core as mx
from tokenizers import Tokenizer

cfg = json.load(open("config-olmoe.json"))
D, NL, nH = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
nKV, Dh, E, K = cfg["num_key_value_heads"], D // cfg["num_attention_heads"], cfg["num_experts"], cfg["num_experts_per_tok"]
EPS, THETA, SCALE = cfg["rms_norm_eps"], cfg["rope_theta"], (D // nH) ** -0.5
GS, BITS, EOS, B = cfg["quantization"]["group_size"], cfg["quantization"]["bits"], cfg.get("eos_token_id", 50279), 1
NORM_TOPK = cfg["norm_topk_prob"]

w = mx.load("model-olmoe.safetensors")
tok = Tokenizer.from_file("tokenizer-olmoe.json")

def qmm(x, p): return mx.quantized_matmul(x, w[f"{p}.weight"], w[f"{p}.scales"], w[f"{p}.biases"], transpose=True, group_size=GS, bits=BITS)
def rms(x, n): return mx.fast.rms_norm(x, w[n], EPS)
def stack(li, proj, c): return mx.stack([w[f"model.layers.{li}.mlp.experts.{e}.{proj}.{c}"] for e in range(E)], axis=0)
EXP = [{p: (stack(li, p, "weight"), stack(li, p, "scales"), stack(li, p, "biases")) for p in ("gate_proj", "up_proj", "down_proj")} for li in range(NL)]

def moe(li, x):                                  # x: [T, D]
    T = x.shape[0]
    gates = mx.softmax(qmm(x, f"model.layers.{li}.mlp.gate"), axis=-1)   # [T, E]
    inds = mx.argpartition(-gates, kth=K - 1, axis=-1)[:, :K]            # [T, K]
    wts = mx.take_along_axis(gates, inds, axis=-1)
    if NORM_TOPK: wts = wts / wts.sum(-1, keepdims=True)
    xb = x[:, None, None, :]
    def gqm(proj): return mx.gather_qmm(xb if proj != "down_proj" else h, *EXP[li][proj], rhs_indices=inds, transpose=True, group_size=GS, bits=BITS)
    g = gqm("gate_proj"); u = gqm("up_proj"); h = (g * mx.sigmoid(g)) * u
    o = mx.gather_qmm(h, *EXP[li]["down_proj"], rhs_indices=inds, transpose=True, group_size=GS, bits=BITS)
    return (o.squeeze(2) * wts[..., None]).sum(axis=1)                   # [T, D]

cache = [None] * NL
def block(li, h, T, off):
    p = f"model.layers.{li}"
    y = rms(h, f"{p}.input_layernorm.weight")
    q = rms(qmm(y, f"{p}.self_attn.q_proj"), f"{p}.self_attn.q_norm.weight").reshape(B, T, nH, Dh).transpose(0, 2, 1, 3)
    k = rms(qmm(y, f"{p}.self_attn.k_proj"), f"{p}.self_attn.k_norm.weight").reshape(B, T, nKV, Dh).transpose(0, 2, 1, 3)
    v = qmm(y, f"{p}.self_attn.v_proj").reshape(B, T, nKV, Dh).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, Dh, traditional=False, base=THETA, scale=1.0, offset=off)
    k = mx.fast.rope(k, Dh, traditional=False, base=THETA, scale=1.0, offset=off)
    if cache[li] is not None: k = mx.concatenate([cache[li][0], k], axis=2); v = mx.concatenate([cache[li][1], v], axis=2)
    cache[li] = (k, v)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal" if T > 1 else None).transpose(0, 2, 1, 3).reshape(B, T, nH * Dh)
    h = h + qmm(o, f"{p}.self_attn.o_proj")
    return h + moe(li, rms(h, f"{p}.post_attention_layernorm.weight").reshape(T, D)).reshape(B, T, D)

def embed(ids):
    e = "model.embed_tokens"
    return mx.dequantize(mx.take(w[f"{e}.weight"], ids, axis=0), mx.take(w[f"{e}.scales"], ids, axis=0), mx.take(w[f"{e}.biases"], ids, axis=0), group_size=GS, bits=BITS)

def step(ids, off):
    T = len(ids); h = embed(mx.array(ids, dtype=mx.int32).reshape(B, T))
    for li in range(NL): h = block(li, h, T, off)
    h = rms(h, "model.norm.weight")
    last = mx.take(h, mx.array([T - 1], dtype=mx.int32), axis=1).reshape(B, D)
    tk = int(mx.argmax(qmm(last, "lm_head"), axis=1).item())
    mx.eval(*[t for c in cache if c for t in c])
    return tk

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
ids = tok.encode(prompt).ids
tk = step(ids, 0); pos = len(ids); gen = []
for i in range(40):
    if tk == EOS: break
    gen.append(tk); tk = step([tk], pos); pos += 1
print("gen ids:   ", gen)
print("completion:", repr(tok.decode(gen)))
