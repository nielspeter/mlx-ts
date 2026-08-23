"""4-bit quantized Qwen3: quantize the Linear projections with mx.quantize,
save to models/model-quant.safetensors, and run KV-cache greedy decode using
mx.quantized_matmul as the reference. model-quant.ts must produce the same ids.

Norms + embedding stay full precision (as in real mlx-community models).
Run: python3 reference-quant.py
"""
import mlx.core as mx

VOCAB, D, nH, nKV, Dh, I, LAYERS = 32, 64, 4, 2, 16, 128, 2
EPS, THETA, SCALE, B = 1e-6, 1_000_000.0, Dh ** -0.5, 1
GS, BITS = 64, 4

save = {}

def det(n, seed):
    return mx.array([(((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.1
                     for i in range(n)], dtype=mx.float32)

def W(out_, in_, seed):
    return det(out_ * in_, seed).reshape([out_, in_])   # standard Linear weight [out, in]

def fp(name, arr):
    save[name] = arr
    return arr

def quant(name, out_, in_, seed):
    wq, scales, biases = mx.quantize(W(out_, in_, seed), GS, BITS)
    save[f"{name}.weight"], save[f"{name}.scales"], save[f"{name}.biases"] = wq, scales, biases
    return (wq, scales, biases)

def qmm(x, packed):
    wq, scales, biases = packed
    return mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=GS, bits=BITS)

embed = fp("embed", det(VOCAB * D, 0).reshape([VOCAB, D]))
finalNorm = fp("finalNorm", det(D, 900))
lmHead = quant("lmHead", VOCAB, D, 901)

layers = []
for l in range(LAYERS):
    s = lambda k: 100 + l * 20 + k
    layers.append(dict(
        inNorm=fp(f"layers.{l}.inNorm", det(D, s(1))),
        qNorm=fp(f"layers.{l}.qNorm", det(Dh, s(5))),
        kNorm=fp(f"layers.{l}.kNorm", det(Dh, s(6))),
        postNorm=fp(f"layers.{l}.postNorm", det(D, s(8))),
        Wq=quant(f"layers.{l}.Wq", nH * Dh, D, s(2)),
        Wk=quant(f"layers.{l}.Wk", nKV * Dh, D, s(3)),
        Wv=quant(f"layers.{l}.Wv", nKV * Dh, D, s(4)),
        Wo=quant(f"layers.{l}.Wo", D, nH * Dh, s(7)),
        Wgate=quant(f"layers.{l}.Wgate", I, D, s(9)),
        Wup=quant(f"layers.{l}.Wup", I, D, s(10)),
        Wdown=quant(f"layers.{l}.Wdown", D, I, s(11)),
    ))

mx.save_safetensors("models/model-quant.safetensors", save)
print(f"wrote models/model-quant.safetensors with {len(save)} tensors (proj=4bit, norms/embed=fp32)")

cache = [None] * LAYERS

def attn_block(li, h, Lc, offset):
    w = layers[li]
    y = mx.fast.rms_norm(h, w["inNorm"], EPS)
    q = mx.fast.rms_norm(qmm(y, w["Wq"]).reshape(B, Lc, nH, Dh), w["qNorm"], EPS).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(qmm(y, w["Wk"]).reshape(B, Lc, nKV, Dh), w["kNorm"], EPS).transpose(0, 2, 1, 3)
    v = qmm(y, w["Wv"]).reshape(B, Lc, nKV, Dh).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)
    k = mx.fast.rope(k, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)
    if cache[li] is not None:
        k = mx.concatenate([cache[li][0], k], axis=2)
        v = mx.concatenate([cache[li][1], v], axis=2)
    cache[li] = (k, v)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal" if Lc > 1 else None)
    o = o.transpose(0, 2, 1, 3).reshape(B, Lc, nH * Dh)
    h = h + qmm(o, w["Wo"])
    y2 = mx.fast.rms_norm(h, w["postNorm"], EPS)
    act = (qmm(y2, w["Wgate"]) * mx.sigmoid(qmm(y2, w["Wgate"]))) * qmm(y2, w["Wup"])
    return h + qmm(act, w["Wdown"])

def step(ids, offset):
    Lc = len(ids)
    h = mx.take(embed, mx.array(ids, dtype=mx.int32).reshape(B, Lc), axis=0)
    for li in range(LAYERS):
        h = attn_block(li, h, Lc, offset)
    h = mx.fast.rms_norm(h, finalNorm, EPS)
    h_last = mx.take(h, mx.array([Lc - 1], dtype=mx.int32), axis=1)
    tok = int(mx.argmax(qmm(h_last, lmHead), axis=2).item())
    mx.eval(*[t for c in cache if c for t in c])
    return tok

prompt = [1, 7, 3, 9]
out = []
tok = step(prompt, 0); out.append(tok)
pos = len(prompt)
for i in range(1, 12):
    tok = step([tok], pos); pos += 1; out.append(tok)

print("Qwen3 4-bit KV-cache decode — MLX Python reference")
print(f"  generated: {out}")
