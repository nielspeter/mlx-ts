"""Reference: small Qwen3 model + KV-cache greedy decode in MLX Python.
Must produce the exact same generated token ids as model-gen.ts.
Run: python3 reference-decode.py
"""
import mlx.core as mx

VOCAB, D, nH, nKV, Dh, I, LAYERS = 32, 64, 4, 2, 16, 128, 2
EPS, THETA, SCALE, B = 1e-6, 1_000_000.0, Dh ** -0.5, 1


def det(n, seed):
    return mx.array([(((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.1
                     for i in range(n)], dtype=mx.float32)


def W(n, shape, seed):
    return det(n, seed).reshape(shape)


embed = W(VOCAB * D, [VOCAB, D], 0)


def make_layer(l):
    s = lambda k: 100 + l * 20 + k
    return dict(
        inNorm=W(D, [D], s(1)),
        Wq=W(D * nH * Dh, [D, nH * Dh], s(2)),
        Wk=W(D * nKV * Dh, [D, nKV * Dh], s(3)),
        Wv=W(D * nKV * Dh, [D, nKV * Dh], s(4)),
        qNorm=W(Dh, [Dh], s(5)),
        kNorm=W(Dh, [Dh], s(6)),
        Wo=W(nH * Dh * D, [nH * Dh, D], s(7)),
        postNorm=W(D, [D], s(8)),
        Wgate=W(D * I, [D, I], s(9)),
        Wup=W(D * I, [D, I], s(10)),
        Wdown=W(I * D, [I, D], s(11)),
    )


layers = [make_layer(l) for l in range(LAYERS)]
finalNorm = W(D, [D], 900)
lmHead = W(D * VOCAB, [D, VOCAB], 901)
cache = [None] * LAYERS


def attn_block(li, h, Lc, offset):
    w = layers[li]
    y = mx.fast.rms_norm(h, w["inNorm"], EPS)
    q = mx.fast.rms_norm(mx.matmul(y, w["Wq"]).reshape(B, Lc, nH, Dh), w["qNorm"], EPS).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(mx.matmul(y, w["Wk"]).reshape(B, Lc, nKV, Dh), w["kNorm"], EPS).transpose(0, 2, 1, 3)
    v = mx.matmul(y, w["Wv"]).reshape(B, Lc, nKV, Dh).transpose(0, 2, 1, 3)

    q = mx.fast.rope(q, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)
    k = mx.fast.rope(k, Dh, traditional=False, base=THETA, scale=1.0, offset=offset)

    if cache[li] is not None:
        k = mx.concatenate([cache[li][0], k], axis=2)
        v = mx.concatenate([cache[li][1], v], axis=2)
    cache[li] = (k, v)

    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal" if Lc > 1 else None)
    o = o.transpose(0, 2, 1, 3).reshape(B, Lc, nH * Dh)
    h = h + mx.matmul(o, w["Wo"])

    y2 = mx.fast.rms_norm(h, w["postNorm"], EPS)
    act = (mx.matmul(y2, w["Wgate"]) * mx.sigmoid(mx.matmul(y2, w["Wgate"]))) * mx.matmul(y2, w["Wup"])
    return h + mx.matmul(act, w["Wdown"])


def step(ids, offset):
    Lc = len(ids)
    h = mx.take(embed, mx.array(ids, dtype=mx.int32).reshape(B, Lc), axis=0)
    for li in range(LAYERS):
        h = attn_block(li, h, Lc, offset)
    h = mx.fast.rms_norm(h, finalNorm, EPS)
    h_last = mx.take(h, mx.array([Lc - 1], dtype=mx.int32), axis=1)
    logits = mx.matmul(h_last, lmHead)
    tok = int(mx.argmax(logits, axis=2).item())
    mx.eval(*[t for c in cache if c for t in c])
    return tok


prompt = [1, 7, 3, 9]
N_NEW = 12
out = []
tok = step(prompt, 0)
out.append(tok)
pos = len(prompt)
for i in range(1, N_NEW):
    tok = step([tok], pos)
    pos += 1
    out.append(tok)

print("Qwen3 KV-cache decode — MLX Python reference")
print(f"  prompt:    {prompt}")
print(f"  generated: {out}")
