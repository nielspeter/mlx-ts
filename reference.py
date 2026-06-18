"""Reference Qwen3 decoder block in MLX Python — must match block.ts exactly.

Same architecture, same deterministic weights, same MLX fast ops. If the TS
path over mlx-c is correct, sum / sum_sq here match block.ts to ~float32.
Run: python3 reference.py
"""
import mlx.core as mx

B, L = 1, 3
D = 64
nH, nKV = 4, 2
Dh = 16
I = 128
EPS = 1e-6
THETA = 1_000_000.0
SCALE = Dh ** -0.5


def det(n, seed):
    return mx.array(
        [(((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.1 for i in range(n)],
        dtype=mx.float32,
    )


def W(n, shape, seed):
    return det(n, seed).reshape(shape)


inNorm   = W(D, [D], 1)
Wq       = W(D * nH * Dh, [D, nH * Dh], 2)
Wk       = W(D * nKV * Dh, [D, nKV * Dh], 3)
Wv       = W(D * nKV * Dh, [D, nKV * Dh], 4)
qNorm    = W(Dh, [Dh], 5)
kNorm    = W(Dh, [Dh], 6)
Wo       = W(nH * Dh * D, [nH * Dh, D], 7)
postNorm = W(D, [D], 8)
Wgate    = W(D * I, [D, I], 9)
Wup      = W(D * I, [D, I], 10)
Wdown    = W(I * D, [I, D], 11)

x = det(B * L * D, 100).reshape([B, L, D])


def block(h):
    y = mx.fast.rms_norm(h, inNorm, EPS)
    q = mx.matmul(y, Wq)
    k = mx.matmul(y, Wk)
    v = mx.matmul(y, Wv)

    q = mx.fast.rms_norm(q.reshape(B, L, nH, Dh), qNorm, EPS).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(B, L, nKV, Dh), kNorm, EPS).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, nKV, Dh).transpose(0, 2, 1, 3)

    q = mx.fast.rope(q, Dh, traditional=False, base=THETA, scale=1.0, offset=0)
    k = mx.fast.rope(k, Dh, traditional=False, base=THETA, scale=1.0, offset=0)

    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal")
    o = o.transpose(0, 2, 1, 3).reshape(B, L, nH * Dh)
    o = mx.matmul(o, Wo)
    h = h + o

    y2 = mx.fast.rms_norm(h, postNorm, EPS)
    act = (mx.matmul(y2, Wgate) * mx.sigmoid(mx.matmul(y2, Wgate))) * mx.matmul(y2, Wup)
    mlp = mx.matmul(act, Wdown)
    return h + mlp


out = block(x)
mx.eval(out)
print("Qwen3 decoder block — MLX Python reference")
print(f"  output shape: {list(out.shape)}")
print(f"  sum    = {float(out.sum()):.6f}")
print(f"  sum_sq = {float((out * out).sum()):.6f}")
