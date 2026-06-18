"""Reference: the same linear-regression training in MLX Python. Final loss and
W must match spike-train.ts. Run: python3 reference-train.py
"""
import mlx.core as mx

N, D, LR, STEPS = 16, 4, 0.1, 50
def det(n, seed):
    return mx.array([((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5 for i in range(n)], dtype=mx.float32)

X = det(N * D, 1).reshape(N, D)
Y = X @ det(D, 2).reshape(D, 1) + 0.5

def loss_fn(W, b):
    diff = X @ W + b - Y
    return (diff * diff).sum() / N

vag = mx.value_and_grad(loss_fn, argnums=(0, 1))
W = mx.zeros((D, 1)); b = mx.zeros((1,))
print("=== training reference: linear regression (MLX Python) ===")
for step in range(STEPS):
    loss, (gW, gb) = vag(W, b)
    W = W - LR * gW; b = b - LR * gb
    mx.eval(W, b, loss)
    if step % 10 == 0 or step == STEPS - 1:
        print(f"  step {step:2d}: loss {float(loss):.6f}")
print(f"final loss: {float(loss):.6f}")
print("W: [" + ", ".join(f"{v:.4f}" for v in W.flatten().tolist()) + "]")
