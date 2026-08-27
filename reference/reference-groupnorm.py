# Oracle for GroupNorm — Stable Diffusion's VAE and UNet normalise with it
# everywhere, and its weights come from PyTorch, so the pytorch_compatible
# grouping is the one that matters.
import mlx.core as mx
import mlx.nn as nn

def det(n, seed):
    return mx.array([((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5 for i in range(n)], dtype=mx.float32)

N, H, W, C, G = 2, 5, 4, 8, 4
x = det(N * H * W * C, 1).reshape(N, H, W, C)

gn = nn.GroupNorm(G, C, eps=1e-5, pytorch_compatible=True)
gn.weight = det(C, 2)
gn.bias = det(C, 3)
y = gn(x)
f = y.flatten()
print(f"groupnorm shape={list(y.shape)} sum={float(f.sum()):.5f} "
      f"first4={[round(float(v), 5) for v in f[:4].tolist()]}")

# A single group is plain layer norm over the whole channel axis — a useful
# second point, since it isolates the grouping from the normalisation.
gn1 = nn.GroupNorm(1, C, eps=1e-5, pytorch_compatible=True)
gn1.weight = det(C, 2); gn1.bias = det(C, 3)
y1 = gn1(x); f1 = y1.flatten()
print(f"groupnorm_g1 shape={list(y1.shape)} sum={float(f1.sum()):.5f} "
      f"first4={[round(float(v), 5) for v in f1[:4].tolist()]}")
