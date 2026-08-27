# Oracle for conv2d / conv_transpose2d — the primitives Stable Diffusion's UNet
# and VAE are built from. Deterministic inputs, so TS and Python see the same
# numbers without shipping a fixture.
import mlx.core as mx

def det(n, seed):
    return mx.array([((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5 for i in range(n)], dtype=mx.float32)

N, H, W, CIN, COUT, KH, KW = 2, 9, 7, 4, 6, 3, 3
x = det(N * H * W * CIN, 1).reshape(N, H, W, CIN)          # channels-last
w = det(COUT * KH * KW * CIN, 2).reshape(COUT, KH, KW, CIN)

def show(tag, a):
    f = a.flatten()
    print(f"{tag} shape={list(a.shape)} sum={float(f.sum()):.6f} "
          f"first4={[round(float(v), 6) for v in f[:4].tolist()]}")

show("conv2d_basic  ", mx.conv2d(x, w, stride=1, padding=0, dilation=1, groups=1))
show("conv2d_stride ", mx.conv2d(x, w, stride=2, padding=1, dilation=1, groups=1))
show("conv2d_dilate ", mx.conv2d(x, w, stride=1, padding=2, dilation=2, groups=1))

# Grouped conv takes [C_out, KH, KW, C_in / groups].
wg = det(COUT * KH * KW * (CIN // 2), 3).reshape(COUT, KH, KW, CIN // 2)
show("conv2d_groups ", mx.conv2d(x, wg, stride=1, padding=1, dilation=1, groups=2))

wt = det(CIN * KH * KW * COUT, 4).reshape(CIN, KH, KW, COUT)
y = mx.conv2d(x, w, stride=2, padding=1)
show("convT_stride2 ", mx.conv_transpose2d(y, wt, stride=2, padding=1, output_padding=1))
