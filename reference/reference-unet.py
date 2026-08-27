# Oracle for the UNet, using mlx-examples' own Stable Diffusion port.
#
# Both sides load stable-diffusion-v1-5's unet/ and run the same fixed latents,
# timestep and conditioning, so any difference is our UNet rather than the
# checkpoint. 16x16 latents keep it cheap; the architecture is identical at 64.
#
#   MLX_SD=/path /tmp/sdvenv/bin/python reference/reference-unet.py
import json, os, sys
sys.path.insert(0, os.environ.get("MLX_SD", "/tmp/mlxsd_pkg"))

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten
from stable_diffusion.config import UNetConfig
from stable_diffusion.model_io import _flatten, map_unet_weights
from stable_diffusion.unet import UNetModel

REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

def det(n, seed):
    return mx.array([((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5 for i in range(n)], dtype=mx.float32)

cfg = json.load(open(hf_hub_download(REPO, "unet/config.json")))
n = len(cfg["block_out_channels"])
model = UNetModel(UNetConfig(
    in_channels=cfg["in_channels"],
    out_channels=cfg["out_channels"],
    block_out_channels=cfg["block_out_channels"],
    layers_per_block=[cfg["layers_per_block"]] * n,
    transformer_layers_per_block=cfg.get("transformer_layers_per_block", (1,) * n),
    num_attention_heads=([cfg["attention_head_dim"]] * n
                         if isinstance(cfg["attention_head_dim"], int) else cfg["attention_head_dim"]),
    cross_attention_dim=[cfg["cross_attention_dim"]] * n,
    norm_num_groups=cfg["norm_num_groups"],
    down_block_types=cfg["down_block_types"],
    up_block_types=cfg["up_block_types"][::-1],
))
raw = mx.load(hf_hub_download(REPO, "unet/diffusion_pytorch_model.safetensors"))
model.update(tree_unflatten(_flatten([map_unet_weights(k, v.astype(mx.float32)) for k, v in raw.items()])))

B, H, W, C = 1, 16, 16, cfg["in_channels"]
x = det(B * H * W * C, 1).reshape(B, H, W, C)
cond = det(B * 77 * cfg["cross_attention_dim"], 2).reshape(B, 77, cfg["cross_attention_dim"])
out = model(x, mx.array([500.0]), encoder_x=cond)
mx.eval(out)

f = out.flatten()
print(f"unet shape={list(out.shape)} mean={float(f.mean()):.6f} "
      f"absmean={float(mx.abs(f).mean()):.6f} "
      f"first4={[round(float(v), 4) for v in f[:4].tolist()]}")
