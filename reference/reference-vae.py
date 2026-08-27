# Oracle for the VAE decoder, using mlx-examples' own Stable Diffusion port.
#
# Both sides load the SAME weight file — stabilityai/sd-vae-ft-mse,
# diffusion_pytorch_model.safetensors — so any difference is our decoder, not
# the checkpoint. mlx-examples' hub-key table points at a gated repo, so the
# model and its weight mapper are used directly rather than load_autoencoder().
#
# MLX_SD holds the directory *containing* the stable_diffusion package.
#   MLX_SD=/path python3 reference/reference-vae.py
import json, os, sys
sys.path.insert(0, os.environ.get("MLX_SD", "/tmp/mlxsd_pkg"))

import mlx.core as mx
from huggingface_hub import hf_hub_download
from stable_diffusion.config import AutoencoderConfig
from stable_diffusion.model_io import _load_safetensor_weights, map_vae_weights
from stable_diffusion.vae import Autoencoder

REPO = "stabilityai/sd-vae-ft-mse"

def det(n, seed):
    return mx.array([((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5 for i in range(n)], dtype=mx.float32)

cfg = json.load(open(hf_hub_download(REPO, "config.json")))
model = Autoencoder(AutoencoderConfig(
    in_channels=cfg["in_channels"],
    out_channels=cfg["out_channels"],
    latent_channels_out=2 * cfg["latent_channels"],
    latent_channels_in=cfg["latent_channels"],
    block_out_channels=cfg["block_out_channels"],
    layers_per_block=cfg["layers_per_block"],
    norm_num_groups=cfg["norm_num_groups"],
    scaling_factor=cfg.get("scaling_factor", 0.18215),
))
# sd-vae-ft-mse predates diffusers renaming attention projections, so its keys
# are query/key/value/proj_attn where mlx-examples' mapper expects
# to_q/to_k/to_v/to_out.0. Rename before handing over, so the oracle stays
# mlx-examples' own model and mapper.
LEGACY = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}

def mapper(key, value):
    for old, new in LEGACY.items():
        key = key.replace(f"attentions.0.{old}.", f"attentions.0.{new}.")
    return map_vae_weights(key, value)

_load_safetensor_weights(mapper, model,
                         hf_hub_download(REPO, "diffusion_pytorch_model.safetensors"), False)

B, h, w, C = 1, 16, 16, 4
z = det(B * h * w * C, 1).reshape(B, h, w, C)
img = model.decode(z)
mx.eval(img)

f = img.flatten()
print(f"vae_decode shape={list(img.shape)} sum={float(f.sum()):.3f} "
      f"mean={float(f.mean()):.5f} "
      f"first4={[round(float(v), 4) for v in f[:4].tolist()]}")
