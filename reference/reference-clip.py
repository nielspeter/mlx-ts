# Oracle for CLIP's text encoder, using mlx-examples' own port.
#
# Both sides load openai/clip-vit-large-patch14 — the text encoder Stable
# Diffusion 1.x conditions on. Fixed token ids, so nothing depends on a
# tokenizer agreeing yet; that is checked separately.
#
# MLX_SD holds the directory *containing* the stable_diffusion package.
#   MLX_SD=/path /tmp/sdvenv/bin/python reference/reference-clip.py
import json, os, sys
sys.path.insert(0, os.environ.get("MLX_SD", "/tmp/mlxsd_pkg"))

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten
from stable_diffusion.clip import CLIPTextModel
from stable_diffusion.config import CLIPTextModelConfig
from stable_diffusion.model_io import map_clip_text_encoder_weights

REPO = "openai/clip-vit-large-patch14"

cfg = json.load(open(hf_hub_download(REPO, "config.json")))["text_config"]
model = CLIPTextModel(CLIPTextModelConfig(
    num_layers=cfg["num_hidden_layers"],
    model_dims=cfg["hidden_size"],
    num_heads=cfg["num_attention_heads"],
    max_length=cfg["max_position_embeddings"],
    vocab_size=cfg["vocab_size"],
    hidden_act=cfg.get("hidden_act", "quick_gelu"),
))

# The checkpoint carries the vision tower too; only the text half is wanted,
# and position_ids is a buffer rather than a parameter.
raw = mx.load(hf_hub_download(REPO, "model.safetensors"))
weights = []
for k, v in raw.items():
    if not k.startswith("text_model.") or k.endswith("position_ids"):
        continue
    weights.extend(map_clip_text_encoder_weights(k, v.astype(mx.float32)))
model.update(tree_unflatten(weights))

# A fixed prompt's worth of ids: BOS, some words, EOS, then padding.
ids = mx.array([[49406, 320, 1125, 539, 320, 2368, 49407] + [49407] * 3])
out = model(ids)
h = out.last_hidden_state
mx.eval(h)

f = h.flatten()
print(f"clip_text shape={list(h.shape)} mean={float(f.mean()):.6f} "
      f"absmean={float(mx.abs(f).mean()):.6f} "
      f"first4={[round(float(v), 4) for v in f[:4].tolist()]}")
