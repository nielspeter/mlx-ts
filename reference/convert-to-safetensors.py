# Convert a PyTorch checkpoint to safetensors, so TypeScript can read it.
#
# The fallback, not the first answer. For MusicGen itself, use the already
# converted jasonvassallo/mlx-musicgen-{medium,large} — fromPretrained reads
# that layout directly, and it costs a download rather than an 8 GB pickle and
# a torch install. This script is for the repos nobody has converted: the many
# musicgen fine-tunes on the Hub, which ship pytorch_model.bin and nothing else.
#
# It downloads a repo's PyTorch weights and rewrites them as safetensors into
# the mlx-ts cache, after which `MusicGen.fromPretrained("<repo>")` works.
#
#   python3 reference/convert-to-safetensors.py <some-user>/musicgen-medium-finetune
#
# Needs torch + transformers + huggingface_hub. medium is ~8 GB to download and
# roughly the same again on disk once converted, so check your free space first.
import os, sys, json, shutil

repo = sys.argv[1] if len(sys.argv) > 1 else "facebook/musicgen-medium"
CACHE = os.environ.get("MLXTS_CACHE", os.path.expanduser("~/.cache/mlx-ts"))
out_dir = os.path.join(CACHE, repo)
os.makedirs(out_dir, exist_ok=True)

target = os.path.join(out_dir, "model.safetensors")
if os.path.exists(target):
    print(f"already converted: {target}")
    raise SystemExit

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

# Bring the small files across so fromPretrained finds a complete repo.
for f in ("config.json", "tokenizer.json", "generation_config.json",
          "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json"):
    try:
        shutil.copyfile(hf_hub_download(repo_id=repo, filename=f), os.path.join(out_dir, f))
    except Exception:
        pass    # not every repo has every file

print(f"downloading {repo}/pytorch_model.bin ...")
bin_path = hf_hub_download(repo_id=repo, filename="pytorch_model.bin")

print("loading (this holds the whole checkpoint in RAM) ...")
sd = torch.load(bin_path, map_location="cpu", weights_only=True)
sd = sd.get("state_dict", sd)

# safetensors refuses tensors that share storage, which a tied embedding does.
seen, out = {}, {}
for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    key = (v.data_ptr(), v.shape, v.stride())
    out[k] = v.clone().contiguous() if key in seen else v.contiguous()
    seen[key] = k

print(f"writing {len(out)} tensors -> {target}")
save_file(out, target, metadata={"format": "pt"})
print(f"done. now: bun examples/musicgen.ts \"trance\" --model {repo}")
