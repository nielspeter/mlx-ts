# Oracle for the CLIP tokenizer, using mlx-examples' own port of it.
#
#   MLX_SD=/path /tmp/sdvenv/bin/python reference/reference-clip-tokenizer.py
import json, os, sys
sys.path.insert(0, os.environ.get("MLX_SD", "/tmp/mlxsd_pkg"))

from huggingface_hub import hf_hub_download
from stable_diffusion.tokenizer import Tokenizer

REPO = "openai/clip-vit-large-patch14"

with open(hf_hub_download(REPO, "vocab.json")) as f:
    vocab = json.load(f)
with open(hf_hub_download(REPO, "merges.txt"), encoding="utf-8") as f:
    lines = f.read().split("\n")
    lines = lines[1:] if lines[0].startswith("#version") else lines
    bpe_ranks = dict(map(lambda x: (tuple(x[0].split()), x[1]),
                         zip(lines, range(len(lines)))))

tok = Tokenizer(bpe_ranks, vocab)

CASES = [
    "a photo of a cat",
    "A  PHOTO   of a CAT",
    "uplifting trance, 138 bpm",
    "an astronaut riding a horse on mars, highly detailed, 4k",
    "hello-world (test) 42",
    "it's a dog's life",
    "cafe naive",
]
for c in CASES:
    print(f"{json.dumps(c)} -> {tok.tokenize(c)}")
