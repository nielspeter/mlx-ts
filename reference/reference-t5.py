# Oracle for src/models/t5.ts: the T5 encoder inside facebook/musicgen-small,
# run through Hugging Face's own implementation.
#   PYTHONPATH=<scratch pylibs> python3 reference/reference-t5.py
import os, torch
from transformers import MusicgenForConditionalGeneration

CACHE = os.environ.get("MLXTS_CACHE", os.path.expanduser("~/.cache/mlx-ts"))
model = MusicgenForConditionalGeneration.from_pretrained(f"{CACHE}/facebook/musicgen-small")
model.eval()

ids = torch.tensor([[3, 17, 1029, 55, 1]], dtype=torch.long)      # arbitrary, fixed
with torch.no_grad():
    out = model.text_encoder(input_ids=ids).last_hidden_state
h = out.reshape(-1).tolist()
print(f"  shape  : {tuple(out.shape)}")
print(f"  first8 : {', '.join(f'{v:.5f}' for v in h[:8])}")
