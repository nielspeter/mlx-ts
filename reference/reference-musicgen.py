# Oracle for src/models/musicgen.ts: one decoding step of MusicGen's LM, run
# through Hugging Face's own MusicgenForConditionalGeneration — the reference
# implementation these weights ship for.
#
# Deliberately NOT a hand-written MLX reimplementation: a reference I wrote
# myself would only prove I was consistent with myself. That is exactly how the
# Metal-kernel grid bug survived (it was wrong in the port AND in my oracle).
#
#   PYTHONPATH=<scratch pylibs> python3 reference/reference-musicgen.py
import os, torch
from transformers import MusicgenForConditionalGeneration

CACHE = os.environ.get("MLXTS_CACHE", os.path.expanduser("~/.cache/mlx-ts"))
model = MusicgenForConditionalGeneration.from_pretrained(f"{CACHE}/facebook/musicgen-small")
model.eval()
dec = model.decoder

B, K, Lt = 1, 4, 6
D = model.config.decoder.hidden_size

# Deterministic inputs, matching the TypeScript side exactly.
tokens = torch.tensor([[[(k * 977 + 7) % 2048 for k in range(K)]]], dtype=torch.long)  # [B,1,K]
cond = torch.tensor([[[((i * 131 + j * 977 + 7) % 1009) / 1009 - 0.5
                       for j in range(D)] for i in range(Lt)]], dtype=torch.float32)   # [B,Lt,D]

with torch.no_grad():
    out = dec(
        input_ids=tokens.reshape(B * K, 1),
        encoder_hidden_states=cond,
        use_cache=False,
    )
logits = out.logits            # [B, K, 1, vocab] or [B*K, 1, vocab]
l = logits.reshape(-1).tolist()
print(f"  logits shape: {tuple(logits.shape)}")
print(f"  first8      : {', '.join(f'{v:.5f}' for v in l[:8])}")
print(f"  sum         : {sum(l):.5f}")
