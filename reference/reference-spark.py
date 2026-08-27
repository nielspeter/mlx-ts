# Oracle for the Spark-TTS prompt and LM, using Hugging Face transformers' own
# Qwen2 — the implementation this checkpoint was published for.
#
# Run in float32, not the checkpoint's bf16, which makes the comparison exact
# rather than approximate. bf16 has eight mantissa bits, and this model carries
# outlier channels in the thousands that cancel in the last layer (+3700 -> -800
# at the final position), so a one-ulp difference at layer 9 becomes a percent at
# the logits. In bf16 no two implementations agree for long — PyTorch's own bf16
# diverges from its float32 after 5 greedy tokens. In float32 they agree exactly.
#
#   /tmp/sdvenv/bin/pip install torch transformers tokenizers
#   /tmp/sdvenv/bin/python reference/reference-spark.py
import os

import torch
from tokenizers import Tokenizer
from transformers import Qwen2ForCausalLM

DIR = os.path.expanduser("~/.cache/mlx-ts/mlx-community/Spark-TTS-0.5B-bf16")
TEXT = "MLX runs on the GPU of your Mac."

# The control prompt: gender 0 = female, level 2 = moderate.
prompt = ("<|task_controllable_tts|><|start_content|>" + TEXT + "<|end_content|>"
          "<|start_style_label|><|gender_0|><|pitch_label_2|><|speed_label_2|><|end_style_label|>")

ids = Tokenizer.from_file(f"{DIR}/tokenizer.json").encode(prompt, add_special_tokens=False).ids
print("prompt ids:", ids)

model = Qwen2ForCausalLM.from_pretrained(DIR, dtype=torch.float32)
model.eval()

with torch.no_grad():
    last = model(torch.tensor([ids])).logits[0, -1]
    top = last.topk(5)
    print("top5:", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])

    # Greedy, so the sequence is deterministic and can be compared id for id.
    cur, out = list(ids), []
    for _ in range(16):
        nxt = int(model(torch.tensor([cur])).logits[0, -1].argmax())
        out.append(nxt)
        cur.append(nxt)
print("gen ids:", out)
