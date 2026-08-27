# Oracle for the Qwen2 backbone, using mlx-lm's own implementation.
#
# Greedy decoding from a fixed prompt, so the comparison is token ids rather
# than text — a tokenizer disagreement would otherwise hide behind matching
# words. Defaults to Spark-TTS's LM because that is what needs it, but any
# qwen2 checkpoint works.
#
#   /tmp/sdvenv/bin/python reference/reference-qwen2.py [repo-or-path]
import os, sys
import mlx.core as mx
from mlx_lm import load

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.cache/mlx-ts/mlx-community/Spark-TTS-0.5B-bf16")

model, tokenizer = load(path)
# Spark's controllable-TTS prompt. A plain sentence makes this model emit the
# same semantic token forever, and two implementations agreeing on a constant
# proves very little — this produces a varied sequence instead.
prompt = (
    "<|task_controllable_tts|><|start_content|>Hello there, this is a test."
    "<|end_content|><|start_style_label|><|gender_1|><|pitch_label_2|>"
    "<|speed_label_2|><|end_style_label|>"
)
ids = tokenizer.encode(prompt)

# Greedy, step by step, so the ids are directly comparable.
cache = None
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)
out = []
cur = mx.array([ids])
for _ in range(12):
    logits = model(cur, cache=cache)
    tok = int(mx.argmax(logits[0, -1]).item())
    out.append(tok)
    cur = mx.array([[tok]])

print(f"prompt ids: {ids}")
print(f"gen ids: {out}")
print(f"completion: {tokenizer.decode(out)!r}")
