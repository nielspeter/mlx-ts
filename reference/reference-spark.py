# Oracle for the Spark-TTS prompt and LM, using mlx-audio's own Qwen2 port.
#
# Greedy (temperature 0), so the token sequence is deterministic and can be
# compared id for id — sampling would only ever agree by luck.
#
# The audio half is checked separately by reference-bicodec.py. mlx-audio's own
# Model.post_load_hook is bypassed here because it also builds a BiCodec
# *encoder*, which wants an audio_tokenizer_config.yaml the repo does not ship.
#
#   /tmp/sdvenv/bin/python reference/reference-spark.py
import os

import mlx.core as mx
from mlx_audio.lm.generate import stream_generate
from mlx_audio.lm.models.qwen2 import Model as Qwen2Model
from mlx_audio.lm.sample_utils import make_logits_processors, make_sampler
from mlx_audio.tts.models.spark.spark import ModelConfig
from tokenizers import Tokenizer

DIR = os.path.expanduser("~/.cache/mlx-ts/mlx-community/Spark-TTS-0.5B-bf16")
TEXT = "MLX runs on the GPU of your Mac."

# The control prompt: gender 0 = female, level 2 = moderate.
prompt = ("<|task_controllable_tts|><|start_content|>" + TEXT + "<|end_content|>"
          "<|start_style_label|><|gender_0|><|pitch_label_2|><|speed_label_2|><|end_style_label|>")

tok = Tokenizer.from_file(f"{DIR}/tokenizer.json")
ids = tok.encode(prompt, add_special_tokens=False).ids
print("prompt ids:", ids)

model = Qwen2Model(ModelConfig())
weights = mx.load(f"{DIR}/model.safetensors")
model.load_weights(list(model.sanitize(weights).items()))
model.eval()


class Detok:  # stream_generate only needs a detokenizer, not a real tokenizer
    def __init__(self, tok):
        self.tok = tok
        self.eos_token_ids = {151645}

    def decode(self, ids): return self.tok.decode(ids)

    @property
    def detokenizer(self): return self


# The primary comparison. Greedy ids are tie-sensitive: the checkpoint is bf16,
# and adjacent audio tokens routinely land on the *same* bf16 logit, where the
# two runtimes break the tie differently and the sequences then re-converge.
# Logits have no such ambiguity.
out0 = model(mx.array([ids]))
last = (out0[0] if isinstance(out0, tuple) else out0)[0, -1].astype(mx.float32)
mx.eval(last)
top = mx.argsort(-last)[:5].tolist()
print("top5:", [(int(t), round(float(last[t]), 4)) for t in top])

out = []
for r in stream_generate(model, tokenizer=Detok(tok), prompt=mx.array(ids), max_tokens=16,
                         sampler=make_sampler(0.0),
                         logits_processors=make_logits_processors(None, 1.3, 20)):
    out.append(r.token)
print("gen ids:", out)
