# Ground truth from Apple's OWN mlx-lm, not a reimplementation.
#
# reference-qwen.py builds the Qwen3 forward pass by hand in MLX Python, which
# checks that our TypeScript matches *that* implementation. Both were written
# from the same reading of the architecture, so a shared misreading would make
# them agree and both be wrong. This one loads the same local weights into
# mlx_lm.models.qwen3.Model — the code Apple ships — and decodes greedily.
#
#   python3 reference/reference-mlxlm-qwen.py "The capital of France is"
import sys, json
import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs
from tokenizers import Tokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
N_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 24

cfg = json.load(open("models/config.json"))
model = Model(ModelArgs.from_dict(cfg))
weights = mx.load("models/model-qwen.safetensors")
model.load_weights(list(model.sanitize(weights).items()))
mx.eval(model.parameters())

tok = Tokenizer.from_file("models/tokenizer.json")
ids = tok.encode(prompt).ids
gen = []
for _ in range(N_TOK):
    logits = model(mx.array([ids + gen]))          # [1, T, vocab]
    nxt = int(mx.argmax(logits[0, -1]).item())
    gen.append(nxt)

print(f"prompt ids: {ids}")
print(f"gen ids:    {gen}")
print(f"completion: {tok.decode(gen)!r}")
