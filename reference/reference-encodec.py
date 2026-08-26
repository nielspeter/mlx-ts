# Oracle for src/models/encodec.ts: EnCodec's decoder in MLX Python, same
# weights, same deterministic codes. Uses mlx-examples/musicgen's encodec.py,
# fetched into /tmp by the setup step in docs (see AGENTS.md).
#   python3 reference/reference-encodec.py
import sys, os
sys.path.insert(0, os.environ.get("MLX_EXAMPLES", "/tmp"))   # encodec.py from mlx-examples
import mlx.core as mx
from encodec import EncodecModel

# Build the model directly from the SAME local weights the TypeScript side
# reads (populated by src/io/hub.ts). Not from_pretrained: it imports
# huggingface_hub unconditionally, before it checks whether the path is local,
# so it needs network dependencies even for a file already on disk.
import json
from types import SimpleNamespace

CACHE = os.environ.get("MLXTS_CACHE", os.path.expanduser("~/.cache/mlx-ts"))
DIR = f"{CACHE}/mlx-community/encodec-32khz-float32"
config = SimpleNamespace(**json.load(open(f"{DIR}/config.json")))
model = EncodecModel(config)
model.load_weights(f"{DIR}/model.safetensors")
mx.eval(model)

B, K, T = 1, 4, 16
# Same deterministic codes as the TS side.
codes = mx.array([[[(t * 131 + k * 977 + 7) % 2048 for t in range(T)] for k in range(K)]], dtype=mx.uint32)

audio = model.decode(codes[:, None], audio_scales=[None])[0]
mx.eval(audio)
a = audio.flatten().tolist()
print(f"  samples: {len(a)}")
print(f"  first8 : {', '.join(f'{v:.6f}' for v in a[:8])}")
print(f"  sum    : {sum(a):.6f}")
