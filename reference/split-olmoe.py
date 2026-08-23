"""Split the single-file OLMoE checkpoint into 2 shards + an index.json, so the
sharded streaming loader can be tested. Shard 1 = embed + layers 0-7,
shard 2 = layers 8-15 + final norm + lm_head. Run: python3 split-olmoe.py
"""
import os, json
import mlx.core as mx

os.makedirs("models/model-olmoe-sharded", exist_ok=True)
w = mx.load("models/model-olmoe.safetensors")
SPLIT = 8  # layers [0, SPLIT) -> shard 1, [SPLIT, NL) -> shard 2

def shard_of(name):
    if name.startswith("model.layers."):
        return 1 if int(name.split(".")[2]) < SPLIT else 2
    return 1 if name == "model.embed_tokens.weight" or name.startswith("model.embed_tokens") else 2

s1 = {k: v for k, v in w.items() if shard_of(k) == 1}
s2 = {k: v for k, v in w.items() if shard_of(k) == 2}
f1, f2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
mx.save_safetensors(f"models/model-olmoe-sharded/{f1}", s1)
mx.save_safetensors(f"models/model-olmoe-sharded/{f2}", s2)

weight_map = {**{k: f1 for k in s1}, **{k: f2 for k in s2}}
json.dump({"metadata": {}, "weight_map": weight_map},
          open("models/model-olmoe-sharded/model.safetensors.index.json", "w"))
print(f"wrote 2 shards: {len(s1)} + {len(s2)} tensors")
