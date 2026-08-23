"""Write the small Qwen3 model's weights to a real .safetensors file, so the
TS side can load them via mlx_load_safetensors. Same deterministic weights and
key names that model-load.ts expects. Run: python3 save-model.py
"""
import mlx.core as mx

VOCAB, D, nH, nKV, Dh, I, LAYERS = 32, 64, 4, 2, 16, 128, 2


def det(n, seed):
    return mx.array([(((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.1
                     for i in range(n)], dtype=mx.float32)


def W(n, shape, seed):
    return det(n, seed).reshape(shape)

w = {"embed": W(VOCAB * D, [VOCAB, D], 0),
     "finalNorm": W(D, [D], 900),
     "lmHead": W(D * VOCAB, [D, VOCAB], 901)}

names = ["inNorm", "Wq", "Wk", "Wv", "qNorm", "kNorm", "Wo", "postNorm", "Wgate", "Wup", "Wdown"]
sizes = [(D, [D]), (D * nH * Dh, [D, nH * Dh]), (D * nKV * Dh, [D, nKV * Dh]),
         (D * nKV * Dh, [D, nKV * Dh]), (Dh, [Dh]), (Dh, [Dh]), (nH * Dh * D, [nH * Dh, D]),
         (D, [D]), (D * I, [D, I]), (D * I, [D, I]), (I * D, [I, D])]
for l in range(LAYERS):
    s = lambda k: 100 + l * 20 + k
    for k, (name, (n, shape)) in enumerate(zip(names, sizes), start=1):
        w[f"layers.{l}.{name}"] = W(n, shape, s(k))

mx.save_safetensors("model.safetensors", w)
print(f"wrote model.safetensors with {len(w)} tensors")
