"""Reference: LoRA fine-tune of 4-bit Qwen3-0.6B in MLX Python (manual Adam +
cross_entropy), mirroring lora-train.ts. Loss curve must match. Run: python3 reference-lora.py
"""
import json
import mlx.core as mx

cfg = json.load(open("config-4bit.json"))
D, NL = cfg["hidden_size"], cfg["num_hidden_layers"]
nH, nKV, Dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
EPS, THETA, SCALE = cfg["rms_norm_eps"], cfg["rope_theta"], cfg["head_dim"] ** -0.5
GS, BITS, B = cfg["quantization"]["group_size"], cfg["quantization"]["bits"], 1
qDim, kvDim, V = nH * Dh, nKV * Dh, cfg["vocab_size"]

R, ALPHA, LSCALE, LR, STEPS = 8, 16, 16 / 8, 1e-3, 60
B1, B2, EPSA = 0.9, 0.999, 1e-8
SEQ = [785, 6722, 315, 9625, 374, 12095, 13, 576]
L = len(SEQ)
ids = mx.array(SEQ, dtype=mx.int32)

w = mx.load("model-q4.safetensors")
def qmm(x, p): return mx.quantized_matmul(x, w[f"{p}.weight"], w[f"{p}.scales"], w[f"{p}.biases"], transpose=True, group_size=GS, bits=BITS)
def rms(x, n): return mx.fast.rms_norm(x, w[n], EPS)

def det(n, seed):
    return mx.array([(((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5) * 0.02 for i in range(n)], dtype=mx.float32)

# flat param list [Aq0, Bq0, Av0, Bv0, Aq1, ...] matching lora-train.ts
params = []
for i in range(NL):
    params += [det(D * R, i * 4 + 1).reshape(D, R), mx.zeros((R, qDim)),
               det(D * R, i * 4 + 3).reshape(D, R), mx.zeros((R, kvDim))]
P = len(params)

def embed(idx):
    e = "model.embed_tokens"
    return mx.dequantize(mx.take(w[f"{e}.weight"], idx, axis=0), mx.take(w[f"{e}.scales"], idx, axis=0), mx.take(w[f"{e}.biases"], idx, axis=0), group_size=GS, bits=BITS)

def forward(ps, idx):
    h = embed(idx.reshape(B, L))
    for i in range(NL):
        p = f"model.layers.{i}"
        Aq, Bq, Av, Bv = ps[4 * i], ps[4 * i + 1], ps[4 * i + 2], ps[4 * i + 3]
        y = rms(h, f"{p}.input_layernorm.weight")
        q = qmm(y, f"{p}.self_attn.q_proj") + (y @ Aq) @ Bq * LSCALE
        k = qmm(y, f"{p}.self_attn.k_proj")
        v = qmm(y, f"{p}.self_attn.v_proj") + (y @ Av) @ Bv * LSCALE
        q = rms(q.reshape(B, L, nH, Dh), f"{p}.self_attn.q_norm.weight").transpose(0, 2, 1, 3)
        k = rms(k.reshape(B, L, nKV, Dh), f"{p}.self_attn.k_norm.weight").transpose(0, 2, 1, 3)
        v = v.reshape(B, L, nKV, Dh).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, Dh, traditional=False, base=THETA, scale=1.0, offset=0)
        k = mx.fast.rope(k, Dh, traditional=False, base=THETA, scale=1.0, offset=0)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal").transpose(0, 2, 1, 3).reshape(B, L, qDim)
        h = h + qmm(o, f"{p}.self_attn.o_proj")
        y2 = rms(h, f"{p}.post_attention_layernorm.weight")
        g = qmm(y2, f"{p}.mlp.gate_proj")
        h = h + qmm((g * mx.sigmoid(g)) * qmm(y2, f"{p}.mlp.up_proj"), f"{p}.mlp.down_proj")
    h = rms(h, "model.norm.weight").reshape(L, D)
    return mx.quantized_matmul(h, w["model.embed_tokens.weight"], w["model.embed_tokens.scales"], w["model.embed_tokens.biases"], transpose=True, group_size=GS, bits=BITS)

def loss_fn(ps, idx):
    logits = forward(ps, idx)[: L - 1]                       # [L-1, V]
    targets = idx[1:L].reshape(L - 1, 1)
    lp = mx.log(mx.softmax(logits, axis=-1))
    return -mx.take_along_axis(lp, targets, axis=1).mean()

vag = mx.value_and_grad(loss_fn, argnums=0)
mom = [mx.zeros(p.shape) for p in params]
vel = [mx.zeros(p.shape) for p in params]
print(f"=== LoRA fine-tune ref: Qwen3-0.6B-4bit, rank {R}, {P} adapter tensors ===")
for step in range(STEPS):
    loss, grads = vag(params, ids)
    t = step + 1
    bc1, bc2 = 1 - B1 ** t, 1 - B2 ** t
    for i in range(P):
        mom[i] = B1 * mom[i] + (1 - B1) * grads[i]
        vel[i] = B2 * vel[i] + (1 - B2) * grads[i] * grads[i]
        params[i] = params[i] - LR * (mom[i] / bc1) / (mx.sqrt(vel[i] / bc2) + EPSA)
    mx.eval(params, mom, vel, loss)
    if step % 5 == 0 or step == STEPS - 1:
        print(f"  step {step:2d}: loss {float(loss):.6f}")
print(f"final loss: {float(loss):.6f}")
