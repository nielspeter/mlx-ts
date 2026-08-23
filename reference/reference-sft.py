"""Oracle for sft.ts: full SFT of real GPT-2-124M in MLX Python, same weights +
same chat data + same completion-only loss. Step-0 loss must match sft.ts (same
loaded weights, same forward, same tokenization); both converge to ~0. Training
isn't bit-reproducible past step 0 (FINDINGS §6 #8) — bar is "same start, both
converge". Run: python3 reference-sft.py
"""
import os, math, json
import mlx.core as mx
from tokenizers import Tokenizer

cfg = json.load(open("models/config-gpt2.json"))
D, NL, nH = cfg["n_embd"], cfg["n_layer"], cfg["n_head"]
Dh, EPS, SCALE, EOS, VOCAB = D // nH, cfg["layer_norm_epsilon"], (D // nH) ** -0.5, 50256, cfg["vocab_size"]
ITERS = int(os.environ.get("ITERS", 120)); LR0 = float(os.environ.get("LR", 3e-5)); WARMUP = 10
B1, B2, EPSA, WD, CLIP = 0.9, 0.95, 1e-8, 0.0, 1.0

tok = Tokenizer.from_file("models/gpt2-tokenizer.json")
w = mx.load("models/gpt2-model.safetensors")

# params tree (same structure/order as sft.ts treeFlatten); split fused QKV
def block(i):
    p = f"h.{i}"
    caW, caB = w[f"{p}.attn.c_attn.weight"], w[f"{p}.attn.c_attn.bias"]
    return {"ln1w": w[f"{p}.ln_1.weight"], "ln1b": w[f"{p}.ln_1.bias"],
            "wq": caW[:, :D], "bq": caB[:D], "wk": caW[:, D:2 * D], "bk": caB[D:2 * D],
            "wv": caW[:, 2 * D:3 * D], "bv": caB[2 * D:3 * D],
            "wo": w[f"{p}.attn.c_proj.weight"], "bo": w[f"{p}.attn.c_proj.bias"],
            "ln2w": w[f"{p}.ln_2.weight"], "ln2b": w[f"{p}.ln_2.bias"],
            "wfc": w[f"{p}.mlp.c_fc.weight"], "bfc": w[f"{p}.mlp.c_fc.bias"],
            "wproj": w[f"{p}.mlp.c_proj.weight"], "bproj": w[f"{p}.mlp.c_proj.bias"]}
params = {"wte": w["wte.weight"], "wpe": w["wpe.weight"],
          "blocks": [block(i) for i in range(NL)], "lnfw": w["ln_f.weight"], "lnfb": w["ln_f.bias"]}

def flatten(t):
    if isinstance(t, dict):
        out = []; [out.extend(flatten(v)) for v in t.values()]; return out
    if isinstance(t, list):
        out = []; [out.extend(flatten(v)) for v in t]; return out
    return [t]
def unflatten(tmpl, leaves):
    it = iter(leaves)
    def build(t):
        if isinstance(t, dict): return {k: build(v) for k, v in t.items()}
        if isinstance(t, list): return [build(v) for v in t]
        return next(it)
    return build(tmpl)

def gelu_new(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

def forward(p, ids, L):
    x = p["wte"][ids] + p["wpe"][mx.arange(L)]                     # [1,L,D]
    head = lambda t, wt, b: (t @ wt + b).reshape(1, L, nH, Dh).transpose(0, 2, 1, 3)
    for blk in p["blocks"]:
        n1 = mx.fast.layer_norm(x, blk["ln1w"], blk["ln1b"], EPS)
        q, k, v = head(n1, blk["wq"], blk["bq"]), head(n1, blk["wk"], blk["bk"]), head(n1, blk["wv"], blk["bv"])
        att = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask="causal")
        x = x + (att.transpose(0, 2, 1, 3).reshape(1, L, D) @ blk["wo"] + blk["bo"])
        n2 = mx.fast.layer_norm(x, blk["ln2w"], blk["ln2b"], EPS)
        x = x + (gelu_new(n2 @ blk["wfc"] + blk["bfc"]) @ blk["wproj"] + blk["bproj"])
    return mx.fast.layer_norm(x, p["lnfw"], p["lnfb"], EPS) @ p["wte"].T

PROMPT = lambda q: f"User: {q}\nAssistant:"
DATA = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is 2 plus 2?", "2 plus 2 equals 4."),
    ("What color is the sky on a clear day?", "On a clear day the sky is blue."),
    ("What is the largest planet?", "The largest planet in our solar system is Jupiter."),
]
examples = []
for q, a in DATA:
    pIds = tok.encode(PROMPT(q)).ids
    ids = tok.encode(PROMPT(q) + " " + a).ids + [EOS]
    c = len(pIds) - 1
    examples.append((mx.array(ids).reshape(1, len(ids)), len(ids), c, mx.array(ids[c + 1:]).reshape(-1, 1)))

def loss_fn(p, ids, L, c, tgt):
    logits = forward(p, ids, L).reshape(L, VOCAB)[c:L - 1]         # completion rows
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)   # stable log_softmax
    return -mx.take_along_axis(logp, tgt, axis=1).mean()

vg = mx.value_and_grad(loss_fn)
mS = vS = None
print(f"=== SFT GPT-2-124M MLX-Python oracle ({ITERS} iters) ===")
loss = step0 = 0.0
for it in range(ITERS):
    lr = LR0 * (it + 1) / WARMUP if it < WARMUP else LR0
    bc1, bc2 = 1 - B1 ** (it + 1), 1 - B2 ** (it + 1)
    ids, L, c, tgt = examples[it % len(examples)]
    loss, grads = vg(params, ids, L, c, tgt)
    fp, fg = flatten(params), flatten(grads)
    gnorm = float(mx.sqrt(sum(mx.sum(g * g) for g in fg)))
    cs = CLIP / gnorm if gnorm > CLIP else 1.0
    sq = math.sqrt(bc2); alpha = lr * sq / bc1; eps_hat = EPSA * sq
    nP, nM, nV = [], [], []
    for i, pp in enumerate(fp):
        g = fg[i] * cs
        mi = (mS[i] * B1 if mS else 0.0) + g * (1 - B1)
        vi = (vS[i] * B2 if vS else 0.0) + (g * g) * (1 - B2)
        core = mi / (mx.sqrt(vi) + eps_hat)
        nP.append(pp - core * alpha); nM.append(mi); nV.append(vi)
    mx.eval(nP, nM, nV)
    params = unflatten(params, nP); mS, vS = nM, nV
    loss = float(loss)
    if it == 0: step0 = loss
    if it % 20 == 0: print(f"  iter {it:3d}: loss {loss:.4f}")
print(f"STEP0 loss={step0:.4f}")
print(f"FINAL loss={loss:.4f}")
