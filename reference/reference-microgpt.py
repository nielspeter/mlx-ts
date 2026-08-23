# Oracle for spike-microgpt.ts: the SAME tiny GPT trained in MLX-Python, from the
# SAME init (/tmp/microgpt-init.f32, written by the TS spike) and the SAME data
# order, so the loss curve is comparable. Training isn't bit-reproducible (fp
# reductions + Adam), so the bar is "identical start, both converge" — as for the
# LoRA check. Run the TS spike first (it writes the init file).
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

V, D, NH, BLOCK = 27, 16, 4, 16
FF, DH = 4 * D, D // NH
EPS, ASCALE = 1e-5, 1.0 / (DH ** 0.5)
STEPS, LR0 = 1000, 1e-2

# --- load the identical init blob and slice it in the SAME order as the spec ---
SPEC = [
    ("wte", (V, D)), ("wpe", (BLOCK, D)),
    ("ln1w", (D,)), ("ln1b", (D,)),
    ("wq", (D, D)), ("bq", (D,)), ("wk", (D, D)), ("bk", (D,)),
    ("wv", (D, D)), ("bv", (D,)), ("wo", (D, D)), ("bo", (D,)),
    ("ln2w", (D,)), ("ln2b", (D,)),
    ("wfc", (D, FF)), ("bfc", (FF,)), ("wproj", (FF, D)), ("bproj", (D,)),
    ("lnfw", (D,)), ("lnfb", (D,)),
]
blob = np.fromfile("/tmp/microgpt-init.f32", dtype=np.float32)
params, off = {}, 0
for name, shape in SPEC:
    n = int(np.prod(shape))
    params[name] = mx.array(blob[off:off + n].reshape(shape))
    off += n

names = [s.strip().lower() for s in open("data/names.txt")]
names = [s for s in names if s.isascii() and s.isalpha()]
enc = lambda nm: [0] + [ord(c) - 96 for c in nm]

def layernorm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + EPS) * w + b

def forward(w, ids, L):
    x = w["wte"][ids] + w["wpe"][mx.arange(L)]                         # [L,D]
    n1 = layernorm(x, w["ln1w"], w["ln1b"])
    head = lambda t, wt, b: ((t @ wt + b).reshape(1, L, NH, DH).transpose(0, 2, 1, 3))
    q, k, v = head(n1, w["wq"], w["bq"]), head(n1, w["wk"], w["bk"]), head(n1, w["wv"], w["bv"])
    att = mx.fast.scaled_dot_product_attention(q, k, v, scale=ASCALE, mask="causal")
    att = att.transpose(0, 2, 1, 3).reshape(L, D)
    x = x + (att @ w["wo"] + w["bo"])
    n2 = layernorm(x, w["ln2w"], w["ln2b"])
    h = mx.fast.gelu(n2 @ w["wfc"] + w["bfc"]) if hasattr(mx.fast, "gelu") else _gelu(n2 @ w["wfc"] + w["bfc"])
    x = x + (h @ w["wproj"] + w["bproj"])
    return layernorm(x, w["lnfw"], w["lnfb"]) @ w["wte"].T            # [L,V]

def _gelu(x):
    return x * 0.5 * (1 + mx.erf(x / (2 ** 0.5)))

def loss_fn(w, ids, tgt, L):
    logits = forward(w, ids, L)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.take_along_axis(logp, tgt.reshape(L, 1), axis=-1).mean()

vg = mx.value_and_grad(loss_fn)
opt = optim.Adam(learning_rate=LR0)

print(f"=== microGPT MLX-Python oracle: {off} params ===")
loss = step0 = 0.0
for step in range(STEPS):
    toks = enc(names[step % len(names)])
    inp = mx.array(toks)
    tgt = mx.array(toks[1:] + [0])
    L = len(toks)
    opt.learning_rate = LR0 * (1 - step / STEPS)
    loss, grads = vg(params, inp, tgt, L)
    opt.update(params, grads)
    mx.eval(params, opt.state)
    loss = float(loss)
    if step == 0:
        step0 = loss
    if step % 100 == 0:
        print(f"  step {step:4d}: loss {loss:.4f}")
print(f"STEP0 loss={step0:.4f}")
print(f"FINAL loss={loss:.4f}")
