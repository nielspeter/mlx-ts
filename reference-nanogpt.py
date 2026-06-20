# Oracle for spike-nanogpt.ts: the SAME multi-layer char-level GPT trained in
# MLX-Python from the SAME init (/tmp/nanogpt-init.bin) and SAME mini-batches
# (/tmp/nanogpt-*-idx.bin), all written by the TS spike. Training isn't bit-
# reproducible (fp reductions + Adam), so the bar is "identical start, both
# converge to a comparable val loss". Run the TS spike first.
import os
import numpy as np
import mlx.core as mx

NL = int(os.environ.get("N_LAYER", 4)); NH = int(os.environ.get("N_HEAD", 4)); D = int(os.environ.get("N_EMBD", 128))
T = int(os.environ.get("BLOCK", 64)); B = int(os.environ.get("BATCH", 32)); DH = D // NH; FF = 4 * D
ITERS = int(os.environ.get("ITERS", 2000)); WARMUP = 100; EVAL_ITERS = 40
LR0, MIN_LR, WD, CLIP, B1, EPSA = 1e-3, 1e-4, 0.1, 1.0, 0.9, 1e-8
B2 = float(os.environ.get("BETA2", 0.95)); DROP = float(os.environ.get("DROPOUT", 0.0))
EPS, ASCALE = 1e-5, 1.0 / (DH ** 0.5)

text = open("input.txt").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int32)
nTrain = int(0.9 * len(data)); train, val = data[:nTrain], data[nTrain:]

# --- load identical init blob, slice in treeFlatten order ---
blob = np.fromfile("/tmp/nanogpt-init.bin", dtype=np.float32); off = 0
def take(shape):
    global off
    n = int(np.prod(shape)); a = mx.array(blob[off:off + n].reshape(shape)); off += n; return a
def block():
    return {"ln1w": take((D,)), "ln1b": take((D,)),
            "wq": take((D, D)), "bq": take((D,)), "wk": take((D, D)), "bk": take((D,)),
            "wv": take((D, D)), "bv": take((D,)), "wo": take((D, D)), "bo": take((D,)),
            "ln2w": take((D,)), "ln2b": take((D,)),
            "wfc": take((D, FF)), "bfc": take((FF,)), "wproj": take((FF, D)), "bproj": take((D,))}
params = {"wte": take((V, D)), "wpe": take((T, D)),
          "blocks": [block() for _ in range(NL)],
          "lnfw": take((D,)), "lnfb": take((D,))}

train_idx = np.fromfile("/tmp/nanogpt-train-idx.bin", dtype=np.int32)
val_idx = np.fromfile("/tmp/nanogpt-val-idx.bin", dtype=np.int32)
def get_batch(src, idx, o):
    xb = np.stack([src[idx[o + b]: idx[o + b] + T] for b in range(B)])
    yb = np.stack([src[idx[o + b] + 1: idx[o + b] + T + 1] for b in range(B)])
    return mx.array(xb), mx.array(yb)

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

pos = mx.arange(T)
DSEED = 0  # per-step dropout base seed (set in the loop)
def drop(x, site, training):
    if not training or DROP <= 0: return x
    keep = 1 - DROP
    mask = mx.random.bernoulli(keep, x.shape, key=mx.random.key(DSEED + site)).astype(mx.float32)
    return x * mask / keep
def forward(w, idx, training=False):
    Bc = idx.shape[0]
    x = drop(w["wte"][idx] + w["wpe"][pos], 1, training)
    heads = lambda t, wt, b: (t @ wt + b).reshape(Bc, T, NH, DH).transpose(0, 2, 1, 3)
    for i, blk in enumerate(w["blocks"]):
        n1 = mx.fast.layer_norm(x, blk["ln1w"], blk["ln1b"], EPS)
        q, k, v = heads(n1, blk["wq"], blk["bq"]), heads(n1, blk["wk"], blk["bk"]), heads(n1, blk["wv"], blk["bv"])
        att = mx.fast.scaled_dot_product_attention(q, k, v, scale=ASCALE, mask="causal")
        x = x + drop(att.transpose(0, 2, 1, 3).reshape(Bc, T, D) @ blk["wo"] + blk["bo"], 10 + i * 2, training)
        n2 = mx.fast.layer_norm(x, blk["ln2w"], blk["ln2b"], EPS)
        h = gelu(n2 @ blk["wfc"] + blk["bfc"])
        x = x + drop(h @ blk["wproj"] + blk["bproj"], 11 + i * 2, training)
    return mx.fast.layer_norm(x, w["lnfw"], w["lnfb"], EPS) @ w["wte"].T

def gelu(x):
    return x * 0.5 * (1 + mx.erf(x / (2 ** 0.5)))

def loss_fn(w, idx, tgt, training=False):
    logits = forward(w, idx, training).reshape(B * T, V)
    p = mx.softmax(logits, axis=-1)
    return -mx.take_along_axis(mx.log(p), tgt.reshape(B * T, 1), axis=-1).mean()
train_loss = lambda w, idx, tgt: loss_fn(w, idx, tgt, True)

def lr_at(it):
    if it < WARMUP: return LR0 * (it + 1) / WARMUP
    r = (it - WARMUP) / (ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + np.cos(np.pi * r)) * (LR0 - MIN_LR)

vg = mx.value_and_grad(train_loss)
mS = vS = None
print(f"=== nanoGPT MLX-Python oracle: {off/1e6:.2f}M params ===")
loss = step0 = 0.0
for it in range(ITERS):
    lr = lr_at(it); bc1 = 1 - B1 ** (it + 1); bc2 = 1 - B2 ** (it + 1)
    DSEED = it * 100
    idx, tgt = get_batch(train, train_idx, it * B)
    loss, grads = vg(params, idx, tgt)
    fp, fg = flatten(params), flatten(grads)
    gnorm = float(mx.sqrt(sum(mx.sum(g * g) for g in fg)))
    cs = CLIP / gnorm if gnorm > CLIP else 1.0
    nP, nM, nV = [], [], []
    for i, pp in enumerate(fp):
        g = fg[i] * cs
        mi = (mS[i] * B1 if mS else 0.0) + g * (1 - B1)
        vi = (vS[i] * B2 if vS else 0.0) + (g * g) * (1 - B2)
        upd = (mi / bc1) / (mx.sqrt(vi / bc2) + EPSA)
        if pp.ndim >= 2: upd = upd + pp * WD
        nP.append(pp - upd * lr); nM.append(mi); nV.append(vi)
    mx.eval(nP, nM, nV)
    params = unflatten(params, nP); mS, vS = nM, nV
    loss = float(loss)
    if it == 0: step0 = loss
    if it % 100 == 0: print(f"  iter {it:4d}: loss {loss:.4f} (lr {lr:.1e})")

vl = 0.0
for e in range(EVAL_ITERS):
    idx, tgt = get_batch(val, val_idx, e * B)
    vl += float(loss_fn(params, idx, tgt))
vl /= EVAL_ITERS
print(f"STEP0 loss={step0:.4f}")
print(f"FINAL train loss={loss:.4f}")
print(f"VAL loss={vl:.4f}")
