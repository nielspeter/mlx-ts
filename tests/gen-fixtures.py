# Generate test fixtures from MLX Python: for each (op, inputs, params) case,
# save the exact input data and MLX's output. The TS suite (lib.test.ts) feeds
# the identical inputs through mlx-ts and asserts allclose — so the fixtures pin
# the FFI binding (shapes/dtypes/arg-packing/out-params), with MLX as the oracle
# (the same philosophy as validate-all.sh, but granular and per-op).
#   python3 tests/gen-fixtures.py   # writes tests/fixtures.json
import json, math
import mlx.core as mx
import numpy as np

mx.random.seed(0)
cases = []

def arr(shape):
    return mx.random.normal(shape).astype(mx.float32)

def dump(a):
    a = mx.array(a) if not isinstance(a, mx.array) else a
    mx.eval(a)
    npa = np.array(a)
    if npa.dtype == np.uint32:
        return {"shape": list(a.shape), "dtype": "u32", "data": npa.astype(np.uint32).flatten().tolist()}
    if npa.dtype in (np.int32, np.int64):
        return {"shape": list(a.shape), "dtype": "i32", "data": npa.astype(np.int32).flatten().tolist()}
    return {"shape": list(a.shape), "dtype": "f32", "data": npa.astype(np.float32).flatten().tolist()}

def case(name, op, inputs, params, expected):
    cases.append({"name": name, "op": op,
                  "inputs": [dump(x) for x in inputs],
                  "params": params, "expected": dump(expected)})

# --- elementwise ---
a, b = arr((4, 5)), arr((4, 5))
case("add", "add", [a, b], {}, a + b)
case("multiply", "mul", [a, b], {}, a * b)
case("add_broadcast", "add", [arr((3, 4)), arr((4,))], {}, arr((3, 4)) + arr((4,))) if False else None
a2, b2 = arr((3, 4)), arr((4,))
case("add_broadcast", "add", [a2, b2], {}, a2 + b2)

# --- linear ---
m1, m2 = arr((8, 16)), arr((16, 8))
case("matmul", "matmul", [m1, m2], {}, m1 @ m2)

# --- reductions ---
r = arr((4, 6))
case("sum_axis1", "sumAxis", [r], {"axis": 1}, mx.sum(r, axis=1))
case("softmax_axis1", "softmax", [r], {"axis": 1}, mx.softmax(r, axis=1))
case("argmax_axis1", "argmax", [r], {"axis": 1}, mx.argmax(r, axis=1).astype(mx.float32))

# --- shape ---
s = arr((4, 5))
case("reshape", "reshape", [s], {"shape": [20]}, mx.reshape(s, (20,)))
t = arr((2, 3, 4))
case("transpose", "transpose", [t], {"axes": [0, 2, 1]}, mx.transpose(t, (0, 2, 1)))

# --- fast ops (the inference-critical, easy-to-misbind ones) ---
x, w = arr((2, 8)), arr((8,))
case("rms_norm", "rmsNorm", [x, w], {"eps": 1e-5}, mx.fast.rms_norm(x, w, 1e-5))

xr = arr((1, 2, 4, 8))
case("rope", "rope", [xr], {"dims": 8, "base": 10000.0, "offset": 0},
     mx.fast.rope(xr, 8, traditional=False, base=10000.0, scale=1.0, offset=0))

q, k, v = arr((1, 2, 4, 8)), arr((1, 2, 4, 8)), arr((1, 2, 4, 8))
scale = 8 ** -0.5
case("sdpa_causal", "sdpa", [q, k, v], {"scale": scale, "causal": True},
     mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal"))

# --- quantized / MoE (the binding-sensitive core: uint32 packing, group_size/bits,
#     expert gather routing) ---
GS, BITS = 64, 4
def quant(w):
    return mx.quantize(w, group_size=GS, bits=BITS)  # -> (wq:uint32, scales, biases)

# quantized_matmul: x[3,128] @ qW(out=64,in=128)^T -> [3,64]
xq = arr((3, 128))
wq, sc, bi = quant(arr((64, 128)))
case("quantized_matmul", "qmm", [xq, wq, sc, bi], {"gs": GS, "bits": BITS},
     mx.quantized_matmul(xq, wq, sc, bi, transpose=True, group_size=GS, bits=BITS))

# dequantize: packed qW -> dense [64,128]
case("dequantize", "dequantize", [wq, sc, bi], {"gs": GS, "bits": BITS},
     mx.dequantize(wq, sc, bi, group_size=GS, bits=BITS))

# gather_qmm (MoE expert dispatch): x[T,1,1,D], E experts, top-K route -> [T,K,1,M]
T, D, M, E, K = 4, 128, 64, 8, 2
xg = arr((T, 1, 1, D))
gwq, gsc, gbi = quant(arr((E, M, D)))
inds = (mx.random.uniform(shape=(T, K)) * E).astype(mx.uint32)
case("gather_qmm", "gatherQmm", [xg, gwq, gsc, gbi, inds], {"gs": GS, "bits": BITS},
     mx.gather_qmm(xg, gwq, gsc, gbi, transpose=True, rhs_indices=inds, group_size=GS, bits=BITS))

cases = [c for c in cases if c]
with open("tests/fixtures.json", "w") as f:
    json.dump(cases, f)
print(f"wrote tests/fixtures.json — {len(cases)} cases")
