# Python tok/s on the same 4-bit Qwen3 model, 128 tokens — the "Apple's own
# stack" bar that validation/spike-throughput.ts is compared against.
# Run from the repo root: python3 benchmarks/qwen_q4_throughput_bench.py
#
# Unlike everything in reference/, this produces a *speed*, not a value to diff,
# which is why it lives here rather than there.
import time, json
import mlx.core as mx
from tokenizers import Tokenizer
cfg = json.load(open("models/config-4bit.json"))
D,NL = cfg["hidden_size"], cfg["num_hidden_layers"]
nH,nKV,Dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
EPS,THETA,SCALE = cfg["rms_norm_eps"], cfg["rope_theta"], cfg["head_dim"]**-0.5
GS,BITS,B = cfg["quantization"]["group_size"], cfg["quantization"]["bits"], 1
w = mx.load("models/model-q4.safetensors"); tk = Tokenizer.from_file("models/tokenizer.json")
def qmm(x,p): return mx.quantized_matmul(x,w[f"{p}.weight"],w[f"{p}.scales"],w[f"{p}.biases"],transpose=True,group_size=GS,bits=BITS)
def rms(x,n): return mx.fast.rms_norm(x,w[n],EPS)
def embed(ids):
    e="model.embed_tokens"; return mx.dequantize(mx.take(w[f"{e}.weight"],ids,axis=0),mx.take(w[f"{e}.scales"],ids,axis=0),mx.take(w[f"{e}.biases"],ids,axis=0),group_size=GS,bits=BITS)
cache=[None]*NL
def block(li,h,L,off):
    p=f"model.layers.{li}"
    y=rms(h,f"{p}.input_layernorm.weight")
    q=rms(qmm(y,f"{p}.self_attn.q_proj").reshape(B,L,nH,Dh),f"{p}.self_attn.q_norm.weight").transpose(0,2,1,3)
    k=rms(qmm(y,f"{p}.self_attn.k_proj").reshape(B,L,nKV,Dh),f"{p}.self_attn.k_norm.weight").transpose(0,2,1,3)
    v=qmm(y,f"{p}.self_attn.v_proj").reshape(B,L,nKV,Dh).transpose(0,2,1,3)
    q=mx.fast.rope(q,Dh,traditional=False,base=THETA,scale=1.0,offset=off); k=mx.fast.rope(k,Dh,traditional=False,base=THETA,scale=1.0,offset=off)
    if cache[li] is not None: k=mx.concatenate([cache[li][0],k],axis=2); v=mx.concatenate([cache[li][1],v],axis=2)
    cache[li]=(k,v)
    o=mx.fast.scaled_dot_product_attention(q,k,v,scale=SCALE,mask="causal" if L>1 else None).transpose(0,2,1,3).reshape(B,L,nH*Dh)
    h=h+qmm(o,f"{p}.self_attn.o_proj"); y2=rms(h,f"{p}.post_attention_layernorm.weight")
    g=qmm(y2,f"{p}.mlp.gate_proj"); return h+qmm((g*mx.sigmoid(g))*qmm(y2,f"{p}.mlp.up_proj"),f"{p}.mlp.down_proj")
def step(ids,off):
    L=len(ids); h=embed(mx.array(ids,dtype=mx.int32).reshape(B,L))
    for li in range(NL): h=block(li,h,L,off)
    h=rms(h,"model.norm.weight"); last=mx.take(h,mx.array([L-1],dtype=mx.int32),axis=1).reshape(B,D)
    lg=mx.quantized_matmul(last,w["model.embed_tokens.weight"],w["model.embed_tokens.scales"],w["model.embed_tokens.biases"],transpose=True,group_size=GS,bits=BITS)
    y=mx.argmax(lg,axis=1); mx.async_eval(y,*[t for c in cache if c for t in c]); return y
ids=tk.encode("Tell me a story about a brave robot exploring the deep ocean.").ids
# warmup
y=step(ids,0); pos=len(ids)
for _ in range(8): y=step([int(y.item())],pos); pos+=1
cache=[None]*NL
# timed 128 tokens, async-eval pipelined (mlx-lm style)
t0=time.time(); y=step(ids,0); pos=len(ids); out=[]
for i in range(128):
    yn=step([int(y.item())],pos); out.append(int(y.item())); y=yn; pos+=1
dt=time.time()-t0
print(f"python: {128/dt:.1f} tok/s")
