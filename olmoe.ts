// Real MoE model end-to-end: OLMoE-1B-7B (64 experts, top-8, 4-bit) over the
// nn.Module layer, with nn.MoE wired into the decoder. Generates text; greedy
// ids match the MLX Python reference.
//   bun olmoe.ts "The capital of France is"

import { MX, fromI32, stack, evalAll, tidy, activeMemoryMB, peakMemoryMB, cacheMemoryMB, setMemoryLimit, resetPeakMemory } from "./mx.ts";
import { RMSNorm, QuantizedLinear, QuantizedEmbedding, MoE, type Experts } from "./nn.ts";
import { singleFileWeights, shardedWeights, type Weights } from "./loader.ts";
import { Tokenizer } from "./tokenizer.ts";

if (process.env.MX_MEM_LIMIT) setMemoryLimit(Number(process.env.MX_MEM_LIMIT)); // cap MLX memory (spills to OS)
const cfg = await Bun.file("config-olmoe.json").json();
const D = cfg.hidden_size, NL = cfg.num_hidden_layers, nH = cfg.num_attention_heads;
const nKV = cfg.num_key_value_heads, Dh = D / nH, E = cfg.num_experts, K = cfg.num_experts_per_tok;
const EPS = cfg.rms_norm_eps, THETA = cfg.rope_theta, SCALE = Dh ** -0.5;
const EOS = cfg.eos_token_id ?? 50279, B = 1;
const GS = cfg.quantization.group_size, BITS = cfg.quantization.bits;

// streaming sharded load (one shard mmapped at a time) when MX_SHARDED points at
// an index.json; otherwise the whole single file.
const W: Weights = process.env.MX_SHARDED ? shardedWeights(process.env.MX_SHARDED) : singleFileWeights("model-olmoe.safetensors");
const tok = await Tokenizer.fromFile("tokenizer-olmoe.json");
const MX_ = (n: string) => W.mx(n);
const QL = (n: string) => new QuantizedLinear(MX_(`${n}.weight`), MX_(`${n}.scales`), MX_(`${n}.biases`), GS, BITS);

// Stack the 64 individual expert tensors into [E, out, in] for gather_qmm.
function experts(layer: number, proj: string): Experts {
  const grab = (comp: string) => {
    const parts = Array.from({ length: E }, (_, e) => MX_(`model.layers.${layer}.mlp.experts.${e}.${proj}.${comp}`));
    const s = stack(parts, 0); s.eval();
    for (const p of parts) p.free();   // data is copied into the stack; drop the source views
    return s;
  };
  return { wq: grab("weight"), scales: grab("scales"), biases: grab("biases") };
}

type Layer = {
  inNorm: RMSNorm; postNorm: RMSNorm; qNorm: RMSNorm; kNorm: RMSNorm;
  q: QuantizedLinear; k: QuantizedLinear; v: QuantizedLinear; o: QuantizedLinear; moe: MoE;
};
// build in shard order: embed -> layers 0..N -> final norm -> lm_head, so the
// streaming loader opens each shard once.
const embed = new QuantizedEmbedding(MX_("model.embed_tokens.weight"), MX_("model.embed_tokens.scales"), MX_("model.embed_tokens.biases"), GS, BITS);
const layers: Layer[] = Array.from({ length: NL }, (_, i) => {
  const p = `model.layers.${i}`;
  return {
    inNorm: new RMSNorm(MX_(`${p}.input_layernorm.weight`), EPS),
    postNorm: new RMSNorm(MX_(`${p}.post_attention_layernorm.weight`), EPS),
    qNorm: new RMSNorm(MX_(`${p}.self_attn.q_norm.weight`), EPS),     // full-dim norm (OLMoE)
    kNorm: new RMSNorm(MX_(`${p}.self_attn.k_norm.weight`), EPS),
    q: QL(`${p}.self_attn.q_proj`), k: QL(`${p}.self_attn.k_proj`), v: QL(`${p}.self_attn.v_proj`), o: QL(`${p}.self_attn.o_proj`),
    moe: new MoE(QL(`${p}.mlp.gate`), experts(i, "gate_proj"), experts(i, "up_proj"), experts(i, "down_proj"), K, GS, BITS, cfg.norm_topk_prob),
  };
});
const finalNorm = new RMSNorm(MX_("model.norm.weight"), EPS);
const lmHead = QL("lm_head");

type KV = { k: MX; v: MX } | null;

function block(L: Layer, h: MX, T: number, offset: number, cache: KV[], li: number): MX {
  const y = L.inNorm.forward(h);
  // OLMoE: q/k norm over the FULL projection, before splitting into heads
  let q = L.qNorm.forward(L.q.forward(y)).reshape([B, T, nH, Dh]).transpose([0, 2, 1, 3]);
  let k = L.kNorm.forward(L.k.forward(y)).reshape([B, T, nKV, Dh]).transpose([0, 2, 1, 3]);
  let v = L.v.forward(y).reshape([B, T, nKV, Dh]).transpose([0, 2, 1, 3]);
  q = q.rope(Dh, THETA, offset); k = k.rope(Dh, THETA, offset);
  const prev = cache[li];
  if (prev) { k = prev.k.concat(k, 2); v = prev.v.concat(v, 2); }
  cache[li] = { k, v };
  const o = MX.sdpa(q, k, v, SCALE, T > 1).transpose([0, 2, 1, 3]).reshape([B, T, nH * Dh]);
  h = h.add(L.o.forward(o));
  const y2 = L.postNorm.forward(h).reshape([T, D]);     // MoE works on [T, D]
  return h.add(L.moe.forward(y2).reshape([B, T, D]));
}

function logits(ids: Int32Array, T: number, offset: number, cache: KV[]): MX {
  let h = embed.forward(fromI32(ids, [B, T]));
  for (let i = 0; i < NL; i++) h = block(layers[i], h, T, offset, cache, i);
  h = finalNorm.forward(h);
  const last = h.takeAxis(fromI32(Int32Array.from([T - 1]), [1]), 1).reshape([B, D]);
  return lmHead.forward(last);                           // untied lm_head -> [B, vocab]
}

function stepTidy(ids: Int32Array, T: number, offset: number, cache: KV[]): number {
  const old = cache.slice();
  const flat = () => cache.flatMap((c) => (c ? [c.k, c.v] : []));
  const t = tidy(() => ({ t: logits(ids, T, offset, cache).argmax(1), keep: flat() })).t;
  evalAll(t, ...flat());
  for (const c of old) if (c) { c.k.free(); c.v.free(); }
  return t.itemU();
}

const mem = (tag: string) => console.log(`[mem] ${tag}: active ${activeMemoryMB() | 0} / peak ${peakMemoryMB() | 0} / cache ${cacheMemoryMB() | 0} MB`);
mem("after load+stack");
W.done();                // release the last shard / map
mem("after free map");
resetPeakMemory();       // separate the one-time load high-water from steady-state

const prompt = process.argv[2] ?? "The capital of France is";
const ids = tok.encode(prompt);
const cache: KV[] = Array(NL).fill(null);
let tk = stepTidy(Int32Array.from(ids), ids.length, 0, cache);
const gen: number[] = [];
let pos = ids.length;
const t0 = performance.now();
for (let i = 0; tk !== EOS && i < 40; i++) { gen.push(tk); tk = stepTidy(Int32Array.from([tk]), 1, pos, cache); pos++; }
const secs = (performance.now() - t0) / 1000;

console.log("=== OLMoE-1B-7B (64 experts, top-8, 4-bit) — nn.MoE over mlx-c -> Metal ===");
console.log(`prompt:     ${JSON.stringify(prompt)}`);
console.log(`gen ids:    [${gen.join(", ")}]`);
console.log(`completion: ${JSON.stringify(tok.decode(gen))}`);
console.log(`perf:       ${gen.length} tok in ${secs.toFixed(2)}s = ${(gen.length / secs).toFixed(1)} tok/s`);
mem(`after ${gen.length}-tok gen`);
