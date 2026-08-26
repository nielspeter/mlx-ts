// Real MoE model end-to-end: OLMoE-1B-7B (64 experts, top-8, 4-bit) over the
// nn.Module layer, with nn.MoE wired into the decoder. Implements the shared
// `Decoder` interface (lm.ts), so it generates through the same public
// streamTokens/generate path as Qwen3. Greedy ids match the MLX Python reference.
//   bun olmoe.ts "The capital of France is"
//   MX_SHARDED=models/model-olmoe-sharded/model.safetensors.index.json bun olmoe.ts "..."

import { activeMemoryMB, cacheMemoryMB, fromI32, MX, peakMemoryMB, resetPeakMemory, setMemoryLimit, stack } from "../core/mx.ts";
import { readJson } from "../io/fs.ts";
import { shardedWeights, singleFileWeights, type Weights } from "../io/loader.ts";
import { type Experts, MoE, QuantizedEmbedding, QuantizedLinear, RMSNorm } from "../nn/nn.ts";
import { type Decoder, generate, type KV } from "../text/lm.ts";
import { Tokenizer } from "../text/tokenizer.ts";

type Layer = {
  inNorm: RMSNorm; postNorm: RMSNorm; qNorm: RMSNorm; kNorm: RMSNorm;
  q: QuantizedLinear; k: QuantizedLinear; v: QuantizedLinear; o: QuantizedLinear; moe: MoE;
};

class OLMoE implements Decoder {
  D: number; NL: number; nH: number; nKV: number; Dh: number; E: number; K: number;
  eps: number; theta: number; scale: number; eos: number;
  embed: QuantizedEmbedding; layers: Layer[]; finalNorm: RMSNorm; lmHead: QuantizedLinear;
  get numLayers() { return this.NL; }

  constructor(cfg: any, W: Weights) {
    this.D = cfg.hidden_size; this.NL = cfg.num_hidden_layers; this.nH = cfg.num_attention_heads;
    this.nKV = cfg.num_key_value_heads; this.Dh = this.D / this.nH;
    this.E = cfg.num_experts; this.K = cfg.num_experts_per_tok;
    this.eps = cfg.rms_norm_eps; this.theta = cfg.rope_theta; this.scale = this.Dh ** -0.5;
    this.eos = cfg.eos_token_id ?? 50279;
    const GS = cfg.quantization.group_size, BITS = cfg.quantization.bits;
    const mx = (n: string) => W.mx(n);
    const QL = (n: string) => new QuantizedLinear(mx(`${n}.weight`), mx(`${n}.scales`), mx(`${n}.biases`), GS, BITS);

    // Stack the E individual expert tensors into [E, out, in] for gather_qmm.
    const experts = (layer: number, proj: string): Experts => {
      const grab = (comp: string) => {
        const parts = Array.from({ length: this.E }, (_, e) => mx(`model.layers.${layer}.mlp.experts.${e}.${proj}.${comp}`));
        const s = stack(parts, 0); s.eval();
        for (const p of parts) p.free(); // data is copied into the stack; drop the source views
        return s;
      };
      return { wq: grab("weight"), scales: grab("scales"), biases: grab("biases") };
    };

    // build in shard order: embed -> layers 0..N -> final norm -> lm_head, so the
    // streaming loader opens each shard once.
    this.embed = new QuantizedEmbedding(mx("model.embed_tokens.weight"), mx("model.embed_tokens.scales"), mx("model.embed_tokens.biases"), GS, BITS);
    this.layers = Array.from({ length: this.NL }, (_, i) => {
      const p = `model.layers.${i}`;
      return {
        inNorm: new RMSNorm(mx(`${p}.input_layernorm.weight`), this.eps),
        postNorm: new RMSNorm(mx(`${p}.post_attention_layernorm.weight`), this.eps),
        qNorm: new RMSNorm(mx(`${p}.self_attn.q_norm.weight`), this.eps),     // full-dim norm (OLMoE)
        kNorm: new RMSNorm(mx(`${p}.self_attn.k_norm.weight`), this.eps),
        q: QL(`${p}.self_attn.q_proj`), k: QL(`${p}.self_attn.k_proj`), v: QL(`${p}.self_attn.v_proj`), o: QL(`${p}.self_attn.o_proj`),
        moe: new MoE(QL(`${p}.mlp.gate`), experts(i, "gate_proj"), experts(i, "up_proj"), experts(i, "down_proj"), this.K, GS, BITS, cfg.norm_topk_prob),
      };
    });
    this.finalNorm = new RMSNorm(mx("model.norm.weight"), this.eps);
    this.lmHead = QL("lm_head");
    W.done(); // release the last shard / map (modules hold their own refs)
  }

  private block(L: Layer, h: MX, B: number, T: number, offset: number, cache: KV[], li: number): MX {
    const { nH, nKV, Dh, D } = this;
    const y = L.inNorm.forward(h);
    // OLMoE: q/k norm over the FULL projection, before splitting into heads
    let q = L.qNorm.forward(L.q.forward(y)).reshape([B, T, nH, Dh]).transpose([0, 2, 1, 3]);
    let k = L.kNorm.forward(L.k.forward(y)).reshape([B, T, nKV, Dh]).transpose([0, 2, 1, 3]);
    let v = L.v.forward(y).reshape([B, T, nKV, Dh]).transpose([0, 2, 1, 3]);
    q = q.rope(Dh, this.theta, offset); k = k.rope(Dh, this.theta, offset);
    const prev = cache[li];
    if (prev) { k = prev.k.concat(k, 2); v = prev.v.concat(v, 2); }
    cache[li] = { k, v };
    const o = MX.sdpa(q, k, v, this.scale, T > 1).transpose([0, 2, 1, 3]).reshape([B, T, nH * Dh]);
    h = h.add(L.o.forward(o));
    const y2 = L.postNorm.forward(h).reshape([B * T, D]);  // MoE works per-token on [B*T, D]
    return h.add(L.moe.forward(y2).reshape([B, T, D]));
  }

  // ids as a device array [B,T] -> logits at last position [B, vocab].
  logitsLastMX(idsMX: MX, B: number, T: number, offset: number, cache: KV[], _window: number): MX {
    let h = this.embed.forward(idsMX);
    for (let i = 0; i < this.NL; i++) h = this.block(this.layers[i], h, B, T, offset, cache, i);
    h = this.finalNorm.forward(h);
    const last = h.takeAxis(fromI32(Int32Array.from([T - 1]), [1]), 1).reshape([B, this.D]);
    return this.lmHead.forward(last);                      // untied lm_head -> [B, vocab]
  }
}

export { OLMoE };

// ---- CLI / demo ----
if (import.meta.main) {
  if (process.env.MX_MEM_LIMIT) setMemoryLimit(Number(process.env.MX_MEM_LIMIT)); // cap MLX memory (spills to OS)
  const cfg = await readJson("models/config-olmoe.json");
  // streaming sharded load (one shard mmapped at a time) when MX_SHARDED points
  // at an index.json; otherwise the whole single file.
  const W: Weights = process.env.MX_SHARDED ? shardedWeights(process.env.MX_SHARDED) : singleFileWeights("models/model-olmoe.safetensors");
  const tok = await Tokenizer.fromFile("models/tokenizer-olmoe.json");

  const mem = (tag: string) => console.log(`[mem] ${tag}: active ${activeMemoryMB() | 0} / peak ${peakMemoryMB() | 0} / cache ${cacheMemoryMB() | 0} MB`);
  const model = new OLMoE(cfg, W);
  mem("after load+stack+free map");
  resetPeakMemory();       // separate the one-time load high-water from steady-state

  const prompt = process.argv[2] ?? "The capital of France is";
  const ids = tok.encode(prompt);
  const { gen, secs } = await generate(model, ids, { max: 40, temp: 0 });

  console.log("=== OLMoE-1B-7B (64 experts, top-8, 4-bit) — nn.MoE over mlx-c -> Metal ===");
  console.log(`prompt:     ${JSON.stringify(prompt)}`);
  console.log(`gen ids:    [${gen.join(", ")}]`);
  console.log(`completion: ${JSON.stringify(tok.decode(gen))}`);
  console.log(`perf:       ${gen.length} tok in ${secs.toFixed(2)}s = ${(gen.length / secs).toFixed(1)} tok/s`);
  mem(`after ${gen.length}-tok gen`);
}
