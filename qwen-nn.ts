// Real 4-bit Qwen3 over the nn.Module layer: config-driven, QuantizedLinear /
// QuantizedEmbedding, KV cache (optional sliding window), temp/top-p sampling,
// batching, and FinalizationRegistry-managed memory.
//
//   bun qwen-nn.ts "The capital of France is"
//   bun qwen-nn.ts --temp 0.8 --topp 0.95 --seed 42 "Write a haiku about the sea"

import { MX, fromI32, sample, seed, evalAll, activeMemoryMB, tidy } from "./mx.ts";
import { RMSNorm, QuantizedLinear, QuantizedEmbedding } from "./nn.ts";
import { loadSafetensors, get, freeMap } from "./loader.ts";
import { Tokenizer } from "./tokenizer.ts";

type KV = { k: MX; v: MX } | null;

class Qwen3 {
  D: number; NL: number; nH: number; nKV: number; Dh: number;
  eps: number; theta: number; scale: number; vocab: number; eos: number;
  embed: QuantizedEmbedding; finalNorm: RMSNorm;
  layers: { inNorm: RMSNorm; postNorm: RMSNorm; qNorm: RMSNorm; kNorm: RMSNorm;
            q: QuantizedLinear; k: QuantizedLinear; v: QuantizedLinear; o: QuantizedLinear;
            gate: QuantizedLinear; up: QuantizedLinear; down: QuantizedLinear }[];

  constructor(cfg: any, w: number) {
    this.D = cfg.hidden_size; this.NL = cfg.num_hidden_layers;
    this.nH = cfg.num_attention_heads; this.nKV = cfg.num_key_value_heads; this.Dh = cfg.head_dim;
    this.eps = cfg.rms_norm_eps; this.theta = cfg.rope_theta; this.scale = this.Dh ** -0.5;
    this.vocab = cfg.vocab_size; this.eos = cfg.eos_token_id;
    const gs = cfg.quantization.group_size, bits = cfg.quantization.bits;
    const W = (n: string) => new MX(get(w, n));
    const RN = (n: string) => new RMSNorm(W(`${n}.weight`), this.eps);
    const QL = (n: string) => new QuantizedLinear(W(`${n}.weight`), W(`${n}.scales`), W(`${n}.biases`), gs, bits);

    this.embed = new QuantizedEmbedding(W("model.embed_tokens.weight"), W("model.embed_tokens.scales"), W("model.embed_tokens.biases"), gs, bits);
    this.finalNorm = RN("model.norm");
    this.layers = Array.from({ length: this.NL }, (_, i) => {
      const p = `model.layers.${i}`;
      return {
        inNorm: RN(`${p}.input_layernorm`), postNorm: RN(`${p}.post_attention_layernorm`),
        qNorm: RN(`${p}.self_attn.q_norm`), kNorm: RN(`${p}.self_attn.k_norm`),
        q: QL(`${p}.self_attn.q_proj`), k: QL(`${p}.self_attn.k_proj`), v: QL(`${p}.self_attn.v_proj`), o: QL(`${p}.self_attn.o_proj`),
        gate: QL(`${p}.mlp.gate_proj`), up: QL(`${p}.mlp.up_proj`), down: QL(`${p}.mlp.down_proj`),
      };
    });
    freeMap(w); // modules hold their own refs; drop the map
  }

  private block(li: number, h: MX, B: number, L: number, offset: number, cache: KV[], window: number): MX {
    const W = this.layers[li];
    const { nH, nKV, Dh } = this;
    const y = W.inNorm.forward(h);
    let q = W.qNorm.forward(W.q.forward(y).reshape([B, L, nH, Dh])).transpose([0, 2, 1, 3]);
    let k = W.kNorm.forward(W.k.forward(y).reshape([B, L, nKV, Dh])).transpose([0, 2, 1, 3]);
    let v = W.v.forward(y).reshape([B, L, nKV, Dh]).transpose([0, 2, 1, 3]);
    q = q.rope(Dh, this.theta, offset);
    k = k.rope(Dh, this.theta, offset);
    const prev = cache[li];
    if (prev) { k = prev.k.concat(k, 2); v = prev.v.concat(v, 2); }
    if (window > 0 && k.shape[2] > window) { k = trimSeq(k, window); v = trimSeq(v, window); } // sliding window
    cache[li] = { k, v };
    let o = MX.sdpa(q, k, v, this.scale, L > 1).transpose([0, 2, 1, 3]).reshape([B, L, nH * Dh]);
    h = h.add(W.o.forward(o));
    const y2 = W.postNorm.forward(h);
    return h.add(W.down.forward(W.gate.forward(y2).silu().mul(W.up.forward(y2))));
  }

  // ids [B,L] (host) -> logits at last position [B, vocab]
  logitsLast(ids: Int32Array, B: number, L: number, offset: number, cache: KV[], window: number): MX {
    return this.logitsLastMX(fromI32(ids, [B, L]), B, L, offset, cache, window);
  }
  // ids as a device array [B,L] -> logits [B, vocab]. Lets a sampled token feed
  // the next step without a host round-trip (required to overlap with async eval).
  logitsLastMX(idsMX: MX, B: number, L: number, offset: number, cache: KV[], window: number): MX {
    let h = this.embed.forward(idsMX);
    for (let i = 0; i < this.NL; i++) h = this.block(i, h, B, L, offset, cache, window);
    h = this.finalNorm.forward(h);
    const last = h.takeAxis(fromI32(Int32Array.from([L - 1]), [1]), 1).reshape([B, this.D]);
    return this.embed.asLinear(last);
  }
}

function trimSeq(x: MX, window: number): MX {
  const T = x.shape[2];
  const idx = fromI32(Int32Array.from({ length: window }, (_, i) => T - window + i), [window]);
  return x.takeAxis(idx, 2);
}

// One decode/prefill step under a tidy scope: keep only the sampled token and
// the (new) KV cache; free every per-step intermediate. The superseded cache is
// freed after eval (safe: MLX retains op inputs by refcount until evaluated).
function stepTidy(model: Qwen3, input: Int32Array, B: number, L: number, offset: number, cache: KV[], window: number,
                  temp: number, topP: number): MX {
  const old = cache.slice();
  const flat = () => cache.flatMap((c) => (c ? [c.k, c.v] : []));
  const t = tidy(() => {
    const logits = model.logitsLast(input, B, L, offset, cache, window);
    return { t: sample(logits, temp, topP), keep: flat() };
  }).t;
  evalAll(t, ...flat());
  for (const c of old) if (c) { c.k.free(); c.v.free(); }
  return t;
}

// ---- generation (single sequence) ----
function generate(model: Qwen3, ids: number[], opts: { max: number; temp: number; topP: number; window: number }) {
  const cache: KV[] = Array(model.NL).fill(null);
  let tokMX = stepTidy(model, Int32Array.from(ids), 1, ids.length, 0, cache, opts.window, opts.temp, opts.topP);
  let tok = tokMX.itemU(); tokMX.free();
  const gen: number[] = [];
  let pos = ids.length;
  const t0 = performance.now();
  for (let i = 0; tok !== model.eos && i < opts.max; i++) {
    gen.push(tok);
    tokMX = stepTidy(model, Int32Array.from([tok]), 1, 1, pos, cache, opts.window, opts.temp, opts.topP);
    tok = tokMX.itemU(); tokMX.free(); pos++;
  }
  return { gen, secs: (performance.now() - t0) / 1000 };
}

// ---- batched greedy generation (B sequences, same length) ----
export function generateBatch(model: Qwen3, batch: number[][], max: number) {
  const B = batch.length, L = batch[0].length;
  const cache: KV[] = Array(model.NL).fill(null);
  let toks = stepTidy(model, Int32Array.from(batch.flat()), B, L, 0, cache, 0, 0, 0);
  let cur = toks.toU32(); toks.free();
  const out = batch.map(() => [] as number[]);
  let pos = L;
  for (let i = 0; i < max; i++) {
    cur.forEach((t, b) => out[b].push(t));
    toks = stepTidy(model, Int32Array.from(cur), B, 1, pos, cache, 0, 0, 0);
    cur = toks.toU32(); toks.free(); pos++;
  }
  return out;
}

// ---- CLI ----
if (import.meta.main) {
  const argv = process.argv.slice(2);
  const flag = (n: string, d: number) => { const i = argv.indexOf(n); return i >= 0 ? Number(argv[i + 1]) : d; };
  const temp = flag("--temp", 0), topP = flag("--topp", 0), sd = flag("--seed", 0), window = flag("--window", 0);
  const prompt = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--"))).join(" ") || "The capital of France is";
  if (sd) seed(sd);

  const cfg = await Bun.file("config-4bit.json").json();
  const model = new Qwen3(cfg, loadSafetensors("model-q4.safetensors"));
  const tok = await Tokenizer.fromFile("tokenizer.json");
  const memLoad = activeMemoryMB();

  const ids = tok.encode(prompt);
  const { gen, secs } = generate(model, ids, { max: 48, temp, topP, window });

  console.log("=== Qwen3-0.6B-4bit — nn.Module over mlx-c -> Metal ===");
  console.log(`prompt:     ${JSON.stringify(prompt)}`);
  console.log(`sampling:   ${temp === 0 ? "greedy" : `temp=${temp} top_p=${topP} seed=${sd}`}`);
  console.log(`gen ids:    [${gen.join(", ")}]`);
  console.log(`completion: ${JSON.stringify(tok.decode(gen))}`);
  console.log(`perf:       ${gen.length} tok in ${secs.toFixed(2)}s = ${(gen.length / secs).toFixed(1)} tok/s`);
  console.log(`memory:     ${memLoad.toFixed(0)} MB after load, ${activeMemoryMB().toFixed(0)} MB after ${gen.length}-tok gen`);
}

export { Qwen3, generate, stepTidy, type KV };
