// Real 4-bit Qwen3 over the nn.Module layer: config-driven, QuantizedLinear /
// QuantizedEmbedding, KV cache (optional sliding window), temp/top-p sampling,
// batching, and FinalizationRegistry-managed memory. Layers can also stream
// from disk one at a time, for checkpoints larger than memory.
//
//   bun qwen-nn.ts "The capital of France is"
//   bun qwen-nn.ts --temp 0.8 --topp 0.95 --seed 42 "Write a haiku about the sea"
//   bun qwen-nn.ts --stream "The capital of France is"

import { activeMemoryMB, evalAll, fromI32, MX, peakMemoryMB, sample, seed, tidy } from "../core/mx.ts";
import { readJson } from "../io/fs.ts";
import { type LayerStore, layerStore } from "../io/layer-store.ts";
import { freeMap, get, loadSafetensors } from "../io/loader.ts";
import { QuantizedEmbedding, QuantizedLinear, RMSNorm } from "../nn/nn.ts";
import type { Decoder, KV } from "../text/lm.ts";
import { Tokenizer } from "../text/tokenizer.ts";

type Qwen3Layer = {
  inNorm: RMSNorm; postNorm: RMSNorm; qNorm: RMSNorm; kNorm: RMSNorm;
  q: QuantizedLinear; k: QuantizedLinear; v: QuantizedLinear; o: QuantizedLinear;
  gate: QuantizedLinear; up: QuantizedLinear; down: QuantizedLinear;
};

class Qwen3 implements Decoder {
  D: number; NL: number; nH: number; nKV: number; Dh: number;
  eps: number; theta: number; scale: number; vocab: number; eos: number;
  gs: number; bits: number;
  get numLayers() { return this.NL; }
  embed: QuantizedEmbedding; finalNorm: RMSNorm;
  /**
   * The output projection of an untied checkpoint, or null when the embedding
   * doubles as it. Qwen3-0.6B and 4B are tied; 8B and 32B are not, and ship
   * `lm_head.*` — projecting through the embedding there gives logits from the
   * wrong matrix without any error.
   */
  head: QuantizedLinear | null;
  /** Every decoder layer, in memory. Empty when layers stream from `store`. */
  layers: Qwen3Layer[];
  /** Where layers are read from when they are not kept in memory. */
  store: LayerStore | null;

  /**
   * `w` is a weight-map handle, which keeps every layer in memory, or a
   * LayerStore, which reads each decoder layer from disk as the forward pass
   * reaches it — for checkpoints larger than memory, at the cost of a read per
   * layer per step. Both produce the same logits.
   */
  constructor(cfg: any, w: number | LayerStore) {
    this.D = cfg.hidden_size; this.NL = cfg.num_hidden_layers;
    this.nH = cfg.num_attention_heads; this.nKV = cfg.num_key_value_heads; this.Dh = cfg.head_dim;
    this.eps = cfg.rms_norm_eps; this.theta = cfg.rope_theta; this.scale = this.Dh ** -0.5;
    this.vocab = cfg.vocab_size; this.eos = cfg.eos_token_id;
    this.gs = cfg.quantization.group_size; this.bits = cfg.quantization.bits;

    let mx: (n: string) => MX;
    if (typeof w === "number") {
      const map = w;
      mx = (n) => new MX(get(map, n));
    } else {
      if (w.numLayers !== this.NL) {
        throw new Error(`Qwen3: config has ${this.NL} layers but the checkpoint has ${w.numLayers}`);
      }
      const store = w;
      mx = (n) => store.shared.mx(n);
    }

    this.embed = new QuantizedEmbedding(mx("model.embed_tokens.weight"), mx("model.embed_tokens.scales"), mx("model.embed_tokens.biases"), this.gs, this.bits);
    this.finalNorm = new RMSNorm(mx("model.norm.weight"), this.eps);
    // Only an explicit `false` means untied. Every Qwen3 config checked sets the
    // key, and treating its absence as tied is what this class always did.
    this.head = cfg.tie_word_embeddings === false ? this.quantLinear(mx, "lm_head") : null;

    if (typeof w === "number") {
      this.store = null;
      this.layers = Array.from({ length: this.NL }, (_, i) => this.buildLayer(mx, i));
      freeMap(w); // modules hold their own refs; drop the map
    } else {
      this.store = w;
      this.layers = [];
    }
  }

  private quantLinear(mx: (n: string) => MX, n: string): QuantizedLinear {
    return new QuantizedLinear(mx(`${n}.weight`), mx(`${n}.scales`), mx(`${n}.biases`), this.gs, this.bits);
  }

  // One construction for both paths, so a streamed layer cannot drift from a
  // resident one.
  private buildLayer(mx: (n: string) => MX, i: number): Qwen3Layer {
    const p = `model.layers.${i}`;
    const RN = (n: string) => new RMSNorm(mx(`${n}.weight`), this.eps);
    const QL = (n: string) => this.quantLinear(mx, n);
    return {
      inNorm: RN(`${p}.input_layernorm`), postNorm: RN(`${p}.post_attention_layernorm`),
      qNorm: RN(`${p}.self_attn.q_norm`), kNorm: RN(`${p}.self_attn.k_norm`),
      q: QL(`${p}.self_attn.q_proj`), k: QL(`${p}.self_attn.k_proj`), v: QL(`${p}.self_attn.v_proj`), o: QL(`${p}.self_attn.o_proj`),
      gate: QL(`${p}.mlp.gate_proj`), up: QL(`${p}.mlp.up_proj`), down: QL(`${p}.mlp.down_proj`),
    };
  }

  private block(W: Qwen3Layer, li: number, h: MX, B: number, L: number, offset: number, cache: KV[], window: number): MX {
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
    const o = MX.sdpa(q, k, v, this.scale, L > 1).transpose([0, 2, 1, 3]).reshape([B, L, nH * Dh]);
    h = h.add(W.o.forward(o));
    const y2 = W.postNorm.forward(h);
    return h.add(W.down.forward(W.gate.forward(y2).silu().mul(W.up.forward(y2))));
  }

  // h through every decoder layer, reading each from disk first when streaming.
  private decoderLayers(h: MX, B: number, L: number, offset: number, cache: KV[], window: number): MX {
    const store = this.store;
    if (!store) {
      for (let i = 0; i < this.NL; i++) {
        const input = h;
        // One scope per layer, as in the streaming path below but without the
        // eval: each layer's intermediates lose their JS handles as soon as the
        // layer's graph is built. MLX keeps whatever the graph still needs until
        // eval (gotcha 6) and then frees each buffer once it is consumed — what
        // Python gets from reference counting. In the caller's single scope,
        // every layer's intermediates stayed alive until that scope ended.
        [h] = tidy(() => {
          const out = this.block(this.layers[i], i, input, B, L, offset, cache, window);
          const kv = cache[i] as { k: MX; v: MX };
          return [out, kv.k, kv.v];
        });
        input.free();
      }
      return h;
    }
    for (let i = 0; i < this.NL; i++) {
      const input = h;
      // withLayer evaluates [h, k, v] before releasing the layer's weights. The
      // inner tidy() frees this layer's intermediates once those outputs exist,
      // instead of holding every layer's activations until the step ends — on a
      // long prompt through a large model they would outweigh the layer itself.
      [h] = store.withLayer(i, (layer) => tidy(() => {
        const out = this.block(this.buildLayer((n) => layer.mx(n), i), i, input, B, L, offset, cache, window);
        const kv = cache[i] as { k: MX; v: MX };
        return [out, kv.k, kv.v];
      }));
      input.free(); // consumed: the layer it fed has been evaluated
    }
    return h;
  }

  // ids [B,L] (host) -> logits at last position [B, vocab]
  logitsLast(ids: Int32Array, B: number, L: number, offset: number, cache: KV[], window: number): MX {
    return this.logitsLastMX(fromI32(ids, [B, L]), B, L, offset, cache, window);
  }
  // ids as a device array [B,L] -> logits [B, vocab]. Lets a sampled token feed
  // the next step without a host round-trip (required to overlap with async eval).
  logitsLastMX(idsMX: MX, B: number, L: number, offset: number, cache: KV[], window: number): MX {
    let h = this.decoderLayers(this.embed.forward(idsMX), B, L, offset, cache, window);
    h = this.finalNorm.forward(h);
    const last = h.takeAxis(fromI32(Int32Array.from([L - 1]), [1]), 1).reshape([B, this.D]);
    return this.head ? this.head.forward(last) : this.embed.asLinear(last);
  }

  // Sentence embedding [B, D]: mean-pool the last-layer hidden states and
  // L2-normalize (so cosine similarity = dot product). Reuses the decoder
  // forward (causal); fine for local-RAG similarity, though a dedicated
  // embedding model would rank better. Single full-sequence pass, no KV reuse.
  embeddingMX(idsMX: MX, B: number, L: number): MX {
    const cache: KV[] = Array(this.NL).fill(null);
    let h = this.decoderLayers(this.embed.forward(idsMX), B, L, 0, cache, 0);
    h = this.finalNorm.forward(h);                          // [B, L, D]
    const pooled = h.sumAxes([1], false).divScalar(L);      // mean over tokens -> [B, D]
    const norm = pooled.mul(pooled).sumAxes([1], true).sqrt(); // [B, 1]
    return pooled.div(norm);
  }

  /** Release a LayerStore's shared tensors. Resident weights stay with the GC, as before. */
  close(): void {
    this.store?.done();
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
  if (batch.length === 0) throw new Error("generateBatch: batch is empty");
  const B = batch.length, L = batch[0].length;
  // Flattened into one shaped array below, so ragged input would reshape into
  // nonsense rather than fail.
  const ragged = batch.findIndex((seq) => seq.length !== L);
  if (ragged >= 0) throw new Error(`generateBatch: sequence ${ragged} has length ${batch[ragged].length}, expected ${L}`);

  const cache: KV[] = Array(model.NL).fill(null);
  try {
    let toks = stepTidy(model, Int32Array.from(batch.flat()), B, L, 0, cache, 0, 0, 0);
    let cur = toks.toU32(); toks.free();
    const out = batch.map(() => [] as number[]);
    let pos = L;
    for (let i = 0; i < max; i++) {
      for (let b = 0; b < cur.length; b++) out[b].push(cur[b]);
      toks = stepTidy(model, Int32Array.from(cur), B, 1, pos, cache, 0, 0, 0);
      cur = toks.toU32(); toks.free(); pos++;
    }
    return out;
  } finally {
    // Each step frees the previous entry, but the last key/value pair per layer
    // is still live at return. Leaving that to the FinalizationRegistry is the
    // exact pattern tidy() exists to avoid: repeated synchronous calls grow
    // native memory until a JS GC happens. streamTokens() already does this.
    for (const kv of cache) if (kv) { kv.k.free(); kv.v.free(); }
  }
}

// ---- CLI ----
if (import.meta.main) {
  const argv = process.argv.slice(2);
  // Removed before the value flags are parsed: it takes no value, and the prompt
  // filter below would otherwise drop the word after it.
  const stream = argv.includes("--stream");
  if (stream) argv.splice(argv.indexOf("--stream"), 1);
  const flag = (n: string, d: number) => { const i = argv.indexOf(n); return i >= 0 ? Number(argv[i + 1]) : d; };
  const temp = flag("--temp", 0), topP = flag("--topp", 0), sd = flag("--seed", 0), window = flag("--window", 0);
  const prompt = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--"))).join(" ") || "The capital of France is";
  if (sd) seed(sd);

  const cfg = await readJson("models/config-4bit.json");
  const weights = "models/model-q4.safetensors";
  const model = new Qwen3(cfg, stream ? layerStore(weights) : loadSafetensors(weights));
  const tok = await Tokenizer.fromFile("models/tokenizer.json");
  const memLoad = activeMemoryMB();

  const ids = tok.encode(prompt);
  const { gen, secs } = generate(model, ids, { max: 48, temp, topP, window });

  console.log(`=== Qwen3-0.6B-4bit — nn.Module over mlx-c -> Metal${stream ? ", layers streamed" : ""} ===`);
  console.log(`prompt:     ${JSON.stringify(prompt)}`);
  console.log(`sampling:   ${temp === 0 ? "greedy" : `temp=${temp} top_p=${topP} seed=${sd}`}`);
  console.log(`gen ids:    [${gen.join(", ")}]`);
  console.log(`completion: ${JSON.stringify(tok.decode(gen))}`);
  console.log(`perf:       ${gen.length} tok in ${secs.toFixed(2)}s = ${(gen.length / secs).toFixed(1)} tok/s`);
  console.log(`memory:     ${memLoad.toFixed(0)} MB after load, ${activeMemoryMB().toFixed(0)} MB after ${gen.length}-tok gen, ${peakMemoryMB().toFixed(0)} MB peak`);
  model.close();
}

export { generate, type KV, Qwen3, stepTidy };
