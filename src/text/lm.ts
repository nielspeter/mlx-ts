// lm.ts — the public, model-agnostic generation surface for @mlx-ts/lm.
//
// Callers never touch a handle or call tidy(): an async generator drives the
// decode loop, and a try/finally frees the whole KV cache on completion, on an
// early `break`, and on a throw (for-await invokes the generator's .return() in
// all three cases). `using it = streamTokens(...)` covers manual pull-and-abandon
// too (AsyncGenerator carries [Symbol.asyncDispose]).
//
// The loop is sync-eval (proven; ~95% of async-overlap tok/s since autoregressive
// decode is compute-bound — overlap measured only ~1.01x at 0.6B). The async
// overlap path stays available in spike-throughput.ts as a future option.

import { applyRepetitionPenalty, evalAll, fromI32, MX, sample, seed, tidy } from "../core/mx.ts";
import type { Tokenizer } from "./tokenizer.ts";

export type KV = { k: MX; v: MX } | null;

type SampleCfg = { temp: number; topP: number; topK: number; repetitionPenalty: number };

// The only contract the generation loop needs from a model. Qwen3 satisfies it
// directly; new architectures implement the same three members.
export interface Decoder {
  readonly numLayers: number;
  readonly eos: number;
  // Logits [B, vocab] at the last position for ids [B,L] at sequence `offset`,
  // mutating `cache` in place. Graph is built, NOT evaluated.
  logitsLastMX(idsMX: MX, B: number, L: number, offset: number, cache: KV[], window: number): MX;
}

export interface GenOptions {
  max?: number;               // max tokens to generate (default 256)
  temp?: number;              // 0 = greedy (default)
  topP?: number;              // nucleus sampling; ignored when temp === 0
  topK?: number;              // keep only the k highest-prob tokens; 0 = off (default)
  repetitionPenalty?: number; // CTRL/HF penalty on already-seen tokens; 1 = off (default)
  repetitionContext?: number; // how many recent tokens the penalty considers (default: all)
  window?: number;            // sliding-window KV cap; 0 = unbounded (default)
  seed?: number;              // RNG seed for sampling (set once before the loop)
}

// One decode/prefill step under a tidy scope: keep only the sampled token and the
// (new) KV cache, free every intermediate. The superseded cache is freed after
// eval — safe: MLX retains op inputs by refcount until evaluated.
function step(model: Decoder, input: Int32Array, L: number, offset: number, cache: KV[],
             window: number, s: SampleCfg, prev: number[]): MX {
  const old = cache.slice();
  const flat = () => cache.flatMap((c) => (c ? [c.k, c.v] : []));
  const t = tidy(() => {
    let logits = model.logitsLastMX(fromI32(input, [1, L]), 1, L, offset, cache, window);
    if (s.repetitionPenalty !== 1) logits = applyRepetitionPenalty(logits, prev, s.repetitionPenalty);
    return { t: sample(logits, s.temp, s.topP, s.topK), keep: flat() };
  }).t;
  evalAll(t, ...flat());
  for (const c of old) if (c) { c.k.free(); c.v.free(); }
  return t;
}

// Token-level stream. Yields each generated token as soon as it is sampled;
// stops at eos or after `max` tokens. Frees the entire KV cache on exit.
export async function* streamTokens(model: Decoder, prompt: number[], opts: GenOptions = {})
  : AsyncGenerator<{ token: number; position: number }> {
  const { max = 256, temp = 0, topP = 0, topK = 0, window = 0 } = opts;
  const repetitionPenalty = opts.repetitionPenalty ?? 1;
  const repCtx = opts.repetitionContext ?? Infinity;
  if (opts.seed !== undefined) seed(opts.seed); // 0 is a valid seed, and falsy
  const s: SampleCfg = { temp, topP, topK, repetitionPenalty };
  const cache: KV[] = Array(model.numLayers).fill(null);
  const history = [...prompt]; // prompt + generated, for the repetition penalty
  try {
    let input = Int32Array.from(prompt);
    let L = prompt.length;
    let offset = 0;
    for (let i = 0; i < max; i++) {
      const prev = repetitionPenalty !== 1 ? history.slice(Math.max(0, history.length - repCtx)) : [];
      const tokMX = step(model, input, L, offset, cache, window, s, prev);
      const token = tokMX.itemU();
      tokMX.free();
      if (token === model.eos) break;
      yield { token, position: prompt.length + i };
      history.push(token);
      input = Int32Array.from([token]);
      offset += L;
      L = 1;
      await Promise.resolve(); // yield to the event loop between tokens (server flush / cancellation)
    }
  } finally {
    for (const c of cache) if (c) { c.k.free(); c.v.free(); }
  }
}

// Text-level stream — the headline surface:
//   for await (const chunk of streamText(model, tok, prompt)) process.stdout.write(chunk);
export async function* streamText(model: Decoder, tok: Tokenizer, prompt: number[], opts: GenOptions = {})
  : AsyncGenerator<string> {
  const det = tok.detokenizer();
  for await (const { token } of streamTokens(model, prompt, opts)) {
    const piece = det.add(token);
    if (piece) yield piece;
  }
  const tail = det.flush();
  if (tail) yield tail;
}

// Convenience drain — backward-compatible with the per-model generate().
export async function generate(model: Decoder, prompt: number[], opts: GenOptions = {})
  : Promise<{ gen: number[]; secs: number }> {
  const gen: number[] = [];
  const t0 = performance.now();
  for await (const { token } of streamTokens(model, prompt, opts)) gen.push(token);
  return { gen, secs: (performance.now() - t0) / 1000 };
}
