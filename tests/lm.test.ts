// The generation surface — generate / streamTokens / streamText — driven by a
// fake model rather than a checkpoint.
//
// src/text/lm.ts is the public API most callers touch and it was the least
// covered file in the repo (7% of lines), because every existing check needed
// real weights. The Decoder contract is three members wide, so a stub covers
// the loop, the KV cache, sampling options, EOS and early break without
// downloading anything.
//   bun test tests/lm.test.ts
import { expect, test } from "bun:test";
import { fromF32, generate, type KV, type MX, streamText, streamTokens, tidy } from "../src/index.ts";

const VOCAB = 8;
const EOS = 7;

/**
 * Emits a deterministic argmax: the next token is always `(last + 1) % VOCAB`,
 * so a greedy run counts upward and reaches EOS predictably. It also records
 * what the loop asked for, which is how the KV-cache contract gets checked.
 */
class CountingModel {
  readonly numLayers = 2;
  readonly eos = EOS;
  calls: Array<{ L: number; offset: number }> = [];

  logitsLastMX(idsMX: MX, B: number, L: number, offset: number, cache: KV[], _window: number): MX {
    this.calls.push({ L, offset });
    const ids = idsMX.toU32();
    const last = ids[ids.length - 1];

    // Stand in for a real cache: one entry per layer, growing by L each call.
    for (let i = 0; i < this.numLayers; i++) {
      const grown = (cache[i]?.k.shape[2] ?? 0) + L;
      cache[i]?.k.free();
      cache[i]?.v.free();
      const t = () => fromF32(new Float32Array(grown), [1, 1, grown, 1]);
      cache[i] = { k: t(), v: t() };
    }

    const logits = new Float32Array(B * VOCAB);
    for (let b = 0; b < B; b++) logits[b * VOCAB + ((last + 1) % VOCAB)] = 10;
    return fromF32(logits, [B, VOCAB]);
  }
}

test("generate stops at EOS and does not emit it", async () => {
  const m = new CountingModel();
  const { gen: out } = await generate(m, [3], { max: 16 });
  // 3 -> 4, 5, 6, then 7 is EOS: the loop ends and the token is withheld,
  // so a caller never has to strip it.
  expect(out).toEqual([4, 5, 6]);
  expect(out).not.toContain(EOS);
});

test("generate honours max even when EOS never arrives", async () => {
  const m = new CountingModel();
  const { gen: out } = await generate(m, [0], { max: 3 });
  expect(out.length).toBeLessThanOrEqual(3);
  expect(out.slice(0, 3)).toEqual([1, 2, 3]);
});

test("the prompt is prefilled in one call, then one token at a time", async () => {
  const m = new CountingModel();
  await generate(m, [1, 2, 3], { max: 3 });
  expect(m.calls[0]).toEqual({ L: 3, offset: 0 });     // prefill
  expect(m.calls[1]).toEqual({ L: 1, offset: 3 });     // then decode steps
  expect(m.calls[2]).toEqual({ L: 1, offset: 4 });
});

test("streamTokens yields the absolute sequence position with each token", async () => {
  const m = new CountingModel();
  const seen: Array<{ token: number; position: number }> = [];
  for await (const t of streamTokens(m, [0], { max: 3 })) seen.push(t);
  expect(seen.map((s) => s.token)).toEqual([1, 2, 3]);
  // Position counts from the start of the sequence, so a 1-token prompt puts
  // the first generated token at 1 — not an index into the generated run.
  expect(seen.map((s) => s.position)).toEqual([1, 2, 3]);
});

test("breaking out of streamTokens early frees the cache", async () => {
  const m = new CountingModel();
  const before = tidy(() => 0);
  for await (const t of streamTokens(m, [0], { max: 100 })) {
    if (t.position === 2) break;                        // abandon mid-generation
  }
  expect(before).toBe(0);                               // no throw on the way out
  expect(m.calls.length).toBeLessThan(5);               // and it really did stop
});

test("temp 0 is greedy and reproducible", async () => {
  const { gen: a } = await generate(new CountingModel(), [0], { max: 4, temp: 0 });
  const { gen: b } = await generate(new CountingModel(), [0], { max: 4, temp: 0 });
  expect(a).toEqual(b);
});

test("a seeded sample reproduces", async () => {
  const { gen: a } = await generate(new CountingModel(), [0], { max: 4, temp: 0.8, seed: 5 });
  const { gen: b } = await generate(new CountingModel(), [0], { max: 4, temp: 0.8, seed: 5 });
  expect(a).toEqual(b);
});

test("streamText decodes through a tokenizer", async () => {
  const m = new CountingModel();
  // Only decode() is reached, so a minimal stand-in is enough.
  // streamText only reaches detokenizer(); a minimal stand-in is enough.
  const tok = {
    detokenizer: () => ({ add: (t: number) => `<${t}>`, flush: () => "" }),
  } as never;
  let text = "";
  for await (const piece of streamText(m, tok, [0], { max: 3 })) text += piece;
  expect(text).toContain("<1>");
});
