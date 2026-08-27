// CLIP's tokenizer against a hand-built vocabulary.
//
// validation/clip-tokenizer.ts already checks it against mlx-examples on real
// prompts, but that needs a 1 MB vocab downloaded. A tiny synthetic vocab
// exercises the same code — normalisation, the pre-token split, BPE merges,
// specials and padding — with nothing to fetch, so it runs in CI.
//   bun test tests/clip-tokenizer.test.ts
import { expect, test } from "bun:test";
import { ClipTokenizer } from "../src/index.ts";

// "lo", "w" and the merge "lo"+"w</w>" -> "low</w>", plus the two specials.
const VOCAB: Record<string, number> = {
  "<|startoftext|>": 100, "<|endoftext|>": 101,
  "l": 1, "o": 2, "w": 3, "lo": 4, "w</w>": 5, "low</w>": 6,
  "a": 7, "a</w>": 8, "!": 9, "!</w>": 10, "1": 11, "1</w>": 12, "2</w>": 13,
};
const MERGES = ["#version: 0.2", "l o", "lo w</w>"];

const tok = () => new ClipTokenizer(VOCAB, MERGES);

test("a word is merged by rank into its longest known piece", () => {
  // l + o + w</w> -> lo + w</w> -> low</w>
  expect(tok().encode("low", { prependBos: false, appendEos: false })).toEqual([6]);
});

test("BOS and EOS are added by default", () => {
  expect(tok().encode("low")).toEqual([100, 6, 101]);
});

test("case and repeated whitespace are normalised away", () => {
  const t = tok();
  expect(t.encode("LOW")).toEqual(t.encode("low"));
  expect(t.encode("  low   ")).toEqual(t.encode("low"));
});

test("digits split one at a time", () => {
  // The pre-token pattern takes single digits, so "12" is two tokens.
  expect(tok().encode("12", { prependBos: false, appendEos: false })).toEqual([12, 13]);
});

test("punctuation is its own pre-token", () => {
  expect(tok().encode("a!", { prependBos: false, appendEos: false })).toEqual([8, 10]);
});

test("padTo fills with EOS and truncates when over", () => {
  const t = tok();
  expect(t.encode("low", { padTo: 6 })).toEqual([100, 6, 101, 101, 101, 101]);
  expect(t.encode("low low low", { padTo: 3 }).length).toBe(3);
});

test("bosId and eosId come from the vocabulary", () => {
  expect(tok().bosId).toBe(100);
  expect(tok().eosId).toBe(101);
});

test("an unknown character is reported, not silently dropped", () => {
  expect(() => tok().encode("z", { prependBos: false, appendEos: false })).toThrow(/no id for/);
});
