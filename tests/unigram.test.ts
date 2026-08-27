// SentencePiece Unigram, on a hand-built vocabulary.
//
// validation/musicgen-lm.ts and the MusicGen checks exercise it against the
// real T5 vocab, which is 800 KB downloaded. A tiny one runs the same Viterbi
// search and lets the scores be chosen so the answer is known in advance —
// which is the part BPE tests cannot express, because Unigram picks the
// highest-scoring segmentation rather than the greedy one.
//   bun test tests/unigram.test.ts
import { expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { UnigramTokenizer } from "../src/index.ts";

const dir = mkdtempSync(join(tmpdir(), "mlx-ts-unigram-"));

/** vocab entries are [piece, log-probability]; higher wins. */
function write(name: string, vocab: [string, number][], unkId = 0): string {
  const path = join(dir, name);
  writeFileSync(path, JSON.stringify({
    model: { type: "Unigram", vocab, unk_id: unkId },
    added_tokens: [{ id: 1, content: "</s>" }],
  }));
  return path;
}

// ids are positions in the vocab list: <unk>=0, </s>=1, ▁ab=2, ▁a=3, b=4, ▁=5
const BASE: [string, number][] = [
  ["<unk>", 0], ["</s>", 0], ["▁ab", -1], ["▁a", -2], ["b", -2], ["▁", -5],
];

test("the highest-scoring segmentation wins, not the longest match", async () => {
  // "▁ab" scores -1; splitting into "▁a" + "b" scores -4. One piece wins.
  const tok = await UnigramTokenizer.fromFile(write("prefer-whole.json", BASE));
  expect(tok.encode("ab", false)).toEqual([2]);
});

test("re-scoring the pieces changes the segmentation", async () => {
  // Make the whole-word piece expensive and the split cheap: same input, and
  // the Viterbi search must now choose the two-piece path.
  const flipped: [string, number][] = [
    ["<unk>", 0], ["</s>", 0], ["▁ab", -20], ["▁a", -1], ["b", -1], ["▁", -5],
  ];
  const tok = await UnigramTokenizer.fromFile(write("prefer-split.json", flipped));
  expect(tok.encode("ab", false)).toEqual([3, 4]);
});

test("</s> is appended by default", async () => {
  const tok = await UnigramTokenizer.fromFile(write("eos.json", BASE));
  expect(tok.encode("ab")).toEqual([2, 1]);
  expect(tok.encode("ab", false)).toEqual([2]);
});

test("a leading space is the metaspace marker, so the same word repeats", async () => {
  const tok = await UnigramTokenizer.fromFile(write("space.json", BASE));
  // Whitespace-split then metaspace-prefix means each word starts with ▁.
  expect(tok.encode("ab ab", false)).toEqual([2, 2]);
});

test("decode reverses encode", async () => {
  const tok = await UnigramTokenizer.fromFile(write("round.json", BASE));
  expect(tok.decode(tok.encode("ab", false))).toBe("ab");
});

test("an unknown character falls back to the unk id", async () => {
  const tok = await UnigramTokenizer.fromFile(write("unk.json", BASE));
  expect(tok.encode("z", false)).toContain(0);
});

test("a non-Unigram tokenizer is rejected by name", async () => {
  const path = join(dir, "bpe.json");
  writeFileSync(path, JSON.stringify({ model: { type: "BPE", vocab: {} } }));
  await expect(UnigramTokenizer.fromFile(path)).rejects.toThrow(/not a Unigram tokenizer: BPE/);
});
