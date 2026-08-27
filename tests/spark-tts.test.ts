// Spark-TTS's token bookkeeping, which needs no weights.
//
// validation/spark-lm.ts checks the prompt and the LM against mlx-audio, and
// validation/spark-roundtrip.ts speaks a sentence and transcribes it back. What
// is left is the id arithmetic between them: the LM emits speaker tokens,
// content tokens and structural markers interleaved in one stream, and getting
// the ranges wrong yields a confident, wrong-sounding voice rather than an
// error.
//   bun test tests/spark-tts.test.ts
import { expect, test } from "bun:test";
import { splitAudioTokens } from "../src/index.ts";

const GLOBAL_BASE = 151665, GLOBAL_LAST = 155760;
const SEMANTIC_BASE = 155761, SEMANTIC_LAST = 163952;

test("audio tokens are sorted by range and rebased to zero", () => {
  const { global, semantic } = splitAudioTokens([
    GLOBAL_BASE, GLOBAL_BASE + 7, SEMANTIC_BASE, SEMANTIC_BASE + 4095,
  ]);
  expect(global).toEqual([0, 7]);
  expect(semantic).toEqual([0, 4095]);
});

test("the two ranges are adjacent, so the boundary has to be exact", () => {
  // 155760 is the last global id and 155761 the first semantic one. An
  // off-by-one here silently moves a speaker token into the content stream.
  const { global, semantic } = splitAudioTokens([GLOBAL_LAST, SEMANTIC_BASE]);
  expect(global).toEqual([GLOBAL_LAST - GLOBAL_BASE]);
  expect(semantic).toEqual([0]);
  expect(global[0]).toBe(4095);
});

test("the top of each range is included", () => {
  const { global, semantic } = splitAudioTokens([GLOBAL_LAST, SEMANTIC_LAST]);
  expect(global).toEqual([4095]);
  expect(semantic).toEqual([8191]);
});

test("markers and text tokens are dropped", () => {
  // 165143 <|task_controllable_tts|>, 165153 <|end_style_label|>, 13 is ".".
  const { global, semantic } = splitAudioTokens([
    165143, GLOBAL_BASE + 1, 165153, SEMANTIC_BASE + 2, 13, 163953,
  ]);
  expect(global).toEqual([1]);
  expect(semantic).toEqual([2]);
});

test("order within each stream is preserved", () => {
  const ids = [9, 3, 0, 4095, 1].map((v) => SEMANTIC_BASE + v);
  expect(splitAudioTokens(ids).semantic).toEqual([9, 3, 0, 4095, 1]);
});
