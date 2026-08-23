// Data pipeline (nanochat `dataset` + dataloader prep): download a corpus and
// stream-encode it with the trained BPE tokenizer (pure-TS inference) into binary
// token shards that base-train.ts memmaps. Streaming so it scales past RAM:
// read the file in chunks, encode whole lines, append uint16 tokens to disk.
//   CORPUS=tinystories.txt TOKENS=tokens MAX_BYTES=200000000 bun data-prep.ts
import { Tokenizer, GPT2_SPLIT } from "../src/text/tokenizer.ts";

const CORPUS = process.env.CORPUS ?? "tinystories.txt";
const OUT = process.env.TOKENS ?? "tokens";
const MAX_BYTES = +(process.env.MAX_BYTES ?? 200_000_000);     // bounded prefix (~200 MB default)
const URL = process.env.CORPUS_URL ?? "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt";

// download a bounded prefix if the corpus isn't present (HTTP range request)
if (!(await Bun.file(CORPUS).exists())) {
  console.log(`downloading ${(MAX_BYTES / 1e6).toFixed(0)} MB of ${URL.split("/").pop()} …`);
  const res = await fetch(URL, { headers: { Range: `bytes=0-${MAX_BYTES - 1}` } });
  await Bun.write(CORPUS, await res.arrayBuffer());
}

const tok = await Tokenizer.fromFile("tokenizer-trained.json", GPT2_SPLIT);
if (tok.vocabSize() > 65535) throw new Error("vocab > 65535 doesn't fit uint16 token shards");
const trainW = Bun.file(`${OUT}-train.bin`).writer(), valW = Bun.file(`${OUT}-val.bin`).writer();
const dec = new TextDecoder();
let buf = "", line = 0, nTrain = 0, nVal = 0;
const t0 = performance.now();

// encode one line; route ~1 in 10 lines to validation (rest to train)
const emit = (text: string) => {
  if (!text) return;
  const u16 = Uint16Array.from(tok.encode(text));
  const bytes = new Uint8Array(u16.buffer);
  if (line++ % 10 === 9) { valW.write(bytes); nVal += u16.length; } else { trainW.write(bytes); nTrain += u16.length; }
};
const drain = (final: boolean) => {
  let i: number;
  while ((i = buf.indexOf("\n")) >= 0) { emit(buf.slice(0, i + 1)); buf = buf.slice(i + 1); }
  if (final) { emit(buf); buf = ""; }
};

const stream = Bun.file(CORPUS).stream();
for await (const chunk of stream) { buf += dec.decode(chunk, { stream: true }); if (buf.length > 1 << 20) drain(false); }
buf += dec.decode(); drain(true);
await trainW.end(); await valW.end();

console.log(`encoded ${CORPUS} -> ${OUT}-train.bin (${nTrain} tok) + ${OUT}-val.bin (${nVal} tok) in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
