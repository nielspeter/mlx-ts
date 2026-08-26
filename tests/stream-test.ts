// Validates the public stream() surface: greedy tokens from lm.streamTokens must
// be token-for-token identical to qwen-nn's proven generate(), and streamText
// must reconstruct exactly what tok.decode() produces over those ids.

import { loadSafetensors } from "../src/io/loader.ts";
import { generate as generateRef, Qwen3 } from "../src/models/qwen-nn.ts";
import { streamText, streamTokens } from "../src/text/lm.ts";
import { Tokenizer } from "../src/text/tokenizer.ts";

const cfg = await Bun.file("models/config-4bit.json").json();
const model = new Qwen3(cfg, loadSafetensors("models/model-q4.safetensors"));
const tok = await Tokenizer.fromFile("models/tokenizer.json");
const ids = tok.encode("The capital of France is");
const OPTS = { max: 48, temp: 0, topP: 0, window: 0 };

const ref = generateRef(model, ids, OPTS).gen;

const streamed: number[] = [];
for await (const { token } of streamTokens(model, ids, OPTS)) streamed.push(token);

let text = "";
for await (const chunk of streamText(model, tok, ids, OPTS)) text += chunk;

const idsMatch = JSON.stringify(ref) === JSON.stringify(streamed);
const textMatch = text === tok.decode(ref);

console.log(`stream ids == generate() ids: ${idsMatch}`);
console.log(`streamText == decode(ids):    ${textMatch}`);
console.log(`gen ids: [${streamed.join(", ")}]`);
console.log(idsMatch && textMatch ? "STREAM OK" : "STREAM MISMATCH");
