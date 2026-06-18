// Public streaming API demo — tokens printed as they're generated, no manual
// tidy()/free() anywhere. The KV cache is freed automatically when the loop ends
// (or on an early break / Ctrl-C).
//
//   bun stream.ts "Write a haiku about the sea"
//   bun stream.ts --temp 0.8 --topp 0.95 --topk 40 --reppenalty 1.1 --seed 42 "Tell me a story"

import { Qwen3 } from "./qwen-nn.ts";
import { streamText } from "./lm.ts";
import { Tokenizer } from "./tokenizer.ts";
import { ChatTemplate } from "./chat-template.ts";
import { loadSafetensors } from "./loader.ts";

const argv = process.argv.slice(2);
const flag = (n: string, d: number) => { const i = argv.indexOf(n); return i >= 0 ? Number(argv[i + 1]) : d; };
const temp = flag("--temp", 0), topP = flag("--topp", 0), sd = flag("--seed", 0);
const topK = flag("--topk", 0), repPenalty = flag("--reppenalty", 1);
const userMsg = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--"))).join(" ")
  || "Write a haiku about the sea";

const cfg = await Bun.file("config-4bit.json").json();
const model = new Qwen3(cfg, loadSafetensors("model-q4.safetensors"));
const tok = await Tokenizer.fromFile("tokenizer.json");
const ct = await ChatTemplate.fromConfig("tokenizer_config-qwen.json");
const ids = tok.encode(ct.render([{ role: "user", content: userMsg }]));

process.stdout.write(`user:      ${userMsg}\nassistant: `);
const t0 = performance.now();
let n = 0;
for await (const chunk of streamText(model, tok, ids, { max: 256, temp, topP, topK, repetitionPenalty: repPenalty, seed: sd })) {
  process.stdout.write(chunk);
  n++;
}
const secs = (performance.now() - t0) / 1000;
process.stdout.write(`\n(${n} chunks in ${secs.toFixed(2)}s)\n`);
