// End-to-end chat: message -> chat template -> tokenizer -> Qwen3-4bit -> tokens
// -> text. Turns the instruct model from a text-completer into a chat assistant.
//   bun chat.ts "What is the capital of France? Answer in one sentence."

import { Qwen3, generate } from "./qwen-nn.ts";
import { Tokenizer } from "./tokenizer.ts";
import { ChatTemplate } from "./chat-template.ts";
import { loadSafetensors } from "./loader.ts";

const cfg = await Bun.file("config-4bit.json").json();
const model = new Qwen3(cfg, loadSafetensors("model-q4.safetensors"));
const tok = await Tokenizer.fromFile("tokenizer.json");
const ct = await ChatTemplate.fromConfig("tokenizer_config-qwen.json");

const userMsg = process.argv[2] ?? "What is the capital of France? Answer in one sentence.";
const prompt = ct.render([{ role: "user", content: userMsg }]); // adds the assistant generation prompt
const ids = tok.encode(prompt);

const { gen, secs } = generate(model, ids, { max: 256, temp: 0, topP: 0, window: 0 }); // stops on <|im_end|> (eos)

console.log("user:      " + userMsg);
console.log("assistant: " + tok.decode(gen).trim());
console.log(`(${gen.length} tok in ${secs.toFixed(2)}s = ${(gen.length / secs).toFixed(0)} tok/s)`);
