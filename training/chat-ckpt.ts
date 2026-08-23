// chat_cli (nanochat stage): chat with the SFT'd checkpoint from chat-sft.ts.
//   bun chat-ckpt.ts "What is the capital of France?"
import { Tokenizer, GPT2_SPLIT } from "../src/text/tokenizer.ts";
import { loadCkpt, generate } from "../src/models/nanogpt-model.ts";

const CKPT = process.env.CHAT_CKPT ?? "checkpoints/chat-ckpt.safetensors";
const tok = await Tokenizer.fromFile("models/tokenizer-trained.json", GPT2_SPLIT);
const EOS = tok.encode("<|endoftext|>")[0];
const { params, cfg } = await loadCkpt(CKPT);

const q = process.argv[2] ?? "What is the capital of France?";
const promptIds = tok.encode(`User: ${q}\nAssistant:`);
const out = generate(params, promptIds, cfg, EOS, { maxNew: 48, temp: +(process.env.TEMP ?? 0) });
console.log(`Q: ${q}`);
console.log(`A: ${tok.decode(out).trim()}`);
