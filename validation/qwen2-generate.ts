// TS side of the Qwen2 backbone parity check, against mlx-lm.
//
// Greedy decoding on Spark-TTS's own prompt format, compared as token ids:
// text would hide a tokenizer disagreement behind matching words, and the
// prompt ids are printed too so a tokenizer fault is distinguishable from a
// model fault.
//
// A plain sentence makes this checkpoint repeat one semantic token forever, and
// two implementations agreeing on a constant proves nothing — hence the real
// prompt.
//   /tmp/sdvenv/bin/python reference/reference-qwen2.py && bun validation/qwen2-generate.ts
import { generate } from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { Qwen2, type Qwen2Config } from "../src/models/qwen2.ts";
import { Tokenizer } from "../src/text/tokenizer.ts";

const REPO = "mlx-community/Spark-TTS-0.5B-bf16";
const cfg = await readJson<Qwen2Config>(await hubFile(REPO, "config.json"));
const tok = await Tokenizer.fromFile(await hubFile(REPO, "tokenizer.json"));
const model = new Qwen2(cfg, singleFileWeights(await hubFile(REPO, "model.safetensors")));

const prompt =
  "<|task_controllable_tts|><|start_content|>Hello there, this is a test." +
  "<|end_content|><|start_style_label|><|gender_1|><|pitch_label_2|>" +
  "<|speed_label_2|><|end_style_label|>";

const ids = tok.encode(prompt);
console.log(`prompt ids: [${ids.join(", ")}]`);

// Through the public generation path rather than a hand-rolled loop: the KV
// cache has to escape each step's tidy(), and doing that by hand here read a
// freed handle on the second token. streamTokens already gets it right.
const { gen } = await generate(model, ids, { max: 12, temp: 0 });

console.log(`gen ids: [${gen.join(", ")}]`);
// decode() returns "" here: these ids are all added tokens (>= 151k) and the
// byte-level BPE decoder only knows the base vocabulary. Not a problem for
// Spark, which reads the codec indices out of the ids numerically, but worth
// stating rather than leaving as a puzzling empty string.
console.log(`completion: ${JSON.stringify(tok.decode(gen))} (added tokens do not decode)`);
