// TS side of the Spark-TTS prompt/LM parity check, against mlx-audio.
//
// Greedy, so the ids can be compared one for one. Prints the prompt ids too:
// a wrong prompt still generates plausible-sounding speech, just not the text
// that was asked for, so the two have to be checked separately.
//   /tmp/sdvenv/bin/python reference/reference-spark.py && bun validation/spark-lm.ts
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { streamTokens } from "../src/text/lm.ts";
import { Tokenizer } from "../src/text/tokenizer.ts";
import { Qwen2, type Qwen2Config } from "../src/models/qwen2.ts";

const REPO = "mlx-community/Spark-TTS-0.5B-bf16";
const TEXT = "MLX runs on the GPU of your Mac.";

const tok = await Tokenizer.fromFile(await hubFile(REPO, "tokenizer.json"));
const prompt = tok.encode(
  "<|task_controllable_tts|><|start_content|>" + TEXT + "<|end_content|>" +
  "<|start_style_label|><|gender_0|><|pitch_label_2|><|speed_label_2|><|end_style_label|>",
);
console.log("prompt ids:", JSON.stringify(prompt));

const cfg = await readJson<Qwen2Config>(await hubFile(REPO, "config.json"));
const lm = new Qwen2(cfg, singleFileWeights(await hubFile(REPO, "model.safetensors")));

// The primary comparison. Greedy ids are tie-sensitive: the checkpoint is bf16,
// and adjacent audio tokens routinely land on the *same* bf16 logit, where the
// two runtimes break the tie differently and the sequences then re-converge.
// Logits have no such ambiguity.
const logits = lm.logitsLast(Int32Array.from(prompt), 1, prompt.length, 0,
                             Array(lm.numLayers).fill(null), 0).toF32();
const top = logits.map((v, i) => [i, v] as const).sort((a, b) => b[1] - a[1]).slice(0, 5);
console.log("top5:", JSON.stringify(top.map(([i, v]) => [i, +v.toFixed(4)])));

const out: number[] = [];
for await (const { token } of streamTokens(lm, prompt, {
  max: 16, temp: 0, repetitionPenalty: 1.3, repetitionContext: 20,
})) out.push(token);
console.log("gen ids:", JSON.stringify(out));
