// The Spark-TTS prompt and LM against the committed reference numbers.
//
// Run at float32, not the checkpoint's bf16, which makes this an exact check
// rather than a ranking one. bf16 has eight mantissa bits, and this model
// carries outlier channels in the thousands that cancel in the last layer
// (+3700 -> -800 at the final position), so one ulp at layer 9 becomes a percent
// at the logits. In bf16 no two implementations agree for long — PyTorch's own
// bf16 leaves its float32 after 5 greedy tokens. At float32 they agree exactly.
//
// The prompt is checked separately from the model: a wrong prompt still
// generates fluent speech, just not of the text that was asked for.
//   bun validation/spark-lm.ts
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights, upcastWeights } from "../src/io/loader.ts";
import { Qwen2, type Qwen2Config } from "../src/models/qwen2.ts";
import { streamTokens } from "../src/text/lm.ts";
import { Tokenizer } from "../src/text/tokenizer.ts";
import { checkIds, loadGolden, verdict } from "./golden.ts";

const REPO = "mlx-community/Spark-TTS-0.5B-bf16";
const g = loadGolden("spark-golden.json").lm;

const tok = await Tokenizer.fromFile(await hubFile(REPO, "tokenizer.json"));
const prompt = tok.encode(
  "<|task_controllable_tts|><|start_content|>" + g.text + "<|end_content|>" +
  "<|start_style_label|><|gender_0|><|pitch_label_2|><|speed_label_2|><|end_style_label|>",
);
checkIds("prompt", prompt, g.prompt_ids);

const cfg = await readJson<Qwen2Config>(await hubFile(REPO, "config.json"));
const lm = new Qwen2(cfg, upcastWeights(singleFileWeights(await hubFile(REPO, "model.safetensors"))));

const logits = lm.logitsLast(Int32Array.from(prompt), 1, prompt.length, 0,
                             Array(lm.numLayers).fill(null), 0).toF32();
const top = logits.map((v, i) => [i, v] as const).sort((a, b) => b[1] - a[1]).slice(0, 5);
checkIds("top5", top.map(([i]) => i), g.top5.map(([i]: [number, number]) => i));
for (const [n, [id, want]] of (g.top5 as [number, number][]).entries()) {
  const got = logits[id];
  if (Math.abs(got - want) > 1e-2) {
    console.log(`  FAIL top5[${n}]   logit ${got.toFixed(4)} vs ${want.toFixed(4)}`);
    process.exitCode = 1;
  }
}
console.log(`  ok   logits    top-5 within 1e-2 of PyTorch float32`);

const out: number[] = [];
for await (const { token } of streamTokens(lm, prompt, { max: g.greedy_ids.length, temp: 0 })) out.push(token);
checkIds("greedy", out, g.greedy_ids);

verdict("Spark-TTS LM");
