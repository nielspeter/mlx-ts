// Validate the TS GPT-2 BPE encoder (tokenizer.ts + GPT2_SPLIT) against HF
// `tokenizers` fixtures (reference-gpt2-tok.py).
//   python3 reference-gpt2-tok.py && bun gpt2-tok-test.ts
import { Tokenizer, GPT2_SPLIT } from "./tokenizer.ts";

const fixtures = await Bun.file("gpt2-tok-fixtures.json").json();
const tok = await Tokenizer.fromFile("gpt2-tokenizer.json", GPT2_SPLIT);

let pass = 0, fail = 0;
for (const f of fixtures) {
  const ids = tok.encode(f.text);
  const idsOk = ids.length === f.ids.length && ids.every((x, i) => x === f.ids[i]);
  const decOk = tok.decode(ids) === f.decoded;
  if (idsOk && decOk) { pass++; continue; }
  fail++;
  console.log(`FAIL ${JSON.stringify(f.text)}`);
  if (!idsOk) console.log(`  ids got ${ids}\n      exp ${f.ids}`);
}
console.log(`GPT2 TOK: ${pass}/${pass + fail} cases pass`);
if (fail) process.exit(1);
