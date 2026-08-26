// Verify our pure-TS BPE inference (tokenizer.ts) correctly consumes a FRESHLY
// TRAINED models/tokenizer.json (from tok-train.py / Rust BpeTrainer): token-exact encode
// vs the Rust tokenizer + clean decode round-trip.
//   VOCAB=2048 python3 tok-train.py && bun tok-train-test.ts
import { GPT2_SPLIT, Tokenizer } from "../src/text/tokenizer.ts";

const fx = await Bun.file("tests/tok-trained-fixtures.json").json();
const tok = await Tokenizer.fromFile("models/tokenizer-trained.json", GPT2_SPLIT);

let pass = 0, fail = 0;
for (const f of fx) {
  const ids = tok.encode(f.text);
  const idsOk = ids.length === f.ids.length && ids.every((x: number, i: number) => x === f.ids[i]);
  const decOk = tok.decode(ids) === f.decoded;
  if (idsOk && decOk) { pass++; continue; }
  fail++;
  console.log(`FAIL ${JSON.stringify(f.text)}`);
  if (!idsOk) console.log(`  ids got ${ids}\n      exp ${f.ids}`);
  if (!decOk) console.log(`  dec got ${JSON.stringify(tok.decode(ids))}`);
}
console.log(`TOKTRAIN: ${pass}/${pass + fail} (TS inference == freshly-trained Rust BPE)`);
if (fail) process.exit(1);
