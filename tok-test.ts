// Validate tokenizer.ts against the Python `tokenizers` fixtures.
//   python3 tok-reference.py && bun tok-test.ts

import { Tokenizer } from "./tokenizer.ts";

const fixtures = await Bun.file("tok-fixtures.json").json();
const tok = await Tokenizer.fromFile("tokenizer.json");

let pass = 0, fail = 0;
for (const f of fixtures) {
  const ids = tok.encode(f.text);
  const dec = tok.decode(ids);
  const idsOk = ids.length === f.ids.length && ids.every((x, i) => x === f.ids[i]);
  const decOk = dec === f.decoded;
  if (idsOk && decOk) { pass++; continue; }
  fail++;
  console.log(`FAIL ${JSON.stringify(f.text)}`);
  if (!idsOk) console.log(`  ids  got ${ids}\n       exp ${f.ids}`);
  if (!decOk) console.log(`  dec  got ${JSON.stringify(dec)}\n       exp ${JSON.stringify(f.decoded)}`);
}
console.log(`\nencode/decode parity vs Python tokenizers: ${pass}/${pass + fail} cases pass`);
if (fail) process.exit(1);
