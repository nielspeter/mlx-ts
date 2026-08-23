// Validate chat-template.ts against the Python jinja2 fixtures.
//   python3 reference-chat.py && bun chat-test.ts

import { ChatTemplate } from "../src/text/chat-template.ts";

const fixtures = await Bun.file("chat-fixtures.json").json();
const ct = await ChatTemplate.fromConfig("tokenizer_config-qwen.json");

let pass = 0, fail = 0;
for (const f of fixtures) {
  const got = ct.render(f.input.messages, {
    addGenerationPrompt: f.input.add_generation_prompt,
    enableThinking: f.input.enable_thinking,
  });
  if (got === f.rendered) { pass++; continue; }
  fail++;
  console.log(`FAIL: ${JSON.stringify(f.input.messages)}`);
  console.log(`  got: ${JSON.stringify(got)}`);
  console.log(`  exp: ${JSON.stringify(f.rendered)}`);
}
console.log(`\nchat-template parity vs Python jinja2: ${pass}/${pass + fail} cases pass`);
if (fail) process.exit(1);
