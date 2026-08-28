// ChatTemplate: the Jinja chat-template renderer.
//
// This was a standalone script, which meant `bun test --coverage` never loaded
// the file and reported it as 0% — exercised but uncounted. The parity cases it
// carried need models/tokenizer_config-qwen.json, which is gitignored and absent
// from a fresh clone, so they still skip; everything else here uses inline
// templates and runs everywhere.
//   bun test tests/chat-template.test.ts
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "bun:test";
import { ChatTemplate, type Message } from "../src/index.ts";

// Small, but exercises what a real template uses: a loop over messages and the
// two flags render() injects.
const TMPL =
  "{%- for m in messages %}{{ '<' + m.role + '>' + m.content }}{%- endfor %}" +
  "{%- if add_generation_prompt %}<assistant>{%- endif %}" +
  "{%- if enable_thinking %}<think>{%- endif %}";

const USER: Message[] = [{ role: "user", content: "hi" }];

test("renders messages through the template", async () => {
  const ct = await ChatTemplate.fromString(TMPL);
  expect(ct.render([{ role: "system", content: "be brief" }, ...USER], { addGenerationPrompt: false }))
    .toBe("<system>be brief<user>hi");
});

test("add_generation_prompt defaults on, enable_thinking defaults off", async () => {
  // The defaults are the contract: a caller that passes nothing should get a
  // prompt the model can continue from, and no thinking block.
  const ct = await ChatTemplate.fromString(TMPL);
  expect(ct.render(USER)).toBe("<user>hi<assistant>");
  expect(ct.render(USER, { addGenerationPrompt: false })).toBe("<user>hi");
  expect(ct.render(USER, { enableThinking: true })).toBe("<user>hi<assistant><think>");
});

test("render is synchronous — a generation loop pays nothing for the lazy import", async () => {
  // Only construction is async. If render ever returned a promise, every caller
  // in a token loop would silently start rendering "[object Promise]".
  const ct = await ChatTemplate.fromString(TMPL);
  const out = ct.render(USER);
  expect(typeof out).toBe("string");
  expect(out).not.toContain("Promise");
});

test("fromConfig reads chat_template out of a tokenizer config", async () => {
  const dir = mkdtempSync(join(tmpdir(), "mlxts-chat-"));
  const p = join(dir, "tokenizer_config.json");
  writeFileSync(p, JSON.stringify({ chat_template: TMPL, model_max_length: 32768 }));
  const ct = await ChatTemplate.fromConfig(p);
  expect(ct.render(USER, { addGenerationPrompt: false })).toBe("<user>hi");
});

test("fromConfig names the file when there is no chat_template", async () => {
  // A base (non-chat) checkpoint has no template. Failing with the path beats a
  // `Cannot read properties of undefined` from inside the Jinja parser.
  const dir = mkdtempSync(join(tmpdir(), "mlxts-chat-"));
  const p = join(dir, "tokenizer_config.json");
  writeFileSync(p, JSON.stringify({ model_max_length: 32768 }));
  await expect(ChatTemplate.fromConfig(p)).rejects.toThrow(/no chat_template/);
});

// --- parity against Python jinja2, when the Qwen config is present -----------
// Regenerate with: python3 reference/reference-chat.py
const QWEN = "models/tokenizer_config-qwen.json";
const FIXTURES = "tests/chat-fixtures.json";
const HAVE = existsSync(QWEN) && existsSync(FIXTURES);

test.skipIf(!HAVE)("renders Qwen3's real template exactly as Python jinja2 does", async () => {
  const fixtures = await Bun.file(FIXTURES).json();
  const ct = await ChatTemplate.fromConfig(QWEN);
  expect(fixtures.length).toBeGreaterThan(0);
  for (const f of fixtures) {
    expect(ct.render(f.input.messages, {
      addGenerationPrompt: f.input.add_generation_prompt,
      enableThinking: f.input.enable_thinking,
    })).toBe(f.rendered);
  }
  console.log(`  chat-template parity vs Python jinja2: ${fixtures.length}/${fixtures.length} cases pass`);
});
