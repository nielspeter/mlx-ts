// HF chat-template engine: render a message list into the model's exact prompt
// string using the checkpoint's Jinja `chat_template`, via @huggingface/jinja
// (the same engine transformers.js uses). The other half of "outside MLX",
// alongside tokenizer.ts.

import { Template } from "@huggingface/jinja";

export type Message = { role: "system" | "user" | "assistant"; content: string };

export class ChatTemplate {
  private tmpl: Template;
  constructor(chatTemplate: string) { this.tmpl = new Template(chatTemplate); }

  static async fromConfig(path: string): Promise<ChatTemplate> {
    const cfg = await Bun.file(path).json();
    return new ChatTemplate(cfg.chat_template);
  }

  render(messages: Message[], opts: { addGenerationPrompt?: boolean; enableThinking?: boolean } = {}): string {
    return this.tmpl.render({
      messages,
      add_generation_prompt: opts.addGenerationPrompt ?? true,
      enable_thinking: opts.enableThinking ?? false,
    });
  }
}
