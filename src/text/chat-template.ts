// HF chat-template engine: render a message list into the model's exact prompt
// string using the checkpoint's Jinja `chat_template`, via @huggingface/jinja
// (the same engine transformers.js uses). The other half of "outside MLX",
// alongside tokenizer.ts.
//
// The engine is loaded on demand, the way src/ffi/index.ts picks its backend:
// nothing else in src/ needs Jinja, so importing mlx-ts to run a diffusion model
// or a codec should not pull in a templating language. That is also why it sits
// in optionalDependencies — the library does not require it, only this file
// does, and only when a chat template is actually compiled.
import { readJson } from "../io/fs.ts";

export type Message = { role: "system" | "user" | "assistant"; content: string };

/** The slice of @huggingface/jinja's Template this needs. */
type Renderer = { render(context: Record<string, unknown>): string };

export class ChatTemplate {
  private tmpl: Renderer;

  private constructor(tmpl: Renderer) { this.tmpl = tmpl; }

  /**
   * Compile a Jinja chat template.
   *
   * Async only because the engine is imported on demand; `render` stays
   * synchronous, so a hot generation loop pays nothing.
   */
  static async fromString(chatTemplate: string): Promise<ChatTemplate> {
    let Template: new (src: string) => Renderer;
    try {
      ({ Template } = await import("@huggingface/jinja"));
    } catch {
      throw new Error(
        "mlx-ts: ChatTemplate needs @huggingface/jinja, which is an optional " +
          "dependency. Install it with `npm i @huggingface/jinja` (nothing else " +
          "in mlx-ts requires it).",
      );
    }
    return new ChatTemplate(new Template(chatTemplate));
  }

  /** Compile the `chat_template` out of a checkpoint's tokenizer_config.json. */
  static async fromConfig(path: string): Promise<ChatTemplate> {
    const cfg = await readJson<{ chat_template?: string }>(path);
    if (!cfg.chat_template) throw new Error(`no chat_template in ${path}`);
    return ChatTemplate.fromString(cfg.chat_template);
  }

  render(messages: Message[], opts: { addGenerationPrompt?: boolean; enableThinking?: boolean } = {}): string {
    return this.tmpl.render({
      messages,
      add_generation_prompt: opts.addGenerationPrompt ?? true,
      enable_thinking: opts.enableThinking ?? false,
    });
  }
}
