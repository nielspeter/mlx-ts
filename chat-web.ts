// chat_web (nanochat stage): serve OUR SFT'd checkpoint (chat-sft.ts) behind the
// same OpenAI-compatible streaming endpoint + chat.html UI that server.ts uses.
// The model is our tiny nanogpt-model checkpoint instead of Qwen3.
//   bun chat-web.ts                 # listens on :8080 (PORT to override)
//   open http://localhost:8080
import { MX, fromI32, tidy, sample } from "./mx.ts";
import { Tokenizer, GPT2_SPLIT } from "./tokenizer.ts";
import { loadCkpt, forward } from "./nanogpt-model.ts";

const CKPT = process.env.CHAT_CKPT ?? "chat-ckpt.safetensors";
const PORT = Number(process.env.PORT ?? 8080);
const MODEL_ID = "mlx-ts-nanochat";

console.log(`loading ${CKPT}…`);
const tok = await Tokenizer.fromFile("tokenizer-trained.json", GPT2_SPLIT);
const EOS = tok.encode("<|endoftext|>")[0];
const { params, cfg } = await loadCkpt(CKPT);
const CHAT_HTML = await Bun.file(new URL("./chat.html", import.meta.url)).text();
console.log("model ready");

// async mutex — one generation at a time (shared model + global tidy arena)
let tail: Promise<void> = Promise.resolve();
function acquire(): Promise<() => void> {
  const prev = tail; let release!: () => void;
  tail = new Promise<void>((r) => (release = r));
  return prev.then(() => { let done = false; return () => { if (!done) { done = true; release(); } }; });
}

type Msg = { role: string; content: string };
// our SFT chat format: "User: ..\nAssistant: ..\n" turns, ending at "Assistant:"
const renderPrompt = (messages: Msg[]) =>
  messages.filter((m) => m.role !== "system").map((m) => (m.role === "user" ? `User: ${m.content}` : `Assistant: ${m.content}`)).join("\n") + "\nAssistant:";

// streaming generation: one token at a time, incremental UTF-8 decode, stop at EOS
// or when the model starts a new "User:" turn.
async function* run(messages: Msg[], temp: number, maxNew: number, cancelled: () => boolean) {
  let ids = tok.encode(renderPrompt(messages));
  const det = tok.detokenizer();
  let full = "", n = 0;
  for (let i = 0; i < maxNew; i++) {
    if (cancelled()) break;
    const ctx = ids.slice(-cfg.block_size), L = ctx.length;
    const tk = tidy(() => {
      const logits = forward(params, fromI32(Int32Array.from(ctx), [1, L]), cfg).reshape([L, cfg.vocab]);
      return sample(logits.slice([L - 1, 0], [L, cfg.vocab]), temp, 0, 0).itemU();
    });
    if (tk === EOS) break;
    ids.push(tk); n++;
    const piece = det.add(tk);
    if (!piece) continue;
    full += piece;
    const cut = full.indexOf("\nUser");                       // weak model may try to continue the convo
    if (cut >= 0) { const keep = piece.length - (full.length - cut); if (keep > 0) yield { text: piece.slice(0, keep) }; return; }
    yield { text: piece };
  }
  const t = det.flush(); if (t) yield { text: t };
}

const CORS = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "*" };
const json = (o: any, status = 200) => new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json", ...CORS } });
const now = () => Math.floor(Date.now() / 1000);
const rid = () => "chatcmpl-" + Math.random().toString(36).slice(2);

async function chat(req: Request): Promise<Response> {
  let body: any;
  try { body = await req.json(); } catch { return json({ error: "invalid JSON body" }, 400); }
  const messages: Msg[] = body.messages;
  if (!Array.isArray(messages) || messages.length === 0) return json({ error: "messages[] required" }, 400);
  const temp = body.temperature ?? 0, maxNew = body.max_tokens ?? 64;
  const id = rid(), created = now();

  if (!body.stream) {
    const release = await acquire();
    try {
      let text = "";
      for await (const p of run(messages, temp, maxNew, () => false)) text += p.text;
      return json({ id, object: "chat.completion", created, model: MODEL_ID, choices: [{ index: 0, message: { role: "assistant", content: text.trim() }, finish_reason: "stop" }] });
    } finally { release(); }
  }

  let cancelled = false;
  const enc = new TextEncoder();
  const chunk = (delta: any, finish: string | null = null) =>
    enc.encode(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: MODEL_ID, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`);
  const stream = new ReadableStream({
    async start(c) {
      const release = await acquire();
      try {
        c.enqueue(chunk({ role: "assistant" }));
        for await (const p of run(messages, temp, maxNew, () => cancelled)) { if (cancelled) break; if (p.text) c.enqueue(chunk({ content: p.text })); }
        c.enqueue(chunk({}, "stop")); c.enqueue(enc.encode("data: [DONE]\n\n")); c.close();
      } catch (e) { c.error(e); } finally { release(); }
    },
    cancel() { cancelled = true; },
  });
  return new Response(stream, { headers: { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive", ...CORS } });
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    const { pathname } = new URL(req.url);
    if (req.method === "OPTIONS") return new Response(null, { headers: CORS });
    if (pathname === "/" || pathname === "/index.html") return new Response(CHAT_HTML, { headers: { "content-type": "text/html; charset=utf-8" } });
    if (pathname === "/favicon.ico") return new Response(null, { status: 204 });
    if (pathname === "/health") return json({ status: "ok", model: MODEL_ID });
    if (pathname === "/v1/models") return json({ object: "list", data: [{ id: MODEL_ID, object: "model", created: now(), owned_by: "mlx-ts" }] });
    if (pathname === "/v1/chat/completions" && req.method === "POST") return chat(req);
    return json({ error: "not found", routes: ["/ (chat UI)", "/health", "POST /v1/chat/completions"] }, 404);
  },
});
console.log(`listening on http://localhost:${PORT}  (chat UI at / , API at POST /v1/chat/completions)`);
