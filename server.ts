// Local inference HTTP server over Bun.serve — an OpenAI-compatible
// /v1/chat/completions endpoint (streaming SSE or single JSON) backed by the
// public lm.ts streaming API and 4-bit Qwen3.
//
//   bun server.ts                 # listens on :8080 (PORT to override)
//   curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'
//
// Concurrency: generation is SERIALIZED behind an async mutex. mlx-ts shares one
// model and a module-global tidy() arena, and streamTokens yields at await points
// between tokens — so two interleaved generations would corrupt each other. One
// at a time; extra requests queue. (A real multi-tenant server needs batched
// decode + per-request arenas — see README "not yet".)

import { Qwen3 } from "./qwen-nn.ts";
import { fromI32, tidy } from "./mx.ts";
import { streamTokens, type GenOptions } from "./lm.ts";
import { Tokenizer } from "./tokenizer.ts";
import { ChatTemplate } from "./chat-template.ts";
import { loadSafetensors } from "./loader.ts";

const MODEL_ID = "qwen3-0.6b-4bit";
const PORT = Number(process.env.PORT ?? 8080);

console.log("loading model…");
const cfg = await Bun.file("config-4bit.json").json();
const model = new Qwen3(cfg, loadSafetensors("model-q4.safetensors"));
const tok = await Tokenizer.fromFile("tokenizer.json");
const ct = await ChatTemplate.fromConfig("tokenizer_config-qwen.json");
const CHAT_HTML = await Bun.file(new URL("./chat.html", import.meta.url)).text();
console.log("model ready");

// ---- async mutex: one generation at a time ----
let tail: Promise<void> = Promise.resolve();
function acquire(): Promise<() => void> {
  const prev = tail;
  let release!: () => void;
  tail = new Promise<void>((r) => (release = r));
  return prev.then(() => {
    let done = false; // idempotent: safe to call from both normal finish and cancel()
    return () => { if (!done) { done = true; release(); } };
  });
}

// ---- request -> GenOptions (OpenAI-ish field names) ----
type Msg = { role: string; content: string };
function toOptions(b: any): GenOptions {
  return {
    max: b.max_tokens ?? 512,
    temp: b.temperature ?? 0,
    topP: b.top_p ?? 0,
    topK: b.top_k ?? 0,
    repetitionPenalty: b.repetition_penalty ?? 1,
    seed: b.seed,
  };
}

// Generate text incrementally; yields decoded pieces, honoring `stop` strings.
// Returns token counts via the final `done` record.
async function* run(messages: Msg[], opts: GenOptions, stops: string[], cancelled: () => boolean) {
  const ids = tok.encode(ct.render(messages));
  const det = tok.detokenizer();
  let full = "", n = 0;
  for await (const { token } of streamTokens(model, ids, opts)) {
    if (cancelled()) break;
    n++;
    const piece = det.add(token);
    if (!piece) continue;
    full += piece;
    const hit = stops.map((s) => full.indexOf(s)).filter((i) => i >= 0).sort((a, b) => a - b)[0];
    if (hit !== undefined) { // emit up to the stop, then end
      const remain = full.length - piece.length;
      if (hit > remain) yield { text: piece.slice(0, hit - remain), done: false };
      yield { text: "", done: true, prompt: ids.length, completion: n, reason: "stop" };
      return;
    }
    yield { text: piece, done: false };
  }
  const tailText = det.flush();
  if (tailText) yield { text: tailText, done: false };
  yield { text: "", done: true, prompt: ids.length, completion: n, reason: "stop" };
}

const json = (o: any, status = 200) =>
  new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json", ...CORS } });
const CORS = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "*" };
const rid = () => "chatcmpl-" + Math.random().toString(36).slice(2);
const now = () => Math.floor(Date.now() / 1000);

// OpenAI-compatible embeddings: input string | string[] -> L2-normalized vectors.
// Each text is embedded separately (B=1) so ragged lengths need no padding.
async function embeddings(req: Request): Promise<Response> {
  let body: any;
  try { body = await req.json(); } catch { return json({ error: "invalid JSON body" }, 400); }
  const inputs: string[] = Array.isArray(body.input) ? body.input : [body.input];
  if (inputs.length === 0 || inputs.some((x) => typeof x !== "string")) return json({ error: "input (string | string[]) required" }, 400);

  const release = await acquire();
  try {
    let total = 0;
    const data = inputs.map((text, index) => {
      const ids = tok.encode(text);
      total += ids.length;
      const e = tidy(() => model.embeddingMX(fromI32(Int32Array.from(ids), [1, ids.length]), 1, ids.length));
      const embedding = Array.from(e.toF32());
      e.free();
      return { object: "embedding", index, embedding };
    });
    return json({ object: "list", data, model: MODEL_ID, usage: { prompt_tokens: total, total_tokens: total } });
  } finally { release(); }
}

async function chat(req: Request): Promise<Response> {
  let body: any;
  try { body = await req.json(); } catch { return json({ error: "invalid JSON body" }, 400); }
  const messages: Msg[] = body.messages;
  if (!Array.isArray(messages) || messages.length === 0) return json({ error: "messages[] required" }, 400);
  const opts = toOptions(body);
  const stops: string[] = body.stop ? (Array.isArray(body.stop) ? body.stop : [body.stop]) : [];
  const id = rid(), created = now();

  if (!body.stream) {
    const release = await acquire();
    try {
      let text = "", usage = { prompt_tokens: 0, completion_tokens: 0 };
      for await (const p of run(messages, opts, stops, () => false)) {
        if (p.done) usage = { prompt_tokens: p.prompt!, completion_tokens: p.completion! };
        else text += p.text;
      }
      return json({
        id, object: "chat.completion", created, model: MODEL_ID,
        choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
        usage: { ...usage, total_tokens: usage.prompt_tokens + usage.completion_tokens },
      });
    } finally { release(); }
  }

  // streaming SSE
  let cancelled = false;
  const enc = new TextEncoder();
  const chunk = (delta: any, finish: string | null = null) =>
    enc.encode(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: MODEL_ID, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`);
  const stream = new ReadableStream({
    async start(c) {
      const release = await acquire();
      try {
        c.enqueue(chunk({ role: "assistant" }));
        for await (const p of run(messages, opts, stops, () => cancelled)) {
          if (cancelled) break;
          if (!p.done && p.text) c.enqueue(chunk({ content: p.text }));
        }
        c.enqueue(chunk({}, "stop"));
        c.enqueue(enc.encode("data: [DONE]\n\n"));
        c.close();
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
    if (pathname === "/v1/embeddings" && req.method === "POST") return embeddings(req);
    return json({ error: "not found", routes: ["/ (chat UI)", "/health", "/v1/models", "POST /v1/chat/completions", "POST /v1/embeddings"] }, 404);
  },
});
console.log(`listening on http://localhost:${PORT}  (chat UI at / , API at POST /v1/chat/completions)`);
