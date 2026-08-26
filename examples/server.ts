// Local inference HTTP server — an OpenAI-compatible
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

import { unlink, writeFile } from "node:fs/promises";
import { decodeAudio, loadMelFilters } from "../src/audio/mel.ts";
import { fromI32, MX, tidy } from "../src/core/mx.ts";
import { readJson, readText } from "../src/io/fs.ts";
import { loadSafetensors } from "../src/io/loader.ts";
import { Qwen3 } from "../src/models/qwen-nn.ts";
import { loadWhisper, Whisper } from "../src/models/whisper.ts";
import { ChatTemplate, type Message } from "../src/text/chat-template.ts";
import { type GenOptions, streamTokens } from "../src/text/lm.ts";
import { Tokenizer } from "../src/text/tokenizer.ts";
import { WhisperTokenizer } from "../src/text/whisper-tokenizer.ts";
import { serve } from "./serve.ts";

const MODEL_ID = "qwen3-0.6b-4bit";
const PORT = Number(process.env.PORT ?? 8080);

console.log("loading model…");
const cfg = await readJson("models/config-4bit.json");
const model = new Qwen3(cfg, loadSafetensors("models/model-q4.safetensors"));
const tok = await Tokenizer.fromFile("models/tokenizer.json");
const ct = await ChatTemplate.fromConfig("models/tokenizer_config-qwen.json");
const CHAT_HTML = await readText(new URL("./chat.html", import.meta.url));
console.log("model ready");

// Whisper is optional: load it only if its assets are present, otherwise the
// /v1/audio/transcriptions route reports unavailable instead of failing startup.
const WHISPER_ID = "whisper-large-v3-turbo";
let whisper: { model: Whisper; tok: WhisperTokenizer; filtersT: MX } | null = null;
try {
  const [model, tok, filtersT] = await Promise.all([
    loadWhisper("models/config-turbo.json", "models/whisper-turbo.safetensors"),
    WhisperTokenizer.fromFile(),
    loadMelFilters("models/whisper-mel-filters-128.f32", 128),
  ]);
  whisper = { model, tok, filtersT };
  console.log("whisper ready (/v1/audio/transcriptions)");
} catch {
  console.log("whisper assets missing — /v1/audio/transcriptions disabled");
}

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
// Roles arrive from an untrusted HTTP body. `Msg` used to type them as plain
// `string`, so anything at all reached the Jinja chat template; validate here so
// the type is true and a bad role is a 400 rather than a surprise in the prompt.
const ROLES = new Set(["system", "user", "assistant"]);
function asMessages(v: unknown): Message[] | null {
  if (!Array.isArray(v) || v.length === 0) return null;
  const out: Message[] = [];
  for (const msg of v) {
    if (!msg || typeof msg.content !== "string" || !ROLES.has(msg.role)) return null;
    out.push({ role: msg.role as Message["role"], content: msg.content });
  }
  return out;
}
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
async function* run(messages: Message[], opts: GenOptions, stops: string[], cancelled: () => boolean) {
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

// OpenAI-compatible speech-to-text: multipart/form-data with a `file` audio part
// -> {text} (or plain text for response_format=text). English transcribe only.
async function transcriptions(req: Request): Promise<Response> {
  if (!whisper) return json({ error: "audio transcription unavailable (whisper assets not loaded)" }, 501);
  let form: FormData;
  try { form = await req.formData(); } catch { return json({ error: "expected multipart/form-data" }, 400); }
  const file = form.get("file");
  if (!(file instanceof Blob)) return json({ error: "missing audio `file` field" }, 400);
  const fmt = String(form.get("response_format") ?? "json");
  const tmp = `/tmp/mlxts-upload-${Math.random().toString(36).slice(2)}`;
  await writeFile(tmp, new Uint8Array(await file.arrayBuffer()));
  const release = await acquire();
  try {
    const pcm = await decodeAudio(tmp);                       // ffmpeg decode (no MLX) then transcribe
    const text = whisper!.tok.decode(whisper!.model.transcribe(pcm, whisper!.filtersT)).trim();
    return fmt === "text"
      ? new Response(text, { headers: { "content-type": "text/plain; charset=utf-8", ...CORS } })
      : json({ text });
  } catch (e) {
    return json({ error: String(e) }, 500);
  } finally {
    release();
    await unlink(tmp).catch(() => {});
  }
}

async function chat(req: Request): Promise<Response> {
  let body: any;
  try { body = await req.json(); } catch { return json({ error: "invalid JSON body" }, 400); }
  const messages = asMessages(body.messages);
  if (!messages) return json({ error: "messages[] required: {role: system|user|assistant, content: string}" }, 400);
  const opts = toOptions(body);
  const stops: string[] = body.stop ? (Array.isArray(body.stop) ? body.stop : [body.stop]) : [];
  const id = rid(), created = now();
  // reply in the user's language unless the caller set its own system message
  const sys: Message = { role: "system", content: "You are a helpful assistant. Always reply in the same language as the user's message." };
  const msgs = messages.some((m) => m.role === "system") ? messages : [sys, ...messages];

  if (!body.stream) {
    const release = await acquire();
    try {
      let text = "", usage = { prompt_tokens: 0, completion_tokens: 0 };
      for await (const p of run(msgs, opts, stops, () => false)) {
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
        for await (const p of run(msgs, opts, stops, () => cancelled)) {
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

await serve(PORT, async (req) => {
  const { pathname } = new URL(req.url);
  if (req.method === "OPTIONS") return new Response(null, { headers: CORS });
  if (pathname === "/" || pathname === "/index.html") return new Response(CHAT_HTML, { headers: { "content-type": "text/html; charset=utf-8" } });
  if (pathname === "/favicon.ico") return new Response(null, { status: 204 });
  if (pathname === "/health") return json({ status: "ok", model: MODEL_ID });
  if (pathname === "/v1/models") return json({ object: "list", data: [{ id: MODEL_ID, object: "model", created: now(), owned_by: "mlx-ts" }, ...(whisper ? [{ id: WHISPER_ID, object: "model", created: now(), owned_by: "mlx-ts" }] : [])] });
  if (pathname === "/v1/chat/completions" && req.method === "POST") return chat(req);
  if (pathname === "/v1/embeddings" && req.method === "POST") return embeddings(req);
  if (pathname === "/v1/audio/transcriptions" && req.method === "POST") return transcriptions(req);
  return json({ error: "not found", routes: ["/ (chat UI)", "/health", "/v1/models", "POST /v1/chat/completions", "POST /v1/embeddings", "POST /v1/audio/transcriptions"] }, 404);
});
console.log(`listening on http://localhost:${PORT}  (chat UI at / , API at POST /v1/chat/completions)`);
