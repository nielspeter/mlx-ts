// Portable HTTP serving for the examples: one `serve(port, handler)` over the
// Web Request/Response API, which all three runtimes speak — they just disagree
// about how you start a server.
//
//   Bun   Bun.serve({ port, fetch })
//   Deno  Deno.serve({ port }, handler)
//   Node  node:http, with a small conversion at each edge
//
// SSE matters here (server.ts streams tokens), so the Node path pumps the
// Response body chunk by chunk instead of buffering it.
import type { Server } from "node:http";

export type Handler = (req: Request) => Response | Promise<Response>;

export async function serve(port: number, handler: Handler): Promise<void> {
  const g = globalThis as any;

  if (typeof g.Bun !== "undefined") { g.Bun.serve({ port, fetch: handler }); return; }
  if (typeof g.Deno !== "undefined") { g.Deno.serve({ port }, handler); return; }

  const { createServer } = await import("node:http");
  const server: Server = createServer(async (nreq, nres) => {
    try {
      // node:http gives a path, not a URL, and a stream, not a body.
      const url = `http://${nreq.headers.host ?? `localhost:${port}`}${nreq.url}`;
      const method = nreq.method ?? "GET";
      let body: Buffer | undefined;
      if (method !== "GET" && method !== "HEAD") {
        const chunks: Buffer[] = [];
        for await (const c of nreq) chunks.push(c as Buffer);
        body = Buffer.concat(chunks);
      }
      const res = await handler(new Request(url, {
        method,
        headers: nreq.headers as any,
        body: body?.length ? new Uint8Array(body) : undefined,
      }));

      nres.writeHead(res.status, Object.fromEntries(res.headers));
      if (!res.body) { nres.end(); return; }
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        nres.write(value);                 // unbuffered, so SSE arrives token by token
      }
      nres.end();
    } catch (err) {
      nres.writeHead(500, { "content-type": "text/plain" });
      nres.end(String(err));
    }
  });
  server.listen(port);
}
