// Picks the FFI backend for the host runtime. The import is dynamic and its
// specifier computed, so Node never tries to resolve "bun:ffi" and Bun never
// pulls in koffi — only the selected backend is loaded.
import type { Backend } from "./types.ts";

const g = globalThis as any;
const runtime: "bun" | "deno" | "node" =
  typeof g.Bun !== "undefined" ? "bun"
  : typeof g.Deno !== "undefined" ? "deno"
  : typeof g.process !== "undefined" && g.process.versions?.node ? "node"
  : (() => { throw new Error("mlx-ts: unsupported runtime (need Bun, Deno, or Node)"); })();

export const backend: Backend = (await import(`./${runtime}.ts`)).backend;

// Bound up front so callers can destructure without losing `this`.
export const open = backend.open.bind(backend);
export const ptr = backend.ptr.bind(backend);
export const view = backend.view.bind(backend);
export const cstring = backend.cstring.bind(backend);
export const callback = backend.callback.bind(backend);
export type { Backend, CType, SymbolSpec, SymbolTable, Callback } from "./types.ts";
