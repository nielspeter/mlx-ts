// The runtime-neutral FFI contract. One implementation per JS runtime
// (bun.ts / deno.ts / node.ts); index.ts picks one at import time.
//
// The load-bearing decision: **every pointer crosses this boundary as a JS
// number.** macOS user-space addresses fit in 2^48, well inside a double, so
// `type Arr = number` survives unchanged — which is why mx.ts, the ~250
// generated wrappers and every model file need no per-runtime code at all.
// Each backend absorbs its own representation (Bun numbers, Deno
// PointerObjects, koffi BigInts) behind these five primitives.

/** C types, spelled the way bun:ffi spells them — the lingua franca here. */
export type CType =
  | "ptr" | "cstring" | "void" | "bool"
  | "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "usize"
  | "f32" | "f64";

export type SymbolSpec = { args: CType[]; returns: CType };
export type SymbolTable = Record<string, SymbolSpec>;
export type Symbols = Record<string, (...args: any[]) => any>;

/** A C function pointer wrapping a JS function; `close()` releases it. */
export type Callback = { addr: number; close(): void };

export interface Backend {
  readonly name: "bun" | "deno" | "node";
  /** Human-readable runtime id, for diagnostics. */
  readonly version: string;
  /** dlopen + bind. Pointer args/returns are plain numbers; 0 is NULL. */
  open(lib: string, symbols: SymbolTable): Symbols;
  /** Address of a typed array's memory, as a number. */
  ptr(view: ArrayBufferView): number;
  /** Zero-copy ArrayBuffer over native memory. Never copies. */
  view(addr: number, byteOffset: number, byteLength: number): ArrayBuffer;
  /** Read a NUL-terminated C string at `addr`. */
  cstring(addr: number): string;
  /** Wrap a JS function as a C function pointer (mlx_closure_new_func). */
  callback(spec: SymbolSpec, fn: (...args: any[]) => any): Callback;
}
