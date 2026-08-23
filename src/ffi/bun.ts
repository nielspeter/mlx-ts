// Bun backend — the reference implementation. bun:ffi already speaks the CType
// vocabulary this contract borrowed, so this is close to a pass-through. The
// one wrinkle is NULL: bun:ffi returns a null pointer as `null`, not 0.
import { dlopen, ptr as bunPtr, toArrayBuffer, CString, JSCallback } from "bun:ffi";
import type { Backend, Callback, SymbolSpec, SymbolTable, Symbols } from "./types.ts";

export const backend: Backend = {
  name: "bun",
  version: `bun ${Bun.version}`,

  open(lib: string, symbols: SymbolTable): Symbols {
    const { symbols: s } = dlopen(lib, symbols as any);
    // Normalize NULL -> 0 so callers never see `null` for a pointer return.
    const out: Symbols = Object.create(null);
    for (const [name, spec] of Object.entries(symbols)) {
      const fn = (s as any)[name];
      if (!fn) continue;
      out[name] = spec.returns === "ptr" ? (...a: any[]) => (fn(...a) ?? 0) as number : fn;
    }
    return out;
  },

  // bun:ffi types ptr() more narrowly than our ArrayBufferView contract.
  ptr: (v: ArrayBufferView) => bunPtr(v as unknown as NodeJS.TypedArray),
  view: (addr: number, off: number, len: number) => toArrayBuffer(addr, off, len),
  cstring: (addr: number) => new CString(addr).toString(),

  callback(spec: SymbolSpec, fn: (...args: any[]) => any): Callback {
    const cb = new JSCallback(fn, spec as any);
    return { addr: Number(cb.ptr), close: () => cb.close() };
  },
};
