// Deno backend. Deno's built-in FFI is the closest match to bun:ffi, but it
// hands pointers back as opaque PointerObjects, so every "ptr" arg and return
// needs an address round-trip. Two things worth knowing:
//
//  - u64/usize cross as BigInt (callers already pass BigInt for size_t args).
//  - a 64-bit *return* falls off V8's fast-call path: ~52 ns/call vs ~2.6 ns
//    for a 32-bit one (spike-ffi-deno.ts). Declare hot accessors 32-bit.
import type { Backend, Callback, CType, SymbolSpec, Symbols, SymbolTable } from "./types.ts";

const D = (globalThis as any).Deno;

const TYPE: Record<CType, string> = {
  ptr: "pointer", cstring: "pointer", void: "void", bool: "bool",
  i8: "i8", u8: "u8", i16: "i16", u16: "u16", i32: "i32", u32: "u32",
  i64: "i64", u64: "u64", usize: "usize", f32: "f32", f64: "f64",
};

const toAddr = (p: unknown): number => (p == null ? 0 : Number(D.UnsafePointer.value(p)));
const toPtr = (n: unknown): unknown =>
  typeof n === "number" ? (n === 0 ? null : D.UnsafePointer.create(BigInt(n))) : n;

export const backend: Backend = {
  name: "deno",
  version: `deno ${D.version.deno}`,

  open(lib: string, symbols: SymbolTable): Symbols {
    const defs: Record<string, unknown> = {};
    for (const [name, spec] of Object.entries(symbols)) {
      defs[name] = { parameters: spec.args.map((a) => TYPE[a]), result: TYPE[spec.returns] };
    }
    const { symbols: s } = D.dlopen(lib, defs);

    const out: Symbols = Object.create(null);
    for (const [name, spec] of Object.entries(symbols)) {
      const raw = (s as any)[name];
      if (!raw) continue;
      const argIsPtr = spec.args.map((a) => a === "ptr" || a === "cstring");
      const anyPtrArg = argIsPtr.some(Boolean);
      const ptrRet = spec.returns === "ptr";
      const strRet = spec.returns === "cstring";   // contract: returns a JS string
      out[name] = !anyPtrArg && !ptrRet && !strRet
        ? raw
        : (...a: any[]) => {
            if (anyPtrArg) for (let i = 0; i < a.length; i++) if (argIsPtr[i]) a[i] = toPtr(a[i]);
            const r = raw(...a);
            if (strRet) return r == null ? "" : new D.UnsafePointerView(r).getCString();
            return ptrRet ? toAddr(r) : r;
          };
    }
    return out;
  },

  // Deno's "pointer" params also accept a TypedArray directly, but the contract
  // is address-in/address-out, so go through UnsafePointer like everything else.
  ptr: (v: ArrayBufferView) => toAddr(D.UnsafePointer.of(v)),
  view: (addr: number, off: number, len: number) =>
    new D.UnsafePointerView(D.UnsafePointer.create(BigInt(addr))).getArrayBuffer(len, off),
  cstring: (addr: number) => new D.UnsafePointerView(D.UnsafePointer.create(BigInt(addr))).getCString(),

  callback(spec: SymbolSpec, fn: (...args: any[]) => any): Callback {
    const def = { parameters: spec.args.map((a) => TYPE[a]), result: TYPE[spec.returns] };
    // Args arrive as PointerObjects; the JS side speaks numbers.
    const argIsPtr = spec.args.map((a) => a === "ptr" || a === "cstring");
    const cb = new D.UnsafeCallback(def, (...a: any[]) =>
      fn(...a.map((v, i) => (argIsPtr[i] ? toAddr(v) : v))));
    return { addr: toAddr(cb.pointer), close: () => cb.close() };
  },
};
