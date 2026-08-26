// Node backend, via koffi — a prebuilt native addon, so no C/C++ of ours and no
// toolchain for users. koffi hands pointers back as BigInt addresses and takes
// either a number or a BigInt going in, so the address round-trip is cheaper
// here than on Deno: a Number() on the way out and nothing on the way in.
import koffi from "koffi";
import type { Backend, Callback, CType, SymbolSpec, Symbols, SymbolTable } from "./types.ts";

const TYPE: Record<CType, string> = {
  ptr: "void *", cstring: "str", void: "void", bool: "bool",
  i8: "int8_t", u8: "uint8_t", i16: "int16_t", u16: "uint16_t",
  i32: "int32_t", u32: "uint32_t", i64: "int64_t", u64: "uint64_t",
  usize: "size_t", f32: "float", f64: "double",
};

// koffi wants a C prototype for callbacks; give each a unique name.
let cbSeq = 0;

export const backend: Backend = {
  name: "node",
  version: `node ${process.versions.node} + koffi ${koffi.version}`,

  open(lib: string, symbols: SymbolTable): Symbols {
    const handle = koffi.load(lib);
    const out: Symbols = Object.create(null);
    for (const [name, spec] of Object.entries(symbols)) {
      let raw: (...a: any[]) => any;
      try {
        raw = handle.func(name, TYPE[spec.returns], spec.args.map((a) => TYPE[a]));
      } catch {
        continue;                       // symbol absent in this mlx-c build
      }
      // koffi accepts numbers for pointer args, so only the return needs work.
      out[name] = spec.returns === "ptr" ? (...a: any[]) => Number(raw(...a)) : raw;
    }
    return out;
  },

  ptr: (v: ArrayBufferView) => Number(koffi.address(v)),
  view: (addr: number, off: number, len: number) => koffi.view(addr + off, len),
  cstring: (addr: number) => koffi.decode(addr, "char *"),

  callback(spec: SymbolSpec, fn: (...args: any[]) => any): Callback {
    const proto = koffi.proto(
      `${TYPE[spec.returns]} mlxts_cb_${cbSeq++}(${spec.args.map((a) => TYPE[a]).join(", ") || "void"})`,
    );
    // Pointer args arrive as BigInt/external; the JS side speaks numbers.
    const argIsPtr = spec.args.map((a) => a === "ptr" || a === "cstring");
    const wrapped = (...a: any[]) => fn(...a.map((v, i) => (argIsPtr[i] ? Number(koffi.address(v)) : v)));
    const registered = koffi.register(wrapped, koffi.pointer(proto));
    return { addr: Number(koffi.address(registered)), close: () => koffi.unregister(registered) };
  },
};
