// op-codegen: parse mlx-c headers -> emit a Bun-FFI symbol table + typed TS
// wrappers. Run: bun codegen.ts   (writes src/ffi/generated.ts + a coverage report)
//
// Strategy: every mlx-c op is `int fn(mlx_array* res, ...args, mlx_stream s)`.
// We parse each decl, map C types -> FFI types for the dlopen table, and for the
// standard single-output ops also emit an ergonomic wrapper.

import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";

// mlx-c headers. `MLXTS_INCLUDE` overrides; otherwise prefer Homebrew's
// version-independent symlinks (so a `brew upgrade mlx-c` doesn't break this),
// then fall back to the newest versioned Cellar install.
function resolveInclude(): string {
  const cellar = (root: string) =>
    existsSync(root) ? readdirSync(root).sort().reverse().map((v) => `${root}/${v}/include/mlx/c`) : [];
  const candidates = [
    process.env.MLXTS_INCLUDE,
    "/opt/homebrew/opt/mlx-c/include/mlx/c",  // Homebrew, Apple silicon
    "/opt/homebrew/include/mlx/c",
    "/usr/local/opt/mlx-c/include/mlx/c",     // Homebrew, Intel
    "/usr/local/include/mlx/c",
    ...cellar("/opt/homebrew/Cellar/mlx-c"),
    ...cellar("/usr/local/Cellar/mlx-c"),
  ].filter(Boolean) as string[];
  const hit = candidates.find((p) => existsSync(`${p}/ops.h`));
  if (!hit) {
    console.error(
      "mlx-c headers not found (looking for ops.h).\n" +
      "Install with `brew install mlx-c`, or set MLXTS_INCLUDE to the dir containing ops.h.\nTried:\n" +
      candidates.map((c) => `  ${c}`).join("\n"),
    );
    process.exit(1);
  }
  return hit;
}

const INC = resolveInclude();
// Runtime headers (for the symbol table) + the op headers we actually wrap.
const RUNTIME_HEADERS = ["array.h", "string.h", "stream.h", "vector.h", "map.h", "io.h", "memory.h", "random.h", "transforms.h", "closure.h"];
const OP_HEADERS = ["ops.h", "fast.h"];

type Param = { ctype: string; name: string; nullable: boolean };
type Fn = { name: string; ret: string; params: Param[]; header: string };

// ---- parse -------------------------------------------------------------
function parseHeader(header: string): Fn[] {
  const text = readFileSync(`${INC}/${header}`, "utf8");
  const lines = text.split("\n");
  const fns: Fn[] = [];
  let inDoc = false;
  let buf = "";
  for (const raw of lines) {
    const line = raw;
    if (inDoc) { if (line.includes("*/")) inDoc = false; continue; }
    const t = line.trimStart();
    if (t.startsWith("/*") && !line.includes("*/")) { inDoc = true; continue; }
    if (t.startsWith("/*") && line.includes("*/")) continue; // one-line doc
    if (buf === "" && !/\bmlx_[a-z0-9_]+\s*\(/.test(line)) continue;
    buf += " " + line;
    if (buf.includes(");")) {
      const decl = buf.replace(/\s+/g, " ").trim();
      buf = "";
      const mm = decl.match(/^([A-Za-z_][A-Za-z0-9_ \t]*?\*?)\s*\b(mlx_[a-z0-9_]+)\s*\(([\s\S]*)\)\s*;/);
      if (!mm) continue;
      const [, ret, name, argstr] = mm;
      const params = splitArgs(argstr);
      fns.push({ name, ret: norm(ret), params, header });
    }
  }
  return fns;
}

function splitArgs(s: string): Param[] {
  if (s.trim() === "void" || s.trim() === "") return [];
  // no nested-paren args survive here (function-pointer params are rejected later)
  return s.split(",").map((rawArg) => {
    const nullable = /may be null/.test(rawArg);
    const a = rawArg.replace(/\/\*[\s\S]*?\*\//g, "").trim();
    // split into type + name: name is the trailing identifier
    const m2 = a.match(/^(.*?)([A-Za-z_][A-Za-z0-9_]*)$/);
    if (!m2) return { ctype: norm(a), name: "_", nullable };
    let [, type, name] = m2;
    // pointer star belongs to the type
    if (type.trim().endsWith("*")) type = type.trim();
    return { ctype: norm(type), name, nullable };
  });
}

const norm = (s: string) =>
  s.replace(/\bconst\b/g, "").replace(/\s*\*\s*/g, "*").replace(/\s+/g, " ").trim();

// ---- C type -> FFI type ------------------------------------------------
const FFI: Record<string, string> = {
  "mlx_array": "ptr", "mlx_array*": "ptr",
  "mlx_vector_array": "ptr", "mlx_vector_array*": "ptr",
  "mlx_string": "ptr", "mlx_string*": "ptr",
  "mlx_stream": "ptr", "mlx_stream*": "ptr",
  "mlx_dtype": "i32", "mlx_dtype*": "ptr",
  "mlx_optional_float": "u64", "mlx_optional_int": "u64",
  "int": "i32", "int*": "ptr", "bool": "bool", "bool*": "ptr",
  "float": "f32", "float*": "ptr", "double": "f64", "double*": "ptr",
  "size_t": "u64", "size_t*": "ptr", "int64_t": "i64", "uint64_t": "u64",
  "uint32_t": "u32", "char*": "ptr", "char**": "ptr", "void": "void", "void*": "ptr",
  // pointer-to-primitive (out params for scalar read-back, etc.)
  "uint8_t*": "ptr", "uint16_t*": "ptr", "uint32_t*": "ptr", "uint64_t*": "ptr",
  "int8_t*": "ptr", "int16_t*": "ptr", "int32_t*": "ptr", "int64_t*": "ptr",
  "mlx_vector_string": "ptr", "mlx_vector_string*": "ptr",
  "mlx_vector_int": "ptr", "mlx_vector_int*": "ptr",
};
// Any other opaque mlx_* handle (or pointer to one) is a single-pointer struct
// -> model as ptr. Catches map / io / device / future handle types for free.
function ffiOf(ct: string): string | null {
  if (FFI[ct]) return FFI[ct];
  if (/^mlx_[a-z0-9_]+\*?$/.test(ct)) return "ptr";
  return null;
}

// ---- generate ----------------------------------------------------------
const RESERVED = new Set(
  "break case catch class const continue debugger default delete do else enum export extends false finally for function if import in instanceof new null return super switch this throw true try typeof var void while with yield let static await async".split(" "));

// Identifiers the emitted wrapper body uses itself. A C parameter named `m`
// (mlx_eye, mlx_tri) would shadow the FFI symbol table and make the wrapper
// throw at runtime — so rename any parameter that collides.
const SHADOWS = new Set(
  "m r ptr stream asBig kptr cstr optFloatU64 optIntU64 KEEP Arr Vec Dtype".split(" "));

const safeParam = (n: string): string => (SHADOWS.has(n) || RESERVED.has(n) ? `${n}_` : n);

function camel(mlxName: string): string {
  let n = mlxName.replace(/^mlx_/, "").replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());
  if (RESERVED.has(n)) n += "_";
  return n;
}

const allFns = [...RUNTIME_HEADERS, ...OP_HEADERS].flatMap(parseHeader);

const symbolLines: string[] = [];
const wrapperLines: string[] = [];
const skipped: { name: string; reason: string }[] = [];
let symCount = 0, wrapCount = 0;

for (const fn of allFns) {
  // map every param + return to FFI; bail (report) on anything unsupported
  const bad = fn.params.find((p) => ffiOf(p.ctype) === null);
  if (bad) { skipped.push({ name: fn.name, reason: `unsupported param type '${bad.ctype}'` }); continue; }
  const retFfi = fn.ret === "int" ? "i32" : ffiOf(fn.ret);
  if (retFfi === null) { skipped.push({ name: fn.name, reason: `unsupported return '${fn.ret}'` }); continue; }

  const argFfi = fn.params.map((p) => ffiOf(p.ctype));
  symbolLines.push(`  ${fn.name}: { args: [${argFfi.map((a) => `"${a}"`).join(", ")}], returns: "${retFfi}" },`);
  symCount++;

  // wrap only standard single-output ops from the op headers
  if (!OP_HEADERS.includes(fn.header)) continue;
  const outs = fn.params.filter((p) => p.ctype === "mlx_array*" || p.ctype === "mlx_vector_array*");
  if (fn.ret !== "int" || outs.length !== 1 || fn.params[0] !== outs[0]) {
    skipped.push({ name: fn.name, reason: "non-standard (multi/zero output)" });
    continue;
  }
  const isVec = outs[0].ctype === "mlx_vector_array*";
  const tsParams: string[] = [];
  const callArgs: string[] = ["ptr(r)"];
  let ok = true;
  for (let i = 1; i < fn.params.length; i++) {
    const p = { ...fn.params[i], name: safeParam(fn.params[i].name) };
    const ct = p.ctype;
    if (ct === "mlx_stream") { callArgs.push("stream"); continue; }
    if (ct === "int*") {                         // const int* X, size_t X_num -> number[]
      const next = fn.params[i + 1];
      if (next && next.ctype === "size_t") {
        tsParams.push(`${p.name}: number[]`);
        callArgs.push(`kptr(new Int32Array(${p.name}))`, `BigInt(${p.name}.length)`);
        i++; continue;
      }
      ok = false; break;
    }
    switch (ct) {
      case "mlx_array":
        tsParams.push(`${p.name}: Arr${p.nullable ? " | null" : ""}`);
        callArgs.push(p.nullable ? `(${p.name} ?? 0)` : p.name); break;
      case "mlx_vector_array": tsParams.push(`${p.name}: Vec`); callArgs.push(p.name); break;
      case "int": case "float": case "double": tsParams.push(`${p.name}: number`); callArgs.push(p.name); break;
      case "bool": tsParams.push(`${p.name}: boolean`); callArgs.push(p.name); break;
      case "size_t": tsParams.push(`${p.name}: number`); callArgs.push(`BigInt(${p.name})`); break;
      case "mlx_dtype": tsParams.push(`${p.name}: number`); callArgs.push(p.name); break;
      case "mlx_optional_float": tsParams.push(`${p.name}: number | null`); callArgs.push(`optFloatU64(${p.name})`); break;
      case "mlx_optional_int": tsParams.push(`${p.name}: number | null`); callArgs.push(`optIntU64(${p.name})`); break;
      case "char*": tsParams.push(`${p.name}: string`); callArgs.push(`cstr(${p.name})`); break;
      default: ok = false;
    }
    if (!ok) break;
  }
  if (!ok) { skipped.push({ name: fn.name, reason: "unwrappable param shape" }); continue; }

  const retT = isVec ? "Vec" : "Arr";
  const init = isVec ? "m.mlx_vector_array_new()" : "m.mlx_array_new()";
  wrapperLines.push(
    `export function ${camel(fn.name)}(${tsParams.join(", ")}): ${retT} {\n` +
    `  const r = new BigUint64Array(1); r[0] = asBig(${init});\n` +
    `  m.${fn.name}(${callArgs.join(", ")});\n` +
    `  return Number(r[0]);\n}`,
  );
  wrapCount++;
}

// ---- runtime preamble (hand-written, fixed) ----------------------------
const PREAMBLE = `// AUTO-GENERATED by codegen.ts — do not edit by hand.
import { open, ptr } from "./index.ts";
import { LIBMLXC } from "./native-lib.ts";

// The symbol table is runtime-neutral: the backend maps these CTypes onto
// bun:ffi / Deno.dlopen / koffi, and hands every pointer back as a number.
export const m = open(LIBMLXC, {
${symbolLines.join("\n")}
}) as any;

export type Arr = number;
export type Vec = number;
// A plain object, not a const enum: TS enums are not erasable, and Node runs
// .ts by stripping types only. Same call sites (Dtype.Float32), same values.
export const Dtype = {
  Bool: 0, Uint8: 1, Uint16: 2, Uint32: 3, Uint64: 4, Int8: 5, Int16: 6, Int32: 7,
  Int64: 8, Float16: 9, Float32: 10, Float64: 11, Bfloat16: 12, Complex64: 13,
} as const;
export type Dtype = (typeof Dtype)[keyof typeof Dtype];

const asBig = (x: unknown) => BigInt((x as number) ?? 0);
const KEEP: unknown[] = [];
const kptr = (a: ArrayBufferView) => { KEEP.push(a); return ptr(a); };
function optFloatU64(v: number | null): bigint {
  if (v === null) return 0n;
  const bits = new Uint32Array(new Float32Array([v]).buffer)[0];
  return BigInt(bits) | (1n << 32n);
}
const optIntU64 = (v: number | null): bigint => v === null ? 0n : BigInt.asUintN(32, BigInt(v)) | (1n << 32n);
function cstr(s: string) { const b = new Uint8Array([...new TextEncoder().encode(s), 0]); KEEP.push(b); return ptr(b); }

export const stream = m.mlx_default_gpu_stream_new() as number;

// minimal runtime helpers used by callers
export function array(data: Float32Array, shape: number[]): Arr {
  KEEP.push(data); const sh = new Int32Array(shape); KEEP.push(sh);
  return m.mlx_array_new_data(ptr(data), ptr(sh), shape.length, Dtype.Float32) as number;
}
export function arrayI32(data: Int32Array, shape: number[]): Arr {
  KEEP.push(data); const sh = new Int32Array(shape); KEEP.push(sh);
  return m.mlx_array_new_data(ptr(data), ptr(sh), shape.length, Dtype.Int32) as number;
}
export function item(a: Arr): number {
  m.mlx_array_eval(a); const o = new Float32Array(1); m.mlx_array_item_float32(ptr(o), a); return o[0];
}
export function itemU32(a: Arr): number {
  m.mlx_array_eval(a); const o = new Uint32Array(1); m.mlx_array_item_uint32(ptr(o), a); return o[0];
}
export const evalArray = (...a: Arr[]): void => { for (const h of a) m.mlx_array_eval(h); };
// build an mlx_vector_array from a list of handles (for concatenate, etc.)
export function vec(arrs: Arr[]): Vec {
  const buf = new BigUint64Array(arrs.length);
  for (let i = 0; i < arrs.length; i++) buf[i] = BigInt(arrs[i] ?? 0);
  KEEP.push(buf);
  return m.mlx_vector_array_new_data(ptr(buf), BigInt(arrs.length)) as number;
}
export function show(a: Arr): string {
  const s = new BigUint64Array(1); s[0] = asBig(m.mlx_string_new());
  m.mlx_array_tostring(ptr(s), a); return m.mlx_string_data(Number(s[0])) as unknown as string;
}
`;

writeFileSync(`${import.meta.dir}/../src/ffi/generated.ts`,
  PREAMBLE + "\n// ---- generated op wrappers ----\n" + wrapperLines.join("\n\n") + "\n");

// ---- report ------------------------------------------------------------
const byReason: Record<string, number> = {};
for (const s of skipped) byReason[s.reason.replace(/'[^']*'/, "'…'")] = (byReason[s.reason.replace(/'[^']*'/, "'…'")] ?? 0) + 1;
console.log(`parsed   ${allFns.length} decls across ${RUNTIME_HEADERS.length + OP_HEADERS.length} headers`);
console.log(`symbols  ${symCount} FFI entries`);
console.log(`wrappers ${wrapCount} typed op wrappers (from ${OP_HEADERS.join(", ")})`);
console.log(`skipped  ${skipped.length}:`);
for (const [r, n] of Object.entries(byReason).sort((a, b) => b[1] - a[1])) console.log(`  ${n.toString().padStart(3)}  ${r}`);
console.log(`\nwrote src/ffi/generated.ts`);
