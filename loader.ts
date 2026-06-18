// Safetensors loading over mlx-c: open a file into a string->array map and pull
// tensors out by name. Built on the generated `m` symbol table.

import { ptr, CString, toArrayBuffer } from "bun:ffi";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { m, type Arr } from "./generated.ts";
import { MX, clearCache } from "./mx.ts";

// The safetensors Load primitive only implements eval_gpu == no -> load on CPU.
const cpuStream = m.mlx_default_cpu_stream_new() as number;
const asBig = (x: unknown) => BigInt((x as number) ?? 0);
const KEEP: unknown[] = [];
function cstr(s: string) { const b = new Uint8Array([...new TextEncoder().encode(s), 0]); KEEP.push(b); return ptr(b); }

export type Weights = number; // handle to mlx_map_string_to_array

// Load a .safetensors file -> map<string, array>.
export function loadSafetensors(path: string): Weights {
  const arrMap = new BigUint64Array(1); arrMap[0] = asBig(m.mlx_map_string_to_array_new());
  const strMap = new BigUint64Array(1); strMap[0] = asBig(m.mlx_map_string_to_string_new());
  const rc = m.mlx_load_safetensors(ptr(arrMap), ptr(strMap), cstr(path), cpuStream);
  if (rc !== 0) throw new Error(`mlx_load_safetensors failed (${rc}) for ${path}`);
  return Number(arrMap[0]);
}

// Free the map once all needed tensors have been pulled into MX handles
// (those keep their own refcount; mmap-backed originals are released).
export function freeMap(w: Weights): void { m.mlx_map_string_to_array_free(w); }

// Fetch one tensor by name.
export function get(w: Weights, key: string): Arr {
  const slot = new BigUint64Array(1); slot[0] = asBig(m.mlx_array_new());
  const rc = m.mlx_map_string_to_array_get(ptr(slot), w, cstr(key));
  if (rc !== 0) throw new Error(`missing tensor '${key}'`);
  return Number(slot[0]);
}

// List every (name, shape) in the map.
export function entries(w: Weights): { name: string; shape: number[] }[] {
  const it = m.mlx_map_string_to_array_iterator_new(w) as number;
  const keyOut = new BigUint64Array(1);
  const valOut = new BigUint64Array(1); valOut[0] = asBig(m.mlx_array_new());
  const out: { name: string; shape: number[] }[] = [];
  while (m.mlx_map_string_to_array_iterator_next(ptr(keyOut), ptr(valOut), it) === 0) {
    const name = new CString(Number(keyOut[0])).toString();
    out.push({ name, shape: shapeOf(Number(valOut[0])) });
  }
  return out;
}

// ---- weights sources: single-file vs sharded streaming -------------------
// A uniform interface so a model builder is agnostic to how weights are stored.
export interface Weights { mx(name: string): MX; done(): void; }

// Whole file mmapped once; tensors returned as (refcounted) views.
export function singleFileWeights(path: string): Weights {
  const w = loadSafetensors(path);
  return { mx: (n) => new MX(get(w, n)), done: () => freeMap(w) };
}

// Multi-file (sharded) checkpoints — the norm for large MoE. Each shard is
// mmapped once on first access and tensors are returned as views; the OS pages
// unused regions out, so resident memory stays ~the working set (stacked
// weights), the same as a single file. This is what actually enables loading a
// large sharded MoE — not a per-shard materialize-and-free, which would copy
// every tensor to the heap and *lose* the mmap's evictability (measured worse).
export function shardedWeights(indexPath: string): Weights {
  const idx = JSON.parse(readFileSync(indexPath, "utf8")).weight_map as Record<string, string>;
  const dir = dirname(indexPath);
  const maps = new Map<string, number>();
  const mapFor = (shard: string) => {
    let h = maps.get(shard);
    if (h === undefined) { h = loadSafetensors(join(dir, shard)); maps.set(shard, h); }
    return h;
  };
  return {
    mx: (n) => new MX(get(mapFor(idx[n]), n)),
    done: () => { for (const h of maps.values()) freeMap(h); maps.clear(); clearCache(); },
  };
}

export function shapeOf(a: Arr): number[] {
  const n = Number(m.mlx_array_ndim(a));
  const p = Number(m.mlx_array_shape(a));
  if (n === 0 || p === 0) return [];
  return Array.from(new Int32Array(toArrayBuffer(p, 0, n * 4)));
}
