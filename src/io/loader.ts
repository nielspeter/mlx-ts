// Safetensors loading over mlx-c: open a file into a string->array map and pull
// tensors out by name. Built on the generated `m` symbol table.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { clearCache, FLOAT32, MX } from "../core/mx.ts";
import { type Arr, m } from "../ffi/generated.ts";
import { cstring, ptr, view as toArrayBuffer } from "../ffi/index.ts";

// The safetensors Load primitive only implements eval_gpu == no -> load on CPU.
const cpuStream = m.mlx_default_cpu_stream_new() as number;
const asBig = (x: unknown) => BigInt((x as number) ?? 0);
// Not retained past the call: mlx-c copies the string, and an FFI argument only
// has to outlive the synchronous call it is passed to.
function cstr(s: string) {
  return ptr(new Uint8Array([...new TextEncoder().encode(s), 0]));
}

// The raw mlx_map_string_to_array handle. Distinct from the `Weights` accessor
// below — they were both called `Weights`, which is a duplicate-identifier error
// and silently gave consumers of the public type the wrong one.
export type WeightMap = number;

// Load a .safetensors file -> map<string, array>.
export function loadSafetensors(path: string): WeightMap {
  const arrMap = new BigUint64Array(1);
  arrMap[0] = asBig(m.mlx_map_string_to_array_new());
  const strMap = new BigUint64Array(1);
  strMap[0] = asBig(m.mlx_map_string_to_string_new());
  const rc = m.mlx_load_safetensors(ptr(arrMap), ptr(strMap), cstr(path), cpuStream);
  if (rc !== 0) throw new Error(`mlx_load_safetensors failed (${rc}) for ${path}`);
  return Number(arrMap[0]);
}

// Free the map once all needed tensors have been pulled into MX handles
// (those keep their own refcount; mmap-backed originals are released).
export function freeMap(w: WeightMap): void {
  m.mlx_map_string_to_array_free(w);
}

// Fetch one tensor by name.
export function get(w: WeightMap, key: string): Arr {
  const slot = new BigUint64Array(1);
  slot[0] = asBig(m.mlx_array_new());
  const rc = m.mlx_map_string_to_array_get(ptr(slot), w, cstr(key));
  if (rc !== 0) throw new Error(`missing tensor '${key}'`);
  return Number(slot[0]);
}

// List every (name, shape) in the map.
export function entries(w: WeightMap): { name: string; shape: number[] }[] {
  const it = m.mlx_map_string_to_array_iterator_new(w) as number;
  const keyOut = new BigUint64Array(1);
  const valOut = new BigUint64Array(1);
  valOut[0] = asBig(m.mlx_array_new());
  const out: { name: string; shape: number[] }[] = [];
  while (m.mlx_map_string_to_array_iterator_next(ptr(keyOut), ptr(valOut), it) === 0) {
    const name = cstring(Number(keyOut[0]));
    out.push({ name, shape: shapeOf(Number(valOut[0])) });
  }
  return out;
}

// ---- weights sources: single-file vs sharded streaming -------------------
// A uniform interface so a model builder is agnostic to how weights are stored.
export interface Weights {
  mx(name: string): MX;
  done(): void;
}

/**
 * Read every tensor at a different dtype — in practice, a bf16 checkpoint at
 * float32.
 *
 * bf16 has eight mantissa bits, which is plenty for generation and far too
 * coarse to compare against a float32 reference: Qwen2 carries outlier channels
 * in the thousands that cancel in the last layer, so a one-ulp difference at
 * layer 9 becomes a percent at the logits. Upcasting makes a parity check exact
 * instead of approximate.
 */
export function upcastWeights(W: Weights, dtype: number = FLOAT32): Weights {
  return { mx: (n) => W.mx(n).astype(dtype), done: () => W.done() };
}

// Whole file mmapped once; tensors returned as (refcounted) views.
export function singleFileWeights(path: string): Weights {
  const w = loadSafetensors(path);
  // done() is idempotent. A model may release the map once it holds its own
  // tensor references, and the caller may reasonably release it too; a second
  // mlx_map_string_to_array_free on the same handle is a double free.
  let closed = false;
  return {
    mx: (n) => new MX(get(w, n)),
    done: () => { if (!closed) { closed = true; freeMap(w); } },
  };
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
    if (h === undefined) {
      h = loadSafetensors(join(dir, shard));
      maps.set(shard, h);
    }
    return h;
  };
  return {
    mx: (n) => new MX(get(mapFor(idx[n]), n)),
    done: () => {
      for (const h of maps.values()) freeMap(h);
      maps.clear();
      clearCache();
    },
  };
}

export function shapeOf(a: Arr): number[] {
  const n = Number(m.mlx_array_ndim(a));
  const p = Number(m.mlx_array_shape(a));
  if (n === 0 || p === 0) return [];
  return Array.from(new Int32Array(toArrayBuffer(p, 0, n * 4)));
}
