// Decoder weights read one layer at a time, from the checkpoint as it is on disk.
//
// A resident model keeps every layer in memory for its whole life. A LayerStore
// keeps only what every step needs — embeddings, the final norm, an output head
// — and loads each decoder layer when the forward pass reaches it, then lets it
// go. Peak memory becomes one layer plus the shared tensors instead of the whole
// model, which is what lets a checkpoint larger than memory run at all. The
// price is a read per layer per step: on a model that already fits, it is pure
// overhead.
//
// No files are rewritten. A layer is read from whichever file already holds it,
// a single .safetensors or a shard named by an index, by reopening that file's
// weight map. Reopening is the mechanism, not an optimisation left undone: a
// map keeps a reference to every tensor it has handed out, so an evaluated
// layer stays in memory until its map is freed, and mlx-c offers no way to drop
// one entry. Splitting the checkpoint into a file per layer was considered and
// rejected: it would only save re-parsing a header, well under 1% of the cost of
// reading a large layer, and it would double the checkpoint on disk.
//
//   const store = layerStore("model.safetensors.index.json");
//   const model = new Qwen3(config, store);

import { closeSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { dirname, join } from "node:path";
import { clearCache, evalAll, MX } from "../core/mx.ts";
import { freeMap, get, loadSafetensors, type Weights } from "./loader.ts";

/** Hugging Face decoder naming: `model.layers.<i>.…`. */
const DEFAULT_LAYER_PATTERN = /^model\.layers\.(\d+)\./;

export type LayerStoreOptions = {
  /** Which decoder layer a tensor name belongs to; its first capture group is the index. */
  layerPattern?: RegExp;
};

export interface LayerStore {
  readonly numLayers: number;
  /**
   * Every tensor that is not part of a decoder layer: embeddings, the final
   * norm, an untied output head. Resident until `done()`; its own `done()` does
   * nothing, because the store owns it.
   */
  readonly shared: Weights;
  /**
   * Load layer `i`, run `fn` over it, evaluate the arrays `fn` returns, then
   * release the layer.
   *
   * The evaluation is the point. MLX is lazy: an array that has not been
   * evaluated still references the weights that will produce it, so releasing
   * a layer before its outputs exist would either keep that layer in memory or
   * read it after it is gone. Doing it here means a model cannot get the order
   * wrong. Any weight `fn` returns directly is left alive for the caller.
   */
  withLayer(i: number, fn: (layer: Weights) => MX[]): MX[];
  /** Release the shared tensors. Idempotent. */
  done(): void;
}

/** Tensor names in a safetensors file, from its header alone — no weights read. */
function tensorNames(path: string): string[] {
  const fd = openSync(path, "r");
  try {
    const size = fstatSync(fd).size;
    const prefix = new Uint8Array(8);
    if (size < 8 || readSync(fd, prefix, 0, 8, 0) !== 8) {
      throw new Error(`${path}: not a safetensors file (${size} bytes, too short for its length prefix)`);
    }
    const len = new DataView(prefix.buffer).getBigUint64(0, true);
    // Bounded by the file before it sizes an allocation: a corrupt or non-
    // safetensors file would otherwise ask for an arbitrary number of bytes.
    if (len === 0n || len > BigInt(size - 8)) {
      throw new Error(`${path}: not a safetensors file (header length ${len} does not fit ${size} bytes)`);
    }
    const header = new Uint8Array(Number(len));
    readSync(fd, header, 0, header.length, 8);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(new TextDecoder().decode(header));
    } catch {
      throw new Error(`${path}: not a safetensors file (header is not JSON)`);
    }
    return Object.keys(parsed).filter((k) => k !== "__metadata__");
  } finally {
    closeSync(fd);
  }
}

/** The weight map for `file`, opened on first use. */
function mapFor(maps: Map<string, number>, file: string): number {
  let map = maps.get(file);
  if (map === undefined) {
    map = loadSafetensors(file);
    maps.set(file, map);
  }
  return map;
}

/**
 * Open a checkpoint for layer-by-layer reading.
 *
 * `checkpoint` is a `.safetensors` file or a `model.safetensors.index.json`.
 * Only headers are read here; no tensor is touched until it is used.
 */
export function layerStore(checkpoint: string, opts: LayerStoreOptions = {}): LayerStore {
  const pattern = opts.layerPattern ?? DEFAULT_LAYER_PATTERN;

  // file -> names, taken from each file's own header rather than trusting an
  // index: an index that disagrees with its shards should fail here, by name,
  // not as a missing tensor halfway through a forward pass.
  const files = new Map<string, string[]>();
  if (checkpoint.endsWith(".json")) {
    const index = JSON.parse(readFileSync(checkpoint, "utf8")).weight_map as Record<string, string>;
    const dir = dirname(checkpoint);
    for (const shard of new Set(Object.values(index)))
      files.set(join(dir, shard), tensorNames(join(dir, shard)));
    for (const [name, shard] of Object.entries(index)) {
      if (!files.get(join(dir, shard))!.includes(name)) {
        throw new Error(`${checkpoint}: index places ${name} in ${shard}, which does not contain it`);
      }
    }
  } else {
    files.set(checkpoint, tensorNames(checkpoint));
  }

  // name -> file, per layer and for the shared remainder.
  const layers = new Map<number, Map<string, string>>();
  const sharedOwner = new Map<string, string>();
  const layerOf = new Map<string, number>();
  for (const [file, names] of files) {
    for (const name of names) {
      const m = name.match(pattern);
      if (!m) {
        sharedOwner.set(name, file);
        continue;
      }
      const i = Number(m[1]);
      layerOf.set(name, i);
      let owner = layers.get(i);
      if (!owner) {
        owner = new Map();
        layers.set(i, owner);
      }
      owner.set(name, file);
    }
  }
  const numLayers = layers.size;
  if (numLayers === 0) {
    throw new Error(
      `${checkpoint}: no tensor name matches ${pattern}, so there are no decoder layers to stream`,
    );
  }
  for (let i = 0; i < numLayers; i++) {
    if (!layers.has(i))
      throw new Error(`${checkpoint}: decoder layers are not contiguous — layer ${i} is missing`);
  }

  const wrongPlace = (name: string, here: string): Error => {
    const i = layerOf.get(name);
    if (i !== undefined)
      return new Error(`${name} belongs to layer ${i}; read it inside withLayer(${i}, …), not ${here}`);
    if (sharedOwner.has(name))
      return new Error(`${name} is a shared tensor; read it from store.shared, not ${here}`);
    return new Error(`${name}: no such tensor in ${checkpoint}`);
  };

  const sharedMaps = new Map<string, number>();
  const shared: Weights = {
    mx(name) {
      const file = sharedOwner.get(name);
      if (file === undefined) throw wrongPlace(name, "store.shared");
      return new MX(get(mapFor(sharedMaps, file), name));
    },
    done() {},
  };

  return {
    numLayers,
    shared,
    withLayer(i, fn) {
      const owner = layers.get(i);
      if (!owner) throw new RangeError(`layer ${i} is out of range: ${checkpoint} has ${numLayers} layers`);
      const maps = new Map<string, number>();
      const opened: MX[] = [];
      const layer: Weights = {
        mx(name) {
          const file = owner.get(name);
          if (file === undefined) throw wrongPlace(name, `withLayer(${i}, …)`);
          const x = new MX(get(mapFor(maps, file), name));
          opened.push(x);
          return x;
        },
        done() {},
      };
      let out: MX[] = [];
      try {
        out = fn(layer);
        evalAll(...out);
        return out;
      } finally {
        const keep = new Set(out);
        for (const x of opened) if (!keep.has(x)) x.free();
        for (const map of maps.values()) freeMap(map);
      }
    },
    done() {
      for (const map of sharedMaps.values()) freeMap(map);
      sharedMaps.clear();
      clearCache();
    },
  };
}
