// Deterministic stand-in weights, so a config-driven model can run without a
// checkpoint.
//
// The model files are the least-covered part of the repo, and not because they
// are unverified — the parity suite checks them against MLX Python — but
// because that suite loads real checkpoints in separate processes, which
// `bun test --coverage` cannot see. A model that reads its weights by name will
// run on any tensors of the right shape, which is enough to exercise the
// forward pass and to assert the structural properties a checkpoint cannot
// tell you about, like whether attention is actually masked.
import { fromF32, type MX } from "../../src/index.ts";
import type { Weights } from "../../src/io/loader.ts";

/** Same values every run, and different per name, so nothing lines up by luck. */
export function detTensor(name: string, shape: number[]): MX {
  let seed = 7;
  for (let i = 0; i < name.length; i++) seed = (seed * 31 + name.charCodeAt(i)) % 1009;
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = ((i * 131 + seed * 977 + 7) % 1009) / 1009 - 0.5;
  return fromF32(data, shape);
}

/**
 * A `Weights` over a name -> shape map. Names absent from the spec throw the
 * same way a real checkpoint does, which matters: models use that to detect
 * optional tensors such as a missing bias or residual shortcut.
 */
export function fakeWeights(spec: Record<string, number[]>): Weights {
  const cache = new Map<string, MX>();
  return {
    mx(name: string): MX {
      const hit = cache.get(name);
      if (hit) return hit;
      const shape = spec[name];
      if (!shape) throw new Error(`missing tensor '${name}'`);
      const t = detTensor(name, shape);
      cache.set(name, t);
      return t;
    },
    done() {
      for (const t of cache.values()) t.free();
      cache.clear();
    },
  };
}
