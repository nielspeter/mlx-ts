// Pytree utilities (cf. mlx.utils): treat nested objects/arrays of MX leaves as
// a single structure. Flatten to a leaf list (stable order), rebuild against a
// template, or map over leaves. The bridge between a structured parameter tree
// and the flat vector the autograd boundary needs.

import { MX } from "./mx.ts";

export type Tree = MX | Tree[] | { [k: string]: Tree };

// leaves in deterministic order (array index / object insertion order)
export function treeFlatten(tree: Tree): MX[] {
  if (tree instanceof MX) return [tree];
  if (Array.isArray(tree)) return tree.flatMap(treeFlatten);
  return Object.keys(tree).flatMap((k) => treeFlatten((tree as any)[k]));
}

// rebuild a tree shaped like `template`, filling leaves from `leaves` in order
export function treeUnflattenLike(template: Tree, leaves: MX[]): Tree {
  let i = 0;
  const build = (t: Tree): Tree => {
    if (t instanceof MX) return leaves[i++];
    if (Array.isArray(t)) return t.map(build);
    const o: any = {}; for (const k of Object.keys(t)) o[k] = build((t as any)[k]); return o;
  };
  return build(template);
}

export function treeMap(fn: (x: MX) => MX, tree: Tree): Tree {
  if (tree instanceof MX) return fn(tree);
  if (Array.isArray(tree)) return tree.map((t) => treeMap(fn, t));
  const o: any = {}; for (const k of Object.keys(tree)) o[k] = treeMap(fn, (tree as any)[k]); return o;
}
