// Adam optimizer (pure TS over MX), pytree-aware: update() takes a params tree
// and a matching grads tree and returns the updated params tree. Per-leaf
// first/second moment state is keyed by position in the flattened tree.

import { evalAll, MX, Owned, scalar } from "../core/mx.ts";
import { type Tree, treeFlatten, treeUnflattenLike } from "../core/pytree.ts";

export class Adam {
  // First and second moments, one pair per flattened leaf. Owned rather than a
  // plain array because they must outlive the update() that produced them: see
  // its doc comment for what goes wrong when the replaced pair is not freed.
  private state = new Owned<{ m: MX; v: MX }>(0);
  private t = 0;
  lr: number; b1: number; b2: number; eps: number;
  constructor(lr: number, b1 = 0.9, b2 = 0.999, eps = 1e-8) { this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps; }

  update(params: Tree, grads: Tree): Tree {
    this.t++;
    const bc1 = 1 - this.b1 ** this.t, bc2 = 1 - this.b2 ** this.t;
    const fp = treeFlatten(params), fg = treeFlatten(grads);
    const out = fp.map((p, i) => {
      const g = fg[i];
      const prev = this.state.get(i);
      const mi = (prev ? prev.m.mul(scalar(this.b1)) : scalar(0)).add(g.mul(scalar(1 - this.b1)));
      const vi = (prev ? prev.v.mul(scalar(this.b2)) : scalar(0)).add(g.mul(g).mul(scalar(1 - this.b2)));
      evalAll(mi, vi);
      // The moments must outlive this step. Called inside a tidy() — the normal
      // way to train without leaking — they would otherwise be freed as
      // scope-local intermediates and the next step would read freed handles.
      this.state.set(i, { m: mi, v: vi });
      const update = mi.div(scalar(bc1)).div(vi.div(scalar(bc2)).sqrt().add(scalar(this.eps)));
      return p.sub(update.mul(scalar(this.lr)));
    });
    return treeUnflattenLike(params, out);
  }
}
