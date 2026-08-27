// The transformer stack shared by CLIP's two towers.
//
// Text and vision are the same pre-norm blocks over different embeddings: the
// text side is causal and starts from token ids, the vision side is not and
// starts from image patches. Keeping one copy means a fix to the attention
// shapes cannot land in one tower and miss the other.
import { MX } from "../core/mx.ts";

export type ClipStack = {
  /** Fetch a weight by its name under the tower's prefix. */
  w: (name: string) => MX;
  layers: number;
  heads: number;
  eps: number;
  /** CLIP's own activation, x * sigmoid(1.702x); "gelu" selects the erf one. */
  quickGelu: boolean;
  /** Text is trained with a causal mask; vision sees the whole image. */
  causal: boolean;
};

/** `[B, L, D]` in, `[B, L, D]` out, through `layers` pre-norm blocks. */
export function clipBlocks(x0: MX, prefix: string, s: ClipStack): MX {
  const [B, L, D] = x0.shape;
  const Dh = D / s.heads;
  const scale = 1 / Math.sqrt(Dh);

  const lin = (name: string, t: MX) =>
    t.matmul(s.w(`${name}.weight`).transpose([1, 0])).add(s.w(`${name}.bias`));
  const ln = (name: string, t: MX) =>
    t.layerNorm(s.w(`${name}.weight`), s.w(`${name}.bias`), s.eps);
  const act = (t: MX) => (s.quickGelu ? t.mul(t.mulScalar(1.702).sigmoid()) : t.gelu());
  const heads = (t: MX) => t.reshape([B, L, s.heads, Dh]).transpose([0, 2, 1, 3]);

  let x = x0;
  for (let l = 0; l < s.layers; l++) {
    const p = `${prefix}.encoder.layers.${l}`;
    const y = ln(`${p}.layer_norm1`, x);
    const o = MX.sdpa(
      heads(lin(`${p}.self_attn.q_proj`, y)),
      heads(lin(`${p}.self_attn.k_proj`, y)),
      heads(lin(`${p}.self_attn.v_proj`, y)),
      scale, s.causal,
    ).transpose([0, 2, 1, 3]).reshape([B, L, D]);
    x = x.add(lin(`${p}.self_attn.out_proj`, o));

    const y2 = ln(`${p}.layer_norm2`, x);
    x = x.add(lin(`${p}.mlp.fc2`, act(lin(`${p}.mlp.fc1`, y2))));
  }
  return x;
}
