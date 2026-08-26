// Shared GPT model for the nanochat-style pipeline: the SAME architecture
// base-train.ts pretrains (learned positional embeddings, pre-LN blocks, GELU
// MLP, separate Q/K/V, tied head), plus checkpoint load/save and generation.
// Used by chat-sft.ts (SFT from a base checkpoint) and chat-ckpt.ts (chat CLI).
import { fromI32, MX, sample, saveSafetensors, tidy } from "../core/mx.ts";
import { type Tree, treeFlatten } from "../core/pytree.ts";
import { readJson, writeJson } from "../io/fs.ts";
import { get, loadSafetensors } from "../io/loader.ts";

export type Cfg = { vocab: number; n_layer: number; n_head: number; n_embd: number; block_size: number };
const EPS = 1e-5;
const cfgPath = (ckpt: string) => ckpt.replace(/\.safetensors$/, "") + "-config.json";

// load a checkpoint into a trainable MX params tree (copied out -> owned buffers)
export async function loadCkpt(ckpt: string): Promise<{ params: Tree; cfg: Cfg }> {
  const cfg: Cfg = await readJson<Cfg>(cfgPath(ckpt));
  const w = loadSafetensors(ckpt);
  const cp = (name: string): MX => new MX(get(w, name) as number).copy();
  const blocks = Array.from({ length: cfg.n_layer }, (_, i) => {
    const p = `blocks.${i}`;
    return {
      ln1w: cp(`${p}.ln1w`), ln1b: cp(`${p}.ln1b`),
      wq: cp(`${p}.wq`), bq: cp(`${p}.bq`), wk: cp(`${p}.wk`), bk: cp(`${p}.bk`),
      wv: cp(`${p}.wv`), bv: cp(`${p}.bv`), wo: cp(`${p}.wo`), bo: cp(`${p}.bo`),
      ln2w: cp(`${p}.ln2w`), ln2b: cp(`${p}.ln2b`),
      wfc: cp(`${p}.wfc`), bfc: cp(`${p}.bfc`), wproj: cp(`${p}.wproj`), bproj: cp(`${p}.bproj`),
    };
  });
  const params: Tree = { wte: cp("wte"), wpe: cp("wpe"), blocks, lnfw: cp("lnfw"), lnfb: cp("lnfb") } as any;
  return { params, cfg };
}

// flatten the params tree to dotted keys and write a checkpoint + config
export async function saveCkpt(ckpt: string, params: any, cfg: Cfg) {
  const rec: Record<string, MX> = { wte: params.wte, wpe: params.wpe, lnfw: params.lnfw, lnfb: params.lnfb };
  params.blocks.forEach((b: any, i: number) => { for (const k of Object.keys(b)) rec[`blocks.${i}.${k}`] = b[k]; });
  saveSafetensors(ckpt, rec);
  await writeJson(cfgPath(ckpt), cfg);
}

// forward over the params tree. idx: [B, L] -> logits [B, L, vocab]
export function forward(p: any, idx: MX, cfg: Cfg): MX {
  const [Bc, L] = idx.shape, NH = cfg.n_head, DH = cfg.n_embd / NH, D = cfg.n_embd, ASCALE = DH ** -0.5;
  const pos = fromI32(Int32Array.from({ length: L }, (_, i) => i), [L]);
  let x = p.wte.takeAxis(idx, 0).add(p.wpe.takeAxis(pos, 0));
  const heads = (t: MX, wt: MX, b: MX) => t.matmul(wt).add(b).reshape([Bc, L, NH, DH]).transpose([0, 2, 1, 3]);
  for (const blk of p.blocks) {
    const n1 = x.layerNorm(blk.ln1w, blk.ln1b, EPS);
    const q = heads(n1, blk.wq, blk.bq), k = heads(n1, blk.wk, blk.bk), v = heads(n1, blk.wv, blk.bv);
    const att = MX.sdpa(q, k, v, ASCALE, true).transpose([0, 2, 1, 3]).reshape([Bc, L, D]);
    x = x.add(att.matmul(blk.wo).add(blk.bo));
    const n2 = x.layerNorm(blk.ln2w, blk.ln2b, EPS);
    x = x.add(n2.matmul(blk.wfc).add(blk.bfc).gelu().matmul(blk.wproj).add(blk.bproj));
  }
  return x.layerNorm(p.lnfw, p.lnfb, EPS).matmul(p.wte.transpose([1, 0]));
}

// autoregressive generation (context cropped to block_size; no KV cache — seqs are short)
export function generate(params: any, promptIds: number[], cfg: Cfg, eos: number, opts: { maxNew?: number; temp?: number } = {}): number[] {
  const { maxNew = 64, temp = 0.8 } = opts;
  return tidy(() => {
    const ids = [...promptIds], out: number[] = [];
    for (let i = 0; i < maxNew; i++) {
      const ctx = ids.slice(-cfg.block_size), L = ctx.length;
      const logits = forward(params, fromI32(Int32Array.from(ctx), [1, L]), cfg).reshape([L, cfg.vocab]);
      const tk = sample(logits.slice([L - 1, 0], [L, cfg.vocab]), temp, 0, 0).itemU();
      if (tk === eos) break;
      ids.push(tk); out.push(tk);
    }
    return out;
  });
}

export const freeParams = (params: any) => {
  for (const x of treeFlatten(params)) x.free();
};
