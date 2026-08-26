// One decoding step of MusicGen's LM vs Hugging Face's own implementation
// (reference/reference-musicgen.py). Same local weights, same inputs.
//   bun validation/musicgen-lm.ts

import { activeMemoryMB, fromF32, fromU32, Owned, tidy } from "../src/core/mx.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";
import { type LayerKV, type MusicGenConfig, MusicGenLM } from "../src/models/musicgen.ts";

const REPO = "facebook/musicgen-small";
const cfg = (await readJson<any>(await hubFile(REPO, "config.json"))).decoder as MusicGenConfig;
const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));
const lm = new MusicGenLM(cfg, W);

const B = 1, K = cfg.num_codebooks, Lt = 6, D = cfg.hidden_size;
const tok = new Uint32Array(K);
for (let k = 0; k < K; k++) tok[k] = (k * 977 + 7) % 2048;
const cond = new Float32Array(Lt * D);
for (let i = 0; i < Lt; i++) for (let j = 0; j < D; j++) cond[i * D + j] = ((i * 131 + j * 977 + 7) % 1009) / 1009 - 0.5;

using cache = new Owned<LayerKV>(cfg.num_hidden_layers);
const logits = tidy(() => lm.step(fromU32(tok, [B, 1, K]), fromF32(cond, [B, Lt, D]), cache, 0));

const a = logits.toF32();                       // [B, 1, vocab, K]
// HF returns [B*K, 1, vocab]; ours is [B, 1, vocab, K]. Reorder to compare.
const vocab = cfg.vocab_size;
const flat: number[] = [];
for (let k = 0; k < K; k++) for (let v = 0; v < vocab; v++) flat.push(a[v * K + k]);
console.log(`  logits shape: (${K}, 1, ${vocab})`);
console.log(`  first8      : ${flat.slice(0, 8).map((v) => v.toFixed(5)).join(", ")}`);
console.log(`  sum         : ${flat.reduce((s, v) => s + v, 0).toFixed(5)}`);
console.log(`  active memory: ${activeMemoryMB().toFixed(0)} MB`);
