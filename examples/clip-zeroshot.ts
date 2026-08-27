// Zero-shot image classification with CLIP: no training, just a photo and some
// sentences.
//
//   bun examples/clip-zeroshot.ts cats.jpg "a photo of a cat" "a photo of a dog"
//   bun examples/clip-zeroshot.ts photo.png            # uses a default prompt set
//
// First run downloads ~1.7 GB (both CLIP towers) into ~/.cache/mlx-ts.
//
// How it works: the vision tower turns the image into a vector, the text tower
// turns each sentence into one, and both are projected into a shared space
// where cosine similarity means something. Whichever sentence sits closest is
// the answer — the model was never trained on these labels.
//
// Needs ffmpeg on PATH for decoding, the same as the Whisper example.
import {
  ClipTextEncoder, type ClipConfig, ClipTokenizer, ClipVisionEncoder,
  type ClipVisionConfig, fromU32, loadImage,
} from "../src/index.ts";
import { readJson } from "../src/io/fs.ts";
import { hubFile } from "../src/io/hub.ts";
import { singleFileWeights } from "../src/io/loader.ts";

const REPO = "openai/clip-vit-large-patch14";
const [file, ...rest] = process.argv.slice(2);
if (!file) {
  console.error('usage: bun examples/clip-zeroshot.ts <image> ["prompt" ...]');
  process.exit(1);
}
const prompts = rest.length ? rest : [
  "a photo of a cat", "a photo of a dog", "a photo of a person",
  "a photo of a car", "a landscape photograph", "a diagram or chart",
];

const t0 = performance.now();
const cfg = await readJson<{ text_config: ClipConfig; vision_config: ClipVisionConfig }>(
  await hubFile(REPO, "config.json"));
const W = singleFileWeights(await hubFile(REPO, "model.safetensors"));
const vision = new ClipVisionEncoder(cfg.vision_config, W);
const text = new ClipTextEncoder(cfg.text_config, W);
const tok = await ClipTokenizer.fromFiles(
  await hubFile(REPO, "vocab.json"), await hubFile(REPO, "merges.txt"));
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s\n`);

const size = cfg.vision_config.image_size;
const img = vision.embed(
  (await import("../src/index.ts")).fromF32(await loadImage(file, { size }), [1, size, size, 3]),
  W.mx("visual_projection.weight"),
).toF32();

const norm = (v: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
};
const imgNorm = norm(img);

const scored = prompts.map((p) => {
  const ids = tok.encode(p, { padTo: 77 });
  const h = text.encode(fromU32(Uint32Array.from(ids), [1, ids.length]));
  // CLIP pools the EOS position — the only one that has seen the whole prompt,
  // because the text tower is causal.
  const eos = ids.indexOf(tok.eosId);
  const [B, , D] = h.shape;
  const pooled = h.slice([0, eos, 0], [B, eos + 1, D]).reshape([1, D]);
  const t = pooled.matmul(W.mx("text_projection.weight").transpose([1, 0])).toF32();
  let dot = 0;
  for (let i = 0; i < t.length; i++) dot += img[i] * t[i];
  return { prompt: p, score: dot / (imgNorm * norm(t)) };
});

scored.sort((a, b) => b.score - a.score);
console.log(`${file}:`);
for (const { prompt, score } of scored) {
  console.log(`  ${score.toFixed(4)}  ${prompt}`);
}
