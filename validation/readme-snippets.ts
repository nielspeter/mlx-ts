// The README's "What that looks like" snippets, kept compilable.
//
// README examples rot silently: the first draft of that section used
// load().template, streamText(prompt: string), ChatTemplate.apply(),
// transcribe(path) and MX.l2Normalize() — five APIs that do not exist. None of
// it was checkable, because prose is not compiled.
//
// This file mirrors those snippets so `bun run typecheck` fails when the API
// moves under them. It is not run; compiling is the whole point.
import {
  Adam, crossEntropy, decodeAudio, fromI32, load, loadMelFilters, loadWhisper,
  metalKernel, MusicGen, saveAudio, scalarI32, seed, streamText, tidy,
  valueAndGrad, WhisperTokenizer, Qwen3, Tokenizer, loadSafetensors, type MX, type Tree,
  StableDiffusion, savePng, ClipVisionEncoder, loadImage, fromF32,
} from "../src/index.ts";
import type { Weights } from "../src/io/loader.ts";

export async function chat() {
  const { model, tokenizer } = await load("mlx-community/Qwen3-0.6B-4bit");
  const ids = tokenizer.encode("The capital of France is");
  for await (const chunk of streamText(model, tokenizer, ids, { max: 64, temp: 0.7 }))
    process.stdout.write(chunk);
}

export async function whisper() {
  const model = await loadWhisper("models/config-turbo.json", "models/whisper-turbo.safetensors");
  const tok = await WhisperTokenizer.fromFile();
  const filters = await loadMelFilters("models/whisper-mel-filters-128.f32", 128);
  const ids = model.transcribe(await decodeAudio("interview.flac"), filters);
  console.log(tok.decode(ids).trim());
}

export async function music() {
  const model = await MusicGen.fromPretrained();
  seed(1234);
  const audio = model.generate("trance", { maxSteps: 500 });
  await saveAudio("out.wav", audio.toF32(), model.samplingRate);
}

export async function embed(cfg: any) {
  const model = new Qwen3(cfg, loadSafetensors("models/model-q4.safetensors"));
  const tokenizer = await Tokenizer.fromFile("models/tokenizer.json");
  const ids = tokenizer.encode("a passage to index");
  const vec = tidy(() => model.embeddingMX(fromI32(Int32Array.from(ids), [1, ids.length]), 1, ids.length));
  return Array.from(vec.toF32());
}

export function train(params: Tree, X: MX, Y: MX, STEPS: number) {
  const forward = (p: Tree, x: MX) => { const { w, b } = p as { w: MX; b: MX }; return x.matmul(w).add(b); };
  const lossFn = (p: Tree, x: MX, y: MX) => crossEntropy(forward(p, x), y);
  const step = valueAndGrad(params, lossFn);
  const opt = new Adam(0.1);
  for (let i = 0; i <= STEPS; i++) {
    const { next } = tidy(() => {
      const { loss, grads } = step(params, X, Y);
      return { loss, next: opt.update(params, grads) };
    });
    params = next;
  }
  return params;
}

export async function image() {
  const sd = await StableDiffusion.fromPretrained();
  const img = sd.generate("a photo of an astronaut riding a horse", {
    width: 384, height: 384, steps: 20, seed: 42,
  });
  await savePng("out.png", img.toF32(), 384, 384);
}

export async function zeroShot(vision: ClipVisionEncoder, W: Weights) {
  const px = await loadImage("photo.jpg", { size: 224 });
  return vision.embed(fromF32(px, [1, 224, 224, 3]), W.mx("visual_projection.weight"));
}

export function kernel(x: MX, hIn: MX, cell: MX, H: number, B: number, T: number) {
  const lstm = metalKernel({
    name: "lstm",
    inputNames: ["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
    outputNames: ["hidden_state", "cell_state"],
    source: `/* Metal */`,
  });
  return tidy(() => lstm.apply(
    [x, hIn, cell, scalarI32(H), scalarI32(0), scalarI32(T)],
    [{ shape: [B, H] }, { shape: [B, H] }],
    [B, B * H, 1],
    [256, 1, 1],
  ));
}
