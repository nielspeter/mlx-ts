// Public API for mlx-ts. Everything here runs unchanged on Bun, Deno and Node:
// the runtime difference is confined to src/ffi/, which picks a backend at
// import time. Apple Silicon + Metal only.
//
//   import { MX, tidy, Tokenizer, generate } from "mlx-ts";


// --- audio: the Whisper front-end (ffmpeg decode -> log-Mel) ----------------
export { decodeAudio, loadMelFilters, logMel, padOrTrim, playAudio, SR  } from "./audio/mel.ts";
export { saveAudio } from "./audio/wav.ts";
// --- arrays and memory ---------------------------------------------------
// `tidy()` is not optional on a hot path: a FinalizationRegistry only runs
// after a GC, which never happens inside a tight synchronous decode loop.
export {
  activeMemoryMB, applyRepetitionPenalty, asyncEval, cacheMemoryMB, clearCache, dropout, escape,evalAll, freeAll,fromF32, fromI32, fromU32, 
  MX, Owned, peakMemoryMB, 
  randomNormal,resetPeakMemory,sample, 
  saveSafetensors, scalar, seed,setCacheLimit,
  setMemoryLimit, setWiredLimit, stack, tidy, 
} from "./core/mx.ts";
export { type Tree, treeFlatten, treeMap, treeUnflattenLike } from "./core/pytree.ts";
// --- runtime / FFI -------------------------------------------------------
// `backend.name` tells you which runtime was detected; LIBMLXC is the resolved
// dylib path (override with MLXTS_LIB).
// The escape hatch: call an mlx-c entry point we do not wrap. `open` binds
// symbols, `ptr`/`view` cross the buffer boundary, `callback` makes a C function
// pointer. Pointers are plain numbers on every runtime.
export { backend, callback, cstring, open, ptr, view } from "./ffi/index.ts";
// Write your own Metal kernel when MLX has no fused op for what you need.
export { metalKernel, scalarI32 } from "./ffi/kernel.ts";
export { LIB_CANDIDATES, LIBMLXC } from "./ffi/native-lib.ts";
export type { Backend, Callback, CType, SymbolSpec, SymbolTable } from "./ffi/types.ts";
export { CLIP_MEAN, CLIP_STD, type LoadImageOptions, loadImage } from "./image/load.ts";
export { savePng } from "./image/png.ts";
export { cacheDir, type FetchOptions, hubFile, isCached } from "./io/hub.ts";
// --- weights -------------------------------------------------------------
export {entries, freeMap,
  get, 
  loadSafetensors, shapeOf, shardedWeights, singleFileWeights, type WeightMap,type Weights, 
} from "./io/loader.ts";
export { type ClipConfig, ClipTextEncoder } from "./models/clip.ts";
export { type ClipVisionConfig, ClipVisionEncoder } from "./models/clip-vision.ts";
// Stable Diffusion: prompt -> image. The pieces are exported too, so a caller
// can drive the loop themselves.
export { type DiffusionConfig, EulerSampler } from "./models/diffusion.ts";
export { type EncodecConfig, EncodecDecoder } from "./models/encodec.ts";
// Fetch-and-construct from a Hugging Face repo id — the step between `npm i`
// and a token.
export { type Loaded, load } from "./models/load.ts";
// Text -> music. T5 conditioning, a codebook LM, EnCodec back to a waveform.
export { type GenerateOptions, type LayerKV, MusicGen, type MusicGenConfig, MusicGenLM } from "./models/musicgen.ts";
// Namespaced: its `generate`/`forward` would collide with the ones above.
export * as nanogpt from "./models/nanogpt-model.ts";
export { OLMoE } from "./models/olmoe.ts";
// Qwen3 is the model class the rest of the repo is built around; generateBatch
// takes one as its first argument, so exporting the function without the class
// gave consumers a signature they could not satisfy.
export { generateBatch, Qwen3, stepTidy } from "./models/qwen-nn.ts";
// Qwen2 / Qwen2.5 at full precision — the backbone under Spark-TTS and many others.
export { Qwen2, type Qwen2Config } from "./models/qwen2.ts";
export { type ImageOptions, StableDiffusion } from "./models/stable-diffusion.ts";
export { type T5Config, T5Encoder } from "./models/t5.ts";
export { timestepEmbedding, Unet, type UnetConfig } from "./models/unet.ts";
export { upsampleNearest, type VaeConfig, VaeDecoder } from "./models/vae.ts";
// --- models --------------------------------------------------------------
// Assembled models that are importable as modules. qwen.ts and gpt2.ts are
// absent because they are CLI scripts, not modules: they build a model with
// top-level await and generate on import. Run those with
// `bun src/models/<name>.ts "prompt"`.
export { loadWhisper, Whisper } from "./models/whisper.ts";
// The training entry point: differentiate a scalar loss w.r.t. a pytree of params.
export { valueAndGrad } from "./nn/autograd.ts";
export { crossEntropy, maskedCrossEntropy } from "./nn/loss.ts";
// --- modules, optimizers, losses ----------------------------------------
export {Embedding, type Experts,
  GroupNorm,Linear, 
  LoraDelta, 
  Module,
  MoE, QuantizedEmbedding,QuantizedLinear, RMSNorm, 
} from "./nn/nn.ts";
export { Adam } from "./nn/optim.ts";
export { ChatTemplate, type Message } from "./text/chat-template.ts";
export { type ClipEncodeOptions, ClipTokenizer } from "./text/clip-tokenizer.ts";
export { type Decoder, type GenOptions, generate, type KV, streamText, streamTokens } from "./text/lm.ts";
// --- text: tokenizing, chat templates, generation ------------------------
export { GPT2_SPLIT, Tokenizer } from "./text/tokenizer.ts";
export { UnigramTokenizer } from "./text/unigram.ts";
export { langToken, WhisperTokenizer } from "./text/whisper-tokenizer.ts";
