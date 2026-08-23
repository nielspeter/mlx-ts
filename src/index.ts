// Public API for mlx-ts. Everything here runs unchanged on Bun, Deno and Node:
// the runtime difference is confined to src/ffi/, which picks a backend at
// import time. Apple Silicon + Metal only.
//
//   import { MX, tidy, Tokenizer, generate } from "mlx-ts";

// --- arrays and memory ---------------------------------------------------
// `tidy()` is not optional on a hot path: a FinalizationRegistry only runs
// after a GC, which never happens inside a tight synchronous decode loop.
export {
  MX, tidy, fromF32, fromI32, fromU32, scalar, stack, evalAll, asyncEval, seed,
  saveSafetensors, dropout, applyRepetitionPenalty, sample, escape,
  activeMemoryMB, peakMemoryMB, cacheMemoryMB, clearCache, resetPeakMemory,
  setMemoryLimit, setWiredLimit,
} from "./core/mx.ts";
export { treeFlatten, treeUnflattenLike, treeMap, type Tree } from "./core/pytree.ts";

// --- modules, optimizers, losses ----------------------------------------
export {
  Module,
  LoraDelta, RMSNorm, Linear, QuantizedLinear, Embedding, QuantizedEmbedding,
  MoE, type Experts,
} from "./nn/nn.ts";
export { Adam } from "./nn/optim.ts";
// The training entry point: differentiate a scalar loss w.r.t. a pytree of params.
export { valueAndGrad } from "./nn/autograd.ts";
export { crossEntropy, maskedCrossEntropy } from "./nn/loss.ts";

// --- text: tokenizing, chat templates, generation ------------------------
export { Tokenizer, GPT2_SPLIT } from "./text/tokenizer.ts";
export { ChatTemplate, type Message } from "./text/chat-template.ts";
export { generate, streamTokens, streamText, type Decoder, type GenOptions, type KV } from "./text/lm.ts";

// --- audio: the Whisper front-end (ffmpeg decode -> log-Mel) ----------------
export { SR, decodeAudio, padOrTrim, loadMelFilters, logMel } from "./audio/mel.ts";

// --- weights -------------------------------------------------------------
export {
  loadSafetensors, singleFileWeights, shardedWeights, shapeOf, freeMap,
  get, entries, type Weights, type WeightMap,
} from "./io/loader.ts";

// --- models --------------------------------------------------------------
// Assembled models that are importable as modules. qwen.ts and gpt2.ts are
// absent because they are CLI scripts, not modules: they build a model with
// top-level await and generate on import. Run those with
// `bun src/models/<name>.ts "prompt"`.
export { Whisper, loadWhisper } from "./models/whisper.ts";
export { OLMoE } from "./models/olmoe.ts";
export { WhisperTokenizer, langToken } from "./text/whisper-tokenizer.ts";
// Qwen3 is the model class the rest of the repo is built around; generateBatch
// takes one as its first argument, so exporting the function without the class
// gave consumers a signature they could not satisfy.
export { Qwen3, generateBatch, stepTidy } from "./models/qwen-nn.ts";
// Namespaced: its `generate`/`forward` would collide with the ones above.
export * as nanogpt from "./models/nanogpt-model.ts";

// --- runtime / FFI -------------------------------------------------------
// `backend.name` tells you which runtime was detected; LIBMLXC is the resolved
// dylib path (override with MLXTS_LIB).
// The escape hatch: call an mlx-c entry point we do not wrap. `open` binds
// symbols, `ptr`/`view` cross the buffer boundary, `callback` makes a C function
// pointer. Pointers are plain numbers on every runtime.
export { backend, open, ptr, view, cstring, callback } from "./ffi/index.ts";
export type { Backend, CType, SymbolSpec, SymbolTable, Callback } from "./ffi/types.ts";
export { LIBMLXC } from "./ffi/native-lib.ts";
