# Findings: a TypeScript MLX inference stack over `mlx-c`

A feasibility study, carried all the way to a working artifact. The question we
started with was *"can we build a TypeScript SDK/API for MLX, the way Apple built
the Python one?"* The answer is **yes**, and this document records what it takes,
what was proven, and the non-obvious things found by actually doing it.

Every claim below is backed by a runnable file in this folder and validated
against either Apple's MLX Python or Huggingface's `tokenizers` — not asserted
from first principles.

---

## 1. Verdict
A Bun-FFI MLX SDK is not just possible; it is **~1,400 lines of hand-written
TypeScript** plus **~1,900 lines of generated FFI bindings**, and it runs real
**dense** (Qwen3-0.6B) and **MoE** (OLMoE-1B-7B, 64 experts) models — bf16 and
4-bit, single-file and sharded — at **~200–300 tok/s** with **bounded memory**,
producing output **token-for-token identical** to MLX's own Python stack
(`../scripts/validate-all.sh`: 53/53). It also **trains transformers from scratch** —
microGPT → nanoGPT (to the ~1.47 Shakespeare baseline) → real GPT-2-124M — the
optimizer driven by MLX `value_and_grad` over FFI, at ~parity with Python (§7d).

The work is *not* "write a TS SDK." It decomposes exactly like Apple's Python
SDK, which is what makes it tractable:

| Apple's Python SDK | This TS stack | Effort |
|---|---|---|
| `mlx.core` — ~15.5k LOC nanobind C++ over the C++ core | `../src/ffi/generated.ts` — FFI over **`mlx-c`** (Apple's C API) | mechanical codegen |
| `mlx.nn` / `optimizers` — ~6.6k LOC **pure Python** | `../src/nn/nn.ts` — pure TypeScript | direct port |

The native binding layer is **generated from headers**, not hand-written. The
framework layer is ordinary TypeScript. Neither requires touching C++.

---

## 2. The pivotal discovery
The MLX core in this repository is **pure C++** (`namespace mlx::core`, `class
array`, `StreamOrDevice` default args) with **no C ABI** — `MLX_API` is only a
visibility macro. If that were the whole story, the real work would be designing
a stable C ABI over the C++ core, and TypeScript would be the easy part.

But the README links it: **[`ml-explore/mlx-c`](https://github.com/ml-explore/mlx-c)**
is Apple's *officially maintained* C API, code-generated from a spec. It is
already installed via Homebrew here (`libmlxc.dylib` 0.6.0 over `libmlx` 0.31.2).
That single fact collapses the project from "design a C ABI" to "wrap an existing,
maintained one."

### The one ABI fact everything rests on

Every mlx-c handle is:

```c
typedef struct mlx_array_ { void* ctx; } mlx_array;
```

A struct containing a single pointer. On Apple-silicon ARM64, such a struct is
passed and returned **in a register, identically to a bare pointer**. So Bun
FFI — which has no by-value-struct support — can model **every** handle as
`ptr`. This is the unlock. It also generalizes: any opaque `mlx_*` type (arrays,
streams, maps, closures, devices) is one pointer, so the codegen treats unknown
`mlx_*` types as `ptr` for free.

---

## 3. What was proven, in order
Each milestone is a runnable file validated against a reference.

| # | Capability | File(s) | Validated against | Result |
|---|---|---|---|---|
| 1 | FFI → mlx-c → Metal; **autograd through a JS closure** | (initial PoC) | — | `value_and_grad` of a JS function returns correct gradient |
| 2 | Full Qwen3 decoder block (fp32) | `../validation/block.ts` / `../reference/reference.py` | MLX Python | `sum=0.005793`, `sum_sq=0.162600` exact |
| 3 | **Op codegen** from headers | `../tools/codegen.ts` → `../src/ffi/generated.ts` | rebuild block | 242 wrappers; `../validation/block-gen.ts` reproduces #2 |
| 4 | KV-cache autoregressive decode | `../validation/model-gen.ts` / `../reference/reference-decode.py` | MLX Python | identical token ids |
| 5 | safetensors loading | `../src/io/loader.ts`, `../validation/model-load.ts` | MLX + real file | round-trip exact; **881 tensors** from a real 27B gemma |
| 6 | 4-bit `quantized_matmul` | `../validation/model-quant.ts` / `../reference/reference-quant.py` | MLX Python | identical token ids |
| 7 | Byte-level BPE **tokenizer** | `../src/text/tokenizer.ts` / `../tests/tok-test.ts` | HF `tokenizers` | **11/11** encode+decode cases |
| 8 | **Real Qwen3-0.6B** (bf16), config-driven | `../src/models/qwen.ts` / `../reference/reference-qwen.py` | MLX Python | coherent text, exact ids, ~190 tok/s |
| 9 | **Real Qwen3-0.6B-4bit** over `nn.Module` | `../src/models/qwen-nn.ts` / `../reference/reference-qwen-q4.py` | MLX Python | coherent text, exact ids, ~264 tok/s |
| 10 | Sampling, batching, sliding window, bounded memory | `../tests/validate-prod.ts` | self/measured | reproducible sampling; B=2; +23 MB / 200 tok |
| 11 | **Real MoE model** (OLMoE-1B-7B, 64 experts top-8, 4-bit) | `../src/models/olmoe.ts` / `../reference/reference-olmoe.py` | MLX Python | coherent text, exact ids, ~209 tok/s |
| 12 | **Sharded** multi-file checkpoint load | `../src/io/loader.ts` (`shardedWeights`) | MLX Python | identical ids, RSS ≈ model size |
| 13 | **HF chat template** + end-to-end chat | `../src/text/chat-template.ts` / `../examples/chat.ts` | Python `jinja2` | 4/4 render parity; real assistant replies |
| 14 | **Training** — `value_and_grad` over a multi-param JS closure + SGD | `../validation/spike-train.ts` / `../reference/reference-train.py` | MLX Python | loss falls 0.237→0.009, final W bit-identical |
| 15 | **LoRA fine-tune** of real 4-bit Qwen3 — Adam + cross_entropy, frozen base | `../training/lora-train.ts` / `../reference/reference-lora.py` | MLX Python | loss falls 3.16→0.0007; tracks MLX to float tolerance |

All fifteen are re-checked together by `../scripts/validate-all.sh` — **53/53 green**
(the fifteen above plus the codegen, async-overlap, public-`stream()`, gather_qmm
op, per-op binding-parity (`bun test tests/`), and cross-runtime checks). The
count tracks which model files you have fetched; checks whose weights are absent
are skipped rather than failed.

The strongest checks are the **discrete** ones (#4, #6, #8, #9): greedy token ids
must match MLX exactly, so any drift anywhere — cache concat, RoPE offset, mask
selection, quantization layout — would flip a token and desync. They don't.

Sample real output (`../src/models/qwen-nn.ts`, 4-bit):

```
"The capital of France is" -> " Paris, and the capital of Italy is Rome.
 The capital of France is also the capital of Italy..."
```

---

## 4. The codegen
`../tools/codegen.ts` parses the mlx-c headers and emits `../src/ffi/generated.ts` — the FFI symbol
table plus typed wrappers. Final coverage:

```
parsed   491 decls across 12 headers
symbols  472 FFI entries
wrappers 242 typed op wrappers (from ops.h, fast.h)
skipped  47:
   24  non-standard (multi/zero output)
   17  unsupported param type '…'
    4  unwrappable param shape
```

It encodes the ABI knowledge as transformation rules: map each C type to an FFI
type, auto-supply the trailing `mlx_stream`, collapse `const int* x, size_t
x_num` pairs into one `number[]`, expose `/* may be null */` arrays as
`Arr | null`, and pack the by-value `mlx_optional_float/int` structs into a
`u64`. The 47 skips are all exotic and named — float16-by-value, function
pointers, and the metal/cuda **kernel-builder** API (opaque builders) — never a
standard tensor op. **No silent truncation.**

The decisive property: across nine milestones, expanding capability almost never
required new binding code. Adding a header to the list (and one rule — "opaque
`mlx_*` → `ptr`") carried entire new surfaces. Quantized matmul, dequantize,
sampling, memory introspection, the safetensors/map API — all generated. Symbol
count grew `322 → 339 → 412 → 440 → 472` purely from config; the 242 wrappers never
needed editing.

---

## 5. Performance
On this machine (Apple Silicon, `Darwin 25.5.0`, Bun 1.3.14):

| Model | tok/s |
|---|---|
| Qwen3-0.6B bf16 (`../src/models/qwen.ts`) | ~190 |
| Qwen3-0.6B 4-bit (`../src/models/qwen-nn.ts`) | ~260–300 |

As predicted at the outset, the JS runtime is not the bottleneck — compute lives
in Metal kernels. The FFI boundary is crossed thousands of times per token and it
doesn't matter.

---

## 6. Engineering findings (the non-obvious part)
These are the things you only learn by building it — the real value of the study.

1. **Empty handles are `null`, not `0`.** `mlx_array_new()` returns a handle with
   `ctx == NULL`, which Bun surfaces as `null`. Every wrapper must normalize
   `?? 0`. You hit this on line one.

2. **Out-params overwrite the handle slot.** Every op is `int fn(mlx_array* res,
   …)`; you must read the *new* handle back from the buffer after the call, not
   reuse the input variable. This is the single most pervasive pattern.

3. **By-value structs via integer-register packing.** `mlx_optional_float
   {float; bool}` is an 8-byte non-HFA struct → one GP register → modeled as a
   `u64` with the float bits low and `has_value` in byte 4. Same trick lets
   `rope`'s base and quantization's `group_size`/`bits` pass without Bun
   supporting structs at all.

4. **The safetensors `Load` primitive is CPU-only** (`[Load::eval_gpu] Not
   implemented`). Load on a **CPU stream**; the resulting concrete arrays then
   feed the GPU graph normally. A genuine MLX-ism, not an FFI issue.

5. **`FinalizationRegistry` does not bound memory in a sync loop.** FR only fires
   after a GC, which never happens inside a tight synchronous decode loop, so
   handles pile up. Measured over 200 tokens: **FR alone +3034 MB**; an explicit
   `tidy()` arena (free everything in scope but the returned token + KV cache)
   **+23 MB**. This is the most important correctness lesson in the whole study.
   FR is the right backstop; it is not the mechanism.

6. **Mask modes are strings.** `mask_mode = "causal"` for prefill, `""` for
   single-token decode (attends to all of history). Per-step RoPE uses
   `offset = position`.

7. **Reductions/gather need the `_axis` variants** (`argmax_axis`, `take_axis`,
   `concatenate_axis`); the bare versions flatten.

8. **mlx-c handles are refcounted**, so freeing a JS-side handle that MLX still
   needs internally (e.g. a cache tensor consumed by a not-yet-evaluated concat)
   is safe — MLX's primitive holds its own reference until eval. This is what
   makes `tidy()` + eager cache-freeing correct.

9. **Load-time memory — `active` over-counts; physical RSS is ~model size.**
   MoE **stacks** the per-expert tensors into `[E,…]` copies for `gather_qmm`,
   and the safetensors map keeps the originals referenced, so MLX *active* memory
   showed **7.2 GB for a 3.6 GB OLMoE (~2×)**. But that figure counts the
   whole-file **mmap** (file-backed, evictable) on top of the stacked copies —
   the measured **physical peak RSS was ~3.8–4.1 GB**, i.e. ≈ model size. The
   originals are mmap views the OS pages out; the only genuinely-resident copy is
   the stacked experts, which *is* the model. So there is no real 2× physical
   blow-up. Three things make this clean and controllable:
   - `freeMap` (free per-expert source views after stacking + the map after
     construction) — drops *active* back to **3.6 GB**, so the metric reflects
     reality. Dense models never needed it (no stacking; 320 MB → 339 MB).
   - `resetPeakMemory()` after load — separates the one-time load high-water from
     the **steady-state peak, which is ~model + KV (3.9 GB)**: what matters for a
     long-running server.
   - `setMemoryLimit(mb)` — caps MLX's retained cache and physical RSS (OLMoE:
     RSS 4.1 → 3.8 GB, cache 3.7 GB → 5 MB) with identical output; for a model
     that exceeds RAM, MLX spills against this limit (slower, but runs).

   The irreducible floor is the model weights in stacked layout. The decode loop
   itself is bounded (finding 5).

10. **Sharded checkpoints: mmap all shards (evictable), don't stream-materialize.**
    Large MoE are always split across files (`model-0000N-of-…` + an index).
    `shardedWeights` (in `../src/io/loader.ts`) mmaps each shard once and returns views;
    the OS pages out unused regions, so resident memory stays ~the working set.
    Measured on a 2-shard OLMoE: identical tokens, **RSS 3.90 GB** — the same as
    (slightly under) the single-file 4.11 GB. A *streaming* variant that copies
    each tensor out of its shard so the shard can be freed early was implemented
    and **measured worse** (7.5 GB): materializing loses the mmap's evictability.
    So whole-shard mmap + OS eviction is already resident-optimal; the real value
    of the sharded loader is simply *enabling* multi-file large-MoE checkpoints,
    not reducing memory below the model-size floor.

---

## 6.6. The memory model has a second half: ownership transfer
`tidy()` (§6.5) frees everything created in its scope except what is returned.
That is the right default and it is what makes a decode loop bounded — but it is
only half a memory model, and the missing half stayed missing long enough to
make the exported optimizer unusable.

Anything whose lifetime spans *more* than one scope does not fit: an optimizer's
first/second-moment buffers are created during `update()` and must survive to the
next step. Called inside a `tidy()`, they were freed as scope-local
intermediates, and the following step read freed handles. The symptom was
`expected a non-empty mlx_array` on step 1.

This had been silently worked around rather than fixed, which is the interesting
part. `../training/sft.ts` and `../validation/spike-nanogpt.ts` both hand-roll AdamW with module-level
state instead of using `../src/nn/optim.ts`, and `../training/lora-train.ts` — the only user of the
class — avoids `tidy()` altogether. Three files had independently routed around
the same defect without anyone naming it.

The fix is an explicit ownership transfer, `escape()`:

```ts
const oldM = this.m[i], oldV = this.v[i];
this.m[i] = escape(mi); this.v[i] = escape(vi);   // out of the caller's arena
oldM?.free(); oldV?.free();                        // we own them now, so we free them
```

`escape()` removes arrays from the enclosing arena and hands ownership to the
caller, who becomes responsible for freeing them. Measured: 500 optimizer steps
inside `tidy()` grow active memory by 0.03 MB.

The general lesson is that an arena needs an escape hatch as much as it needs a
scope, and that a workaround appearing independently in three places is evidence
of a defect rather than a style choice.

## 7. What a production `@mlx-ts/lm` still needs
The list has shrunk a lot — most of the original items are now built and
validated (§3, §6, §7b). None of what remains is a feasibility risk; it is
ergonomics, packaging, and breadth.

**Done since this list was first written:** tokenizer (11/11 vs HF, and it
generalized unchanged from Qwen BPE to OLMoE GPT-NeoX BPE), **HF chat template**
(`@huggingface/jinja`, 4/4 vs Python `jinja2`, real assistant replies via
`../examples/chat.ts`), KV-cache decode, temp/top-p sampling, equal-length batching, **MoE**
+ **sharded multi-file loading**, deterministic memory management (`tidy` /
`freeMap` / memory limits), and the **async-eval overlap mechanism** (proven in
§7b — works, composes with `tidy`, and is simply not a lever at 0.6B because
decode is compute-bound).

**Genuinely remaining:**

- ~~**Public streaming / disposable API.**~~ **Done** (`../src/text/lm.ts`). An async-generator
  `streamTokens()` / `streamText()` drives the decode loop over a model-agnostic
  `Decoder` interface; a `try/finally` frees the whole KV cache on completion, on
  early `break`, and on throw — so callers never call `tidy()` or free a handle.
  `MX` is now `Disposable` (`using a = x.add(y)`), and the async generators carry
  `[Symbol.asyncDispose]`. Greedy output is token-for-token identical to the
  proven `generate()` (`../tests/stream-test.ts`, in `../scripts/validate-all.sh`); memory is flat
  across repeated early-broken streams (338 MB steady state, no leak). The
  `Decoder` interface (`numLayers`, `eos`, `logitsLastMX`) is proven across both
  families: **Qwen3** (dense) and **OLMoE** (MoE) generate through the identical
  `streamTokens`/`generate` path, each token-exact vs MLX Python.
- **Ragged-batch padding masks.** Batching is proven at equal length; ragged
  prompts need an additive mask (`sdpa`'s `mask_arr` already accepts one) plus
  left-padding/position bookkeeping.
- **Training is demonstrated end to end with ergonomic APIs** (milestones 14–15):
  the keystone (SGD over a multi-param JS closure, bit-identical to MLX) *and* the
  mechanical tail — **Adam** (`../src/nn/optim.ts`), **cross_entropy** (`../src/nn/loss.ts`), **pytree
  utilities** (`../src/core/pytree.ts`), **`Module.parameters()`** (`../src/nn/nn.ts`), and a tree-based
  **`valueAndGrad`** (`../src/nn/autograd.ts`). The real **LoRA fine-tune** of the 4-bit Qwen3
  (`../training/lora-train.ts`: frozen quantized base, rank-8 adapters on q/v) now reads like
  mlx.nn — parameters come from `Module.parameters()`, gradients come back as a
  matching tree, and `Adam.update(params, grads)` flattens/unflattens internally;
  no hand-threaded parameter indices. Loss 3.16→0.0007. Honest caveat: training
  over a 4-bit base is **not bit-reproducible** across implementations — bf16
  rounding accumulates, and near a training instability it amplifies into visibly
  different trajectories; with a stable LR the curves track to float tolerance and
  converge identically (validation criterion: identical start + both converge).
  The former "what's left" breadth — **AdamW, cosine LR + warmup, gradient
  clipping, dropout, and multi-example minibatches** — is now demonstrated *from
  scratch* in §7d (microGPT → nanoGPT → a real GPT-2-architecture model trained to
  nanoGPT's Shakespeare baseline), including the backward-memory regime (activations
  held for backward, so per-step `tidy()` runs *after* the step). Standard minibatch
  training needs no `vmap`;
  only per-sample-gradient methods do — and `vmap` itself turned out to be
  **recoverable over FFI** (see below), so even that is no longer a hard gap.
- **Breadth.** More architectures (dense + MoE families proven; each new one is
  key-mapping), more quant formats (AWQ/GPTQ, other bit-widths), more sampling
  (top-k, repetition penalties, min-p), and **bundling a prebuilt `libmlxc`** —
  the only native artifact to ship, and still blocked: the `prebuilds/` build
  disagrees with MLX-Python numerically (§7e). The package itself ships.

The one piece genuinely outside MLX — the **tokenizer** — is done and validated.

---

## 7b. Spikes: de-risking the kill-risk unknowns
Before committing to a production `@mlx-ts/lm`, we spiked the unknowns that could
*kill* the project or limit it to a partial (correctness-only) toy. Both came
back green. (`../validation/spike-throughput.ts`, `../benchmarks/qwen_q4_throughput_bench.py`, `../validation/spike-moe.ts`.)

**Spike A — serving viability (is JS/FFI overhead a wall?).** Measured tok/s on
the same 4-bit Qwen3-0.6B, 128 tokens:

| path | tok/s |
|---|---|
| TS, sync eval | 342 |
| TS, async-eval overlap | 347 |
| **Python (MLX, same model)** | **362** |

TS runs at **~95% of Apple's own Python stack**. The conclusion: we are
**compute-bound, not FFI-bound** — JS op-construction overhead is ~5%, not a
wall. mlx-ts is a serving library, not a correctness toy.

The async-eval *overlap* gave only ~1.01× — but that is because at this scale
we're already GPU-bound and autoregressive decode is an inherently serial
dependency chain (each step reads the prior step's KV cache), so there is
nothing to hide. Python barely benefits from `async_eval` here either. Overlap
is not a lever at 0.6B; the point of the spike was to confirm the *ceiling*
isn't the FFI, and it isn't.

**Spike B — lifetime × async composition (the keystone).** The async run fed the
sampled token back as a *device array* (no host round-trip) and ran `tidy()` +
eager cache-freeing while async evals were in flight. Result: **identical tokens
and bounded memory (385 MB)**. So the trio *streaming + memory management +
async eval* composes correctly — the refcount safety of mlx-c handles is what
makes freeing-before-eval safe. The one design problem flagged as highest-risk
is retired.

**Spike C — MoE (is `gather_qmm` callable over FFI?).** The expert-dispatch op
(`gather_qmm` with `lhs_indices`/`rhs_indices` routing + optional group_size/bits)
reproduced Python's result exactly (`0.409543`) over Bun FFI (`../validation/spike-moe.ts`).
Then the **full MoE layer** — router → softmax → top-K (`argpartition`) →
weight renormalize → quantized expert dispatch → weighted combine — was built as
an `nn.MoE` module (`../src/nn/nn.ts`) and matched the Python reference **exactly**
(`sum=9.353439`, `sumsq=11773.206`; `../validation/spike-moe-layer.ts`).

Finally, a **real MoE model end-to-end**: `nn.MoE` wired into a decoder loads
**OLMoE-1B-7B** (64 experts, top-8, 4-bit, ~3.6 GB), generating coherent text —
`"The capital of France is"` → `" Paris.\n\nThe Louvre is a world-famous art
museum in Paris.\n\nThe Eiffel Tower is a famous iron lattice tower..."` at
~209 tok/s — and **token-for-token identical** to MLX Python (`../src/models/olmoe.ts` /
`../reference/reference-olmoe.py`). This exercised the real-model quirks: experts stored
*individually* (stacked into `[E,out,in]` at load via `mlx_stack`), a *quantized
router*, full-projection q/k-norm (vs Qwen3's per-head), untied lm_head, and
`norm_topk_prob=false`. The tokenizer (`../src/text/tokenizer.ts`) also handled OLMoE's
GPT-NeoX BPE unchanged. MoE is fully proven — op, layer, and real model.

**Spike D — `vmap` recovered over FFI (the last asterisk).** `mlx-c` exposes no
public `vmap`, only the internal `mlx_detail_vmap_trace` / `mlx_detail_vmap_replace`
primitives its own `vmap` composes from. `../spikes/spike-vmap.ts` `dlopen`s those two and
replicates the composition (trace the function once, then replace with the real
batched inputs) — the per-op batching rules live in the C++ core and run inside
`vmap_replace`, so we only orchestrate; the closure plumbing is the same
`mlx_closure_new_func` + `JSCallback` trick as `../src/nn/autograd.ts`. Validated three ways:
single-input `vmap` (sum-of-squares), a **shared (non-mapped) input** via the
`-1` axis sentinel (`A @ x`), and **per-sample gradients** `vmap(grad(f))` — `grad
of sum(x²)` per example equals `2x` exactly. So the one capability previously
called a genuine gap is reachable; per-sample-gradient methods (DP-SGD, etc.) are
no longer blocked. Caveat: these are `mlx_detail_*` (internal, version-unstable),
so it's a capability spike, not a committed public API.

**Net:** none of the suspected unknowns kill or even meaningfully partial the
project. The two that *could* have — "too slow to serve" and "MoE impossible
over FFI" — both resolved positive. Remaining caveats are scope, not risk:
throughput was measured at 0.6B/4-bit only (larger/batched serving may shift the
JS:GPU ratio, which is exactly where the now-proven overlap mechanism would
start to pay); MoE is proven at the op level but not yet wired into a full model;
and Node/Deno FFI parity, once deferred as v1 scope, is now done — see §7e.

## 7g. Custom Metal kernels, written in TypeScript

The kernel-builder API was the one surface `../tools/codegen.ts` skips — opaque
builder structs, no single-output shape to wrap — and §7e named it as the one
place a native shim might genuinely be needed. It is not.

The symbols were in the FFI table the whole time; only the ergonomic wrappers
were missing. `../src/ffi/kernel.ts` hand-writes those (~60 lines, the way
`../validation/mlx.ts` bootstrapped everything before codegen existed) and
`../examples/metal-kernel.ts` runs
**Apple's own fused-LSTM kernel, source verbatim** from
[mlx-examples/musicgen](https://github.com/ml-explore/mlx-examples/tree/main/musicgen). It matches MLX Python to every printed
digit.

Two things cost a round each, and both are the kind of detail no amount of
reading the header would have surfaced:

- **Scalars must be 0-d arrays.** Passing `hidden_size` as a `[1]`-shaped float
  array makes MLX bind it as a *device buffer*, and the Metal compiler rejects
  the shader with `incompatible pointer to integer conversion assigning to
  'uint' from 'const constant float *'`. A 0-d array becomes a scalar kernel
  parameter instead. Apple's Python passes plain ints, which MLX converts for
  you; over the C API you do it yourself.
- **`mlx_array_new_int` is the way to build one.** `fromI32(data, [])` crashes
  inside `mlx_array_new_data`, because `ptr()` on a zero-length shape array is
  null.

That the failure was a *Metal shader compile error* rather than an FFI error was
the good news buried in it: the name, input/output names, source and header had
all crossed the boundary correctly, and MLX had got as far as compiling the
shader.

The consequence is larger than one kernel. TypeScript can now author and
dispatch arbitrary Metal — so the stack is not limited to ops someone else
compiled, and MusicGen's EnCodec decoder (which needs exactly this LSTM, because
MLX has no fused one) is reachable. Conv1d and, importantly, `convTranspose1d`
are already generated wrappers, so the rest of that decoder is ordinary work.

## 7c. Audio: speech-to-text (done) and text-to-speech (de-risked)
A second modality, carried to the same bar. The reusable insight: for a fixed,
small `n_fft`, **the FFT is a matmul** — no FFT binding needed. The forward rfft
(audio → spectrum) is `frames @ cos / @ sin`; the inverse rfft (spectrum → audio)
is the same with the inverse-DFT basis (Hermitian 2× weighting). Both run on ops
we already have, validated to ~1e-6/1e-7 against numpy / mlx-audio.

**Speech-to-text — done, validated.**
- `../src/audio/mel.ts`: ffmpeg-decode → 16 kHz mono → Whisper log-Mel (rfft-as-matmul + the
  shipped librosa filterbank). Matches numpy FFT to ~1e-6.
- `../src/models/whisper.ts`: the full Whisper architecture over mlx-c — Conv1d stem,
  bidirectional encoder, causal decoder with **cross-attention** + a KV cache
  (self-attn grows, cross-attn k/v computed once), tied-embedding logits. Greedy
  transcription is **token-for-token identical to `mlx_whisper`**
  (`../tests/whisper-transcribe-test.ts`), encoder/decoder parity separately checked
  (`../tests/whisper-test.ts`). Special-token ids derived from `n_vocab` (handles v3's 100
  languages), with **language auto-detection** and a **sliding 25 s window** for
  unbounded dictation. Runs `whisper-large-v3-turbo` (128-mel); Danish and Swedish
  verified end to end (macOS `say` clips). Served at `/v1/audio/transcriptions`.
  The only non-MLX step is ffmpeg decoding — preprocessing, not the model.

**Text-to-speech — de-risked, then scoped out.** A neural vocoder's one *novel*
unknown is audio synthesis (spectrum → samples). `../spikes/spike-istft.ts` implements the
iSTFT in mlx-c/TS (inverse rfft matmul + windowed overlap-add) and matches
mlx-audio's `istft` to **1.2e-7**. So TTS feasibility is **proven at the kill-risk
level**. The remainder of a Kokoro-class port — weight-norm conv, AdaIN,
instance-norm, snake, transposed conv, and the acoustic transformer/predictors —
are conv/norm/transformer variants this project has **already** shown portable;
porting them is labor, not new evidence. And **grapheme→phoneme (G2P) is not an
MLX computation in any TTS** (it is rules + lexicon, i.e. `espeak-ng`), so a full
talking pipeline would pull in a non-MLX dependency analogous to ffmpeg. That is a
*product* decision, deliberately out of scope for a feasibility study: the study's
TTS question — "does the audio-synthesis math run in mlx-c/TS?" — is answered yes.

## 7d. From-scratch training: microGPT → nanoGPT → GPT-2-124M
The training above (§7, LoRA over a frozen base) is *fine-tuning*. This is the
harder claim: **train a transformer from random init, end to end, in TypeScript
driving MLX** — the autograd being real `value_and_grad` over FFI, not a
hand-rolled engine. Three rungs, each validated against an MLX-Python mirror.

**microGPT (~4k params).** Karpathy's minimal GPT (1 block, 4 heads, char-level
names) — but where his version hand-rolls a `Value` autograd class, here the
autograd is real MLX. Trains end to end (forward + backprop + Adam + sampling),
loss 3.19 → 1.72, emits name-like strings. Identical init (shared via a flat f32
blob) + identical data order → **step-0 loss bit-exact** vs `../reference/reference-microgpt.py`
(3.1896 = 3.1896); both converge (not bit-reproducible past step 0 — the §6
gotcha — so the bar is "same start, both converge"). It re-confirmed gotcha #5
live: the 1000-step loop blew MLX's buffer limit until per-step `tidy()` + freeing
the prior params/moments bounded it.

**nanoGPT (multi-layer, the real recipe).** Scaled to nanoGPT's char-level
Shakespeare: N pre-LN blocks, mini-batched `[B,T]`, **AdamW + cosine LR + linear
warmup + global grad-norm clipping + dropout** — the actual production training
machinery, all just MLX ops from TS. At nanoGPT's exact `shakespeare-char` config
(6 layers / 6 heads / 384-d / 10.7M params, dropout 0.2) it reaches **best val
loss 1.4964 — matching nanoGPT's published ~1.47 baseline** — and writes coherent
Shakespeare (speaker turns, real words, real *Winter's Tale* character names). The
dropout-free path is **bit-exact end to end** vs `../reference/reference-nanogpt.py` (shared
init + mini-batches): step-0, final-train, and val all match to 4 decimals. Two
capabilities fell out: **device-side dropout** (`mx.dropout` via
`mlx_random_bernoulli`, seed-derived so the oracle reproduces the same masks) and
**best-checkpoint eval** (a 10.7M model memorizes 1 MB of text — textbook
overfitting — so *best* val is the honest metric, as nanoGPT reports). Honest gap
to the baseline: we apply residual dropout but not attention-weight dropout, since
MLX's *fused* SDPA exposes no dropout parameter; best-val lands on it regardless.

**GPT-2-124M (real OpenAI weights).** The capstone: load the actual
`openai-community/gpt2` weights and generate, **token-exact** vs an MLX-Python
mirror. Two pieces. (1) A **GPT-2 BPE encoder** — `../src/text/tokenizer.ts` was already
byte-level BPE; GPT-2 needed its r50k pretokenization (`GPT2_SPLIT`) and no NFC
normalization, now **8/8 token-exact** vs HF `tokenizers`. (2) GPT-2's architecture
exactly (`../src/models/gpt2.ts`): learned positional embeddings, LayerNorm-with-bias, fused QKV
(Conv1D weight `[in,out]` → `matmul` directly, no transpose), **`gelu_new`** (tanh
approx via `mlx_tanh`), tied `lm_head`, KV-cached greedy decode. Generation is
token-for-token identical to `../reference/reference-gpt2.py` on every prompt, at ~210–250
tok/s, with optional temperature / top-k / top-p / repetition-penalty sampling.

**Scope — what "reproduce GPT-2-124M" means.** Two senses: reproduce its *outputs*
(done, token-exact) and reproduce its *from-scratch training* on OpenWebText. The
latter is the *same recipe* the nanoGPT rung runs at 10.7M, only bigger — nothing
in the TS↔MLX bridge is missing. What stops a full run is **single-Mac wall-clock**
(~4 days on 8×A100 over a 40 GB corpus) and a data pipeline: a practical limit,
not a feasibility one.

**Training performance — at parity where it matters.** Both TS and Python run the
identical MLX Metal kernels, so they share a GPU floor; the only difference is
TS's host-side orchestration. Per-iteration, training-loop only:

| model | TS | MLX-Python | gap |
|---|---|---|---|
| 0.4M (toy) | 7.9 ms | sub-ms | host-dominated |
| 10.7M (real) | 257 ms | 251 ms | **~2.4%** |

The host tax is a roughly *fixed* ~6 ms/step (FFI dispatch + the `value_and_grad`
JS-callback + per-leaf optimizer ops). On a toy model the GPU does ~nothing so the
tax is the whole story; at real scale it's ~2.4% — effectively parity, the same as
GPT-2-124M *inference* (TS 248 vs Python 234 tok/s). Two optimizations keep it
small without harming scale: hoisting per-step scalar constants, and folding
Adam's bias-correction into the step size (PyTorch's efficient form, applied
identically to the oracle so parity holds). A flat-vector optimizer was
deliberately *not* done — it helps toy models but adds per-step concat/slice
bandwidth that pessimizes large (GPU-bound) ones.

Deep-dive notes: `MICROGPT.md`, `NANOGPT.md`, `GPT2.md`.

## 7e. The FFI layer is not Bun-specific
Bun-only was the single biggest limit on who could use this, so it was worth
testing rather than assuming. It turned out to be a thin constraint: the same
code now runs on **Bun, Deno and Node**, producing bit-identical output, with the
runtime difference confined to five primitives in `../src/ffi/`
(`open`/`ptr`/`view`/`cstring`/`callback`).

Two risks were expected and both evaporated:

- **The ABI trick ports.** `mlx_array` is `struct { void* ctx; }`, passed and
  returned in a register like a bare pointer. Declaring it `void *` works
  identically in `bun:ffi`, `Deno.dlopen` and koffi — no struct support needed.
- **Zero-copy readback exists everywhere**: `toArrayBuffer` (Bun),
  `UnsafePointerView.getArrayBuffer` (Deno), `koffi.view` (Node). Verified by
  *aliasing* — write through one view, read from another — not by timing, which
  cannot distinguish a fast copy from a genuine view. koffi's `decode()` is the
  copying path and is the wrong one.

The real cost was **handle representation**: Bun hands back numbers (with NULL as
`null`), Deno opaque PointerObjects, koffi BigInt addresses. Normalising every
pointer to a JS number at the boundary — safe because macOS user-space addresses
fit in 2^48 — kept `type Arr = number` intact, so `../src/core/mx.ts`, the 242 generated
wrappers and every model file are untouched by any of this.

Dispatch cost, 500k calls warmed, best of 3, result observed so the JIT cannot
elide it: **Bun ~12 ns, Deno ~3 ns, Node/koffi ~21 ns**. All are cheap beside an
MLX op, and end-to-end generation is equal across the three within noise — the
compute-bound finding of §5 holding up. One sharp edge: a 64-bit *return* costs
Deno ~52 ns, falling off V8's fast-call path, so hot accessors declare a 32-bit
return. First measurements were 4–8x higher and ranked Deno last; that was JIT
warmup, not dispatch, and is a caution about single-shot FFI microbenchmarks.

Node imposed two source constraints, since it runs `.ts` by stripping types only:
no `enum` (codegen emits a const object) and no parameter properties (expanded to
explicit fields). Both are what `isolatedModules` wants anyway.

**Unresolved:** the relocatable `prebuilds/` bundle is a *different MLX build*
from Homebrew's mlx-c and does not agree with it numerically — real Qwen3 (bf16
and 4-bit) and LoRA diverge from MLX-Python when the bundle is loaded, and agree
when Homebrew's is. **Resolved in §7f** by not building `libmlx` at all: `prebuilds/` is now
assembled from Apple's own `mlx-metal` binaries, and the whole suite passes
(53/53) forced onto the bundled path.

The sharper version of that finding is how it was nearly missed. The resolver
*preferred* the bundle over Homebrew, on the reasoning that a bundled copy makes
the package self-contained. So the default configuration silently produced
different numbers, and the suite only passed because `MLXTS_LIB` was being set by
hand — the parity checks were, for a while, testing a path that was not the
default. The order is now inverted: the validated build wins when present, the
bundle is the fallback for machines without Homebrew, and using it prints a
warning. `../scripts/validate-all.sh` also prints which dylib it resolved, so a parity
failure can never again be mistaken for a code regression.

## 7f. How Apple ships the native dependency (and what that means for `prebuilds/`)

`mlx-lm` never touches a binary itself. The dependency chain is:

```
mlx-lm                     pure Python
  └─ mlx                   0.7 MB wheel, one per CPython ABI (cp310…cp314)
      └─ mlx-metal         42–62 MB, py3-none-macosx_{14,15,26}_0_arm64
      └─ mlx-cuda-12 / mlx-cpu   (Linux, behind extras)
```

The heavy payload is a **separate package pulled in by an environment marker**
(`mlx-metal==0.32.1 ; platform_system == "Darwin"`), and it is `py3-none-*`:
Python-version independent, OS/arch specific. Unzipping the macOS 15 wheel:

| inside `mlx-metal` | size |
|---|---|
| `mlx/lib/mlx.metallib` | 135.5 MB |
| `mlx/lib/libmlx.dylib` | 21.8 MB |
| `mlx/lib/libjaccl.dylib` | 1.6 MB |

That is exactly the npm `optionalDependencies` pattern (esbuild, swc, rollup) in
Python's idiom — a small front package plus a platform-gated backend — so the
shape transfers directly:

```
@nielspeter/mlx-ts                    the JS/TS (96 KB)
  └─ optionalDependencies
       @nielspeter/mlx-ts-darwin-arm64   libmlxc + libmlx + metallib
```

**It also explains §7e's unresolved divergence.** Comparing Apple's binaries with
the ones in `prebuilds/`:

| artifact | Apple `mlx-metal` | our `prebuilds/` |
|---|---|---|
| `libmlx.dylib` | 21.8 MB | 15.5 MB |
| `libjaccl.dylib` | 1.6 MB | 0.6 MB |
| `mlx.metallib` | 135.5 MB | 157.7 MB |

Different builds, plainly. That is why the bundle disagrees with MLX-Python on
real Qwen3: the Python reference runs on Homebrew's `libmlx`, and `prebuilds/`
is something else compiled with different options.

The fix follows from the structure rather than from debugging: **do not build
`libmlx` at all.** Take `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` from
Apple's own `mlx-metal` wheel — the exact binaries the parity oracle runs
against — and compile only `libmlxc.dylib` (0.7 MB, a thin C shim) against them.
Then the bundled path is the *same* build the suite validates, not a lookalike.

One wrinkle: Apple ships three macOS variants (14, 15, 26), so the Metal target
is not one artifact — a single blob was never going to be portable.
`tools/fetch-prebuilds.sh` picks the newest wheel at or below the host's macOS
major version.

**This is done.** The script downloads the wheel, copies `libmlx` / `libjaccl` /
`mlx.metallib` out of it, adds an `@loader_path` rpath (Apple's `libmlx`
references `@rpath/libjaccl.dylib` but carries no `LC_RPATH` — in the wheel the
Python extension supplies it), takes `libmlxc` from Homebrew and repoints it at
the neighbouring `libmlx`. The bundled path then reproduces Apple's `mlx-lm`
token for token, and `../scripts/validate-all.sh` is **53/53 forced onto it**.

Two things worth keeping:

- **The verification nearly did not work.** The obvious check — the synthetic
  Qwen3 block fingerprint — *passed with the old, broken bundle*. The divergence
  only appeared on a real model. A smoke test that cannot fail is worse than no
  test, so the script verifies with real Qwen3 whenever the weights are present
  and says so loudly when it has to fall back.
- **`libmlxc` is still built against Homebrew's `libmlx`.** It works because both
  are the same mlx version and the C API is ABI-stable across it — verified, not
  assumed — but the rigorous version compiles mlx-c from source against the
  wheel's `libmlx`.

## 8. File map
**Runtime & codegen**
- `../tools/codegen.ts` → `../src/ffi/generated.ts` — header parser → FFI bindings (472 symbols, 242 wrappers)
- `../src/core/mx.ts` — `MX` array class, `FinalizationRegistry` + `tidy()`, ops, sampling,
  **`dropout`** (device-side Bernoulli), `async_eval`, `stack`, `copy`, memory introspection/limits
- `../src/nn/nn.ts` — `Module`, `Linear`, `QuantizedLinear`, `RMSNorm`, `Embedding`,
  `QuantizedEmbedding`, **`MoE`** (router + top-K + quantized expert dispatch)
- `../src/nn/optim.ts` — `Adam` (pytree-aware); `../src/nn/loss.ts` — `crossEntropy`
- `../src/core/pytree.ts` — `treeFlatten` / `treeUnflattenLike` / `treeMap`;
  `../src/nn/autograd.ts` — tree-based `valueAndGrad` (the ergonomic training layer)
- `../src/text/lm.ts` — public generation surface: `Decoder` interface + async-generator
  `streamTokens` / `streamText` / `generate` (auto KV-cache cleanup, no manual `tidy`)
- `../src/io/loader.ts` — safetensors loading; `singleFileWeights` / **`shardedWeights`**
  (multi-file, mmap-evictable) + `freeMap`
- `../src/text/tokenizer.ts` — pure-TS byte-level BPE (Qwen + **GPT-2 `GPT2_SPLIT`**, 8/8 vs HF)
- `../src/text/chat-template.ts` — HF chat template via `@huggingface/jinja`
- `../src/audio/mel.ts` — speech front-end: ffmpeg decode + log-Mel (rfft-as-matmul)
- `../src/models/whisper.ts` / `../src/text/whisper-tokenizer.ts` — Whisper STT (encoder + cross-attn decoder
  + KV cache, multilingual, auto language detect); tiktoken decode

**Models / demos**
- `../src/models/qwen.ts` — config-driven Qwen3-0.6B (bf16)
- `../src/models/qwen-nn.ts` — config-driven Qwen3-0.6B-4bit over `nn.Module` (CLI: sampling, window)
- `../examples/chat.ts` — end-to-end chat: message → template → tokenizer → model → reply
- `../examples/stream.ts` — streaming chat over the public `lm.streamText` API (tokens printed live)
- `../examples/server.ts` / `../examples/chat.html` — Bun.serve OpenAI-compatible server (chat / embeddings /
  audio transcriptions) + a chat web UI with live mic transcription
- `../training/lora-train.ts` — **LoRA fine-tune** of real 4-bit Qwen3 (Adam + cross_entropy)
- `../src/models/olmoe.ts` — config-driven **OLMoE-1B-7B MoE** (single-file or `MX_SHARDED`)
- `../src/models/gpt2.ts` — real OpenAI **GPT-2-124M** (BPE + `gelu_new` + tied head), token-exact; sampling flags
- `block*.ts`, `model-*.ts` — the staged PoCs (block, decode, safetensors, quant)
- `../tools/inspect-real.ts` — enumerate a real model file's tensors
- `../reference/split-olmoe.py` — split a single file into shards (to exercise the sharded loader)

**Spikes** — `../validation/spike-throughput.ts` (async overlap / serving viability),
`../benchmarks/qwen_q4_throughput_bench.py` (Python tok/s bar), `../validation/spike-moe.ts` (gather_qmm op),
`../validation/spike-moe-layer.ts` (full MoE layer), `../validation/spike-train.ts` (training: value_and_grad
+ SGD), `../spikes/spike-istft.ts` (iSTFT vocoder synthesis — TTS de-risk),
`../spikes/spike-vmap.ts` (`vmap` recovered from the detail trace/replace primitives),
`../validation/spike-microgpt.ts` (microGPT from scratch), `../validation/spike-nanogpt.ts` (multi-layer GPT
from scratch — trains to nanoGPT's Shakespeare baseline). See §7d.

**References & validation**
- `reference*.py` — MLX Python mirrors for every milestone
- `../reference/tok-reference.py` / `../tests/tok-test.ts` — tokenizer ground truth
- `../tests/validate-prod.ts` — sampling / batching / bounded-memory checks
- **`../scripts/validate-all.sh`** — the full suite: every TS path vs its reference (53/53)

---

## 9. Environment
- Apple Silicon, macOS (`Darwin 25.5.0`)
- Bun 1.3.14 (FFI host)
- `mlx-c` 0.6.0, `libmlx` (MLX) 0.31.2 — both prebuilt via Homebrew
- No compilation step; `dlopen("/opt/homebrew/lib/libmlxc.dylib")` and go
- Models: `Qwen/Qwen3-0.6B` (bf16), `mlx-community/Qwen3-0.6B-4bit`, tokenizer from `Qwen/Qwen3-0.6B`

---

## 10. Bottom line
The original architecture instinct — build a runtime-agnostic TS layer over a
stable C ABI, let Bun be the nice frontend — was correct, and **cheaper than
expected**, because the C ABI already exists and is maintained by Apple, the
binding is generatable, and the framework layer is a port. From "is this
feasible?" to "a real quantized LLM generating text from TypeScript at 260 tok/s,
matching MLX exactly" is the span of this folder.
