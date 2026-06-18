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
(`validate-all.sh`: 15/15).

The work is *not* "write a TS SDK." It decomposes exactly like Apple's Python
SDK, which is what makes it tractable:

| Apple's Python SDK | This TS stack | Effort |
|---|---|---|
| `mlx.core` — ~15.5k LOC nanobind C++ over the C++ core | `generated.ts` — FFI over **`mlx-c`** (Apple's C API) | mechanical codegen |
| `mlx.nn` / `optimizers` — ~6.6k LOC **pure Python** | `nn.ts` — pure TypeScript | direct port |

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
| 2 | Full Qwen3 decoder block (fp32) | `block.ts` / `reference.py` | MLX Python | `sum=0.005793`, `sum_sq=0.162600` exact |
| 3 | **Op codegen** from headers | `codegen.ts` → `generated.ts` | rebuild block | 242 wrappers; `block-gen.ts` reproduces #2 |
| 4 | KV-cache autoregressive decode | `model-gen.ts` / `reference-decode.py` | MLX Python | identical token ids |
| 5 | safetensors loading | `loader.ts`, `model-load.ts` | MLX + real file | round-trip exact; **881 tensors** from a real 27B gemma |
| 6 | 4-bit `quantized_matmul` | `model-quant.ts` / `reference-quant.py` | MLX Python | identical token ids |
| 7 | Byte-level BPE **tokenizer** | `tokenizer.ts` / `tok-test.ts` | HF `tokenizers` | **11/11** encode+decode cases |
| 8 | **Real Qwen3-0.6B** (bf16), config-driven | `qwen.ts` / `reference-qwen.py` | MLX Python | coherent text, exact ids, ~190 tok/s |
| 9 | **Real Qwen3-0.6B-4bit** over `nn.Module` | `qwen-nn.ts` / `reference-qwen-q4.py` | MLX Python | coherent text, exact ids, ~264 tok/s |
| 10 | Sampling, batching, sliding window, bounded memory | `validate-prod.ts` | self/measured | reproducible sampling; B=2; +23 MB / 200 tok |
| 11 | **Real MoE model** (OLMoE-1B-7B, 64 experts top-8, 4-bit) | `olmoe.ts` / `reference-olmoe.py` | MLX Python | coherent text, exact ids, ~209 tok/s |
| 12 | **Sharded** multi-file checkpoint load | `loader.ts` (`shardedWeights`) | MLX Python | identical ids, RSS ≈ model size |
| 13 | **HF chat template** + end-to-end chat | `chat-template.ts` / `chat.ts` | Python `jinja2` | 4/4 render parity; real assistant replies |
| 14 | **Training** — `value_and_grad` over a multi-param JS closure + SGD | `spike-train.ts` / `reference-train.py` | MLX Python | loss falls 0.237→0.009, final W bit-identical |
| 15 | **LoRA fine-tune** of real 4-bit Qwen3 — Adam + cross_entropy, frozen base | `lora-train.ts` / `reference-lora.py` | MLX Python | loss falls 3.16→0.0007; tracks MLX to float tolerance |

All fifteen are re-checked together by `validate-all.sh` — **18/18 green** (the
fifteen above plus the codegen, async-overlap, and gather_qmm op checks).

The strongest checks are the **discrete** ones (#4, #6, #8, #9): greedy token ids
must match MLX exactly, so any drift anywhere — cache concat, RoPE offset, mask
selection, quantization layout — would flip a token and desync. They don't.

Sample real output (`qwen-nn.ts`, 4-bit):

```
"The capital of France is" -> " Paris, and the capital of Italy is Rome.
 The capital of France is also the capital of Italy..."
```

---

## 4. The codegen

`codegen.ts` parses the mlx-c headers and emits `generated.ts` — the FFI symbol
table plus typed wrappers. Final coverage:

```
parsed   446 decls across 10 headers
symbols  440 FFI entries
wrappers 242 typed op wrappers (from ops.h, fast.h)
skipped  34   (every one reported by name)
```

It encodes the ABI knowledge as transformation rules: map each C type to an FFI
type, auto-supply the trailing `mlx_stream`, collapse `const int* x, size_t
x_num` pairs into one `number[]`, expose `/* may be null */` arrays as
`Arr | null`, and pack the by-value `mlx_optional_float/int` structs into a
`u64`. The 34 skips are all exotic and named — float16-by-value, function
pointers, and the metal/cuda **kernel-builder** API (opaque builders) — never a
standard tensor op. **No silent truncation.**

The decisive property: across nine milestones, expanding capability almost never
required new binding code. Adding a header to the list (and one rule — "opaque
`mlx_*` → `ptr`") carried entire new surfaces. Quantized matmul, dequantize,
sampling, memory introspection, the safetensors/map API — all generated. Symbol
count grew `322 → 339 → 412 → 440` purely from config; the 242 wrappers never
needed editing.

---

## 5. Performance

On this machine (Apple Silicon, `Darwin 25.5.0`, Bun 1.3.14):

| Model | tok/s |
|---|---|
| Qwen3-0.6B bf16 (`qwen.ts`) | ~190 |
| Qwen3-0.6B 4-bit (`qwen-nn.ts`) | ~260–300 |

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
    `shardedWeights` (in `loader.ts`) mmaps each shard once and returns views;
    the OS pages out unused regions, so resident memory stays ~the working set.
    Measured on a 2-shard OLMoE: identical tokens, **RSS 3.90 GB** — the same as
    (slightly under) the single-file 4.11 GB. A *streaming* variant that copies
    each tensor out of its shard so the shard can be freed early was implemented
    and **measured worse** (7.5 GB): materializing loses the mmap's evictability.
    So whole-shard mmap + OS eviction is already resident-optimal; the real value
    of the sharded loader is simply *enabling* multi-file large-MoE checkpoints,
    not reducing memory below the model-size floor.

---

## 7. What a production `@mlx-ts/lm` still needs

The list has shrunk a lot — most of the original items are now built and
validated (§3, §6, §7b). None of what remains is a feasibility risk; it is
ergonomics, packaging, and breadth.

**Done since this list was first written:** tokenizer (11/11 vs HF, and it
generalized unchanged from Qwen BPE to OLMoE GPT-NeoX BPE), **HF chat template**
(`@huggingface/jinja`, 4/4 vs Python `jinja2`, real assistant replies via
`chat.ts`), KV-cache decode, temp/top-p sampling, equal-length batching, **MoE**
+ **sharded multi-file loading**, deterministic memory management (`tidy` /
`freeMap` / memory limits), and the **async-eval overlap mechanism** (proven in
§7b — works, composes with `tidy`, and is simply not a lever at 0.6B because
decode is compute-bound).

**Genuinely remaining:**

- **Public streaming / disposable API.** The async-eval mechanism and `tidy()`
  both work; what's missing is the *surface*: an async-generator `stream()` and a
  `using`-based disposable so callers never call `tidy()` or free handles by hand.
- **Ragged-batch padding masks.** Batching is proven at equal length; ragged
  prompts need an additive mask (`sdpa`'s `mask_arr` already accepts one) plus
  left-padding/position bookkeeping.
- **Training is demonstrated end to end with ergonomic APIs** (milestones 14–15):
  the keystone (SGD over a multi-param JS closure, bit-identical to MLX) *and* the
  mechanical tail — **Adam** (`optim.ts`), **cross_entropy** (`loss.ts`), **pytree
  utilities** (`pytree.ts`), **`Module.parameters()`** (`nn.ts`), and a tree-based
  **`valueAndGrad`** (`train.ts`). The real **LoRA fine-tune** of the 4-bit Qwen3
  (`lora-train.ts`: frozen quantized base, rank-8 adapters on q/v) now reads like
  mlx.nn — parameters come from `Module.parameters()`, gradients come back as a
  matching tree, and `Adam.update(params, grads)` flattens/unflattens internally;
  no hand-threaded parameter indices. Loss 3.16→0.0007. Honest caveat: training
  over a 4-bit base is **not bit-reproducible** across implementations — bf16
  rounding accumulates, and near a training instability it amplifies into visibly
  different trajectories; with a stable LR the curves track to float tolerance and
  converge identically (validation criterion: identical start + both converge).
  What's left is just more breadth: AdamW/schedulers, more losses, multi-example
  batches, and the backward-memory regime at scale (activations held for backward,
  so `tidy()` runs *after* the step). Standard minibatch training needs no `vmap`;
  only per-sample-gradient methods do (the one genuine mlx-c capability gap).
- **Breadth.** More architectures (dense + MoE families proven; each new one is
  key-mapping), more quant formats (AWQ/GPTQ, other bit-widths), more sampling
  (top-k, repetition penalties, min-p), and **npm packaging of a prebuilt
  `libmlxc`** — the only native artifact to ship.

The one piece genuinely outside MLX — the **tokenizer** — is done and validated.

---

## 7b. Spikes: de-risking the kill-risk unknowns

Before committing to a production `@mlx-ts/lm`, we spiked the unknowns that could
*kill* the project or limit it to a partial (correctness-only) toy. Both came
back green. (`spike-throughput.ts`, `spike-bench.py`, `spike-moe.ts`.)

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
reproduced Python's result exactly (`0.409543`) over Bun FFI (`spike-moe.ts`).
Then the **full MoE layer** — router → softmax → top-K (`argpartition`) →
weight renormalize → quantized expert dispatch → weighted combine — was built as
an `nn.MoE` module (`nn.ts`) and matched the Python reference **exactly**
(`sum=9.353439`, `sumsq=11773.206`; `spike-moe-layer.ts`).

Finally, a **real MoE model end-to-end**: `nn.MoE` wired into a decoder loads
**OLMoE-1B-7B** (64 experts, top-8, 4-bit, ~3.6 GB), generating coherent text —
`"The capital of France is"` → `" Paris.\n\nThe Louvre is a world-famous art
museum in Paris.\n\nThe Eiffel Tower is a famous iron lattice tower..."` at
~209 tok/s — and **token-for-token identical** to MLX Python (`olmoe.ts` /
`reference-olmoe.py`). This exercised the real-model quirks: experts stored
*individually* (stacked into `[E,out,in]` at load via `mlx_stack`), a *quantized
router*, full-projection q/k-norm (vs Qwen3's per-head), untied lm_head, and
`norm_topk_prob=false`. The tokenizer (`tokenizer.ts`) also handled OLMoE's
GPT-NeoX BPE unchanged. MoE is fully proven — op, layer, and real model.

**Net:** none of the suspected unknowns kill or even meaningfully partial the
project. The two that *could* have — "too slow to serve" and "MoE impossible
over FFI" — both resolved positive. Remaining caveats are scope, not risk:
throughput was measured at 0.6B/4-bit only (larger/batched serving may shift the
JS:GPU ratio, which is exactly where the now-proven overlap mechanism would
start to pay); MoE is proven at the op level but not yet wired into a full model;
and Node FFI parity remains deliberately deferred (Bun-only is the recommended v1
scope).

## 8. File map

**Runtime & codegen**
- `codegen.ts` → `generated.ts` — header parser → FFI bindings (448 symbols, 242 wrappers)
- `mx.ts` — `MX` array class, `FinalizationRegistry` + `tidy()`, ops, sampling,
  `async_eval`, `stack`, `copy`, memory introspection/limits
- `nn.ts` — `Module`, `Linear`, `QuantizedLinear`, `RMSNorm`, `Embedding`,
  `QuantizedEmbedding`, **`MoE`** (router + top-K + quantized expert dispatch)
- `optim.ts` — `Adam` (pytree-aware); `loss.ts` — `crossEntropy`
- `pytree.ts` — `treeFlatten` / `treeUnflattenLike` / `treeMap`;
  `train.ts` — tree-based `valueAndGrad` (the ergonomic training layer)
- `loader.ts` — safetensors loading; `singleFileWeights` / **`shardedWeights`**
  (multi-file, mmap-evictable) + `freeMap`
- `tokenizer.ts` — pure-TS byte-level BPE
- `chat-template.ts` — HF chat template via `@huggingface/jinja`

**Models / demos**
- `qwen.ts` — config-driven Qwen3-0.6B (bf16)
- `qwen-nn.ts` — config-driven Qwen3-0.6B-4bit over `nn.Module` (CLI: sampling, window)
- `chat.ts` — end-to-end chat: message → template → tokenizer → model → reply
- `lora-train.ts` — **LoRA fine-tune** of real 4-bit Qwen3 (Adam + cross_entropy)
- `olmoe.ts` — config-driven **OLMoE-1B-7B MoE** (single-file or `MX_SHARDED`)
- `block*.ts`, `model-*.ts` — the staged PoCs (block, decode, safetensors, quant)
- `inspect-real.ts` — enumerate a real model file's tensors
- `split-olmoe.py` — split a single file into shards (to exercise the sharded loader)

**Spikes** — `spike-throughput.ts` (async overlap / serving viability),
`spike-bench.py` (Python tok/s bar), `spike-moe.ts` (gather_qmm op),
`spike-moe-layer.ts` (full MoE layer), `spike-train.ts` (training: value_and_grad
+ SGD).

**References & validation**
- `reference*.py` — MLX Python mirrors for every milestone
- `tok-reference.py` / `tok-test.ts` — tokenizer ground truth
- `validate-prod.ts` — sampling / batching / bounded-memory checks
- **`validate-all.sh`** — the full suite: every TS path vs its reference (15/15)

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
