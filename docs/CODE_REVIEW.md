# Source review: TypeScript over mlx-c

Reviewed 2026-08-28.

## Scope

This review treats the README as the specification and examines the source as a
TypeScript SDK over Apple's `mlx-c` API. The focus is the actual binding and
runtime design: ARM64 ABI handling, mlx-c calls, native ownership, lazy
evaluation, tensor layouts, autograd, generation, and model forward passes.

The tensor math and model layouts are generally credible. The Qwen/Qwen2/MoE
attention shapes, RoPE offsets, KV-cache construction, channels-last
convolutions, quantized matmul conventions, and lazy-evaluation boundaries are
consistent with MLX. The implementation is real TypeScript driving mlx-c, not a
Python sidecar or an HTTP wrapper.

The binding is not yet fully safe as a general-purpose TypeScript SDK. The main
issues are below, in priority order.

## Findings

### P1: mlx-c errors terminate the JavaScript process

Generated operation wrappers discard the `int` status returned by mlx-c:

- `tools/codegen.ts:199-203`
- `src/core/mx.ts:168-172`

The generated symbol table does not include `error.h`, so the SDK never installs
an `mlx_error_handler_func`:

- `tools/codegen.ts:39-40`

As a result, invalid input does not become a catchable TypeScript exception. A
shape-invalid `matmul` placed inside `try/catch` printed an MLX error and exited
the Bun process with status 255.

This is especially important for the HTTP server and any application accepting
user-derived dimensions, model configurations, or tensor shapes. A malformed
request should not be able to terminate the host process.

Recommended direction:

1. Bind `mlx_set_error_handler`.
2. Capture the current native error message without throwing through the C
   callback boundary.
3. Check every status-returning mlx-c call immediately.
4. Convert failures into a TypeScript `Error` containing the operation name and
   native message.

### P1: host-array constructors can read beyond TypedArray memory

The public constructors pass a data pointer and shape directly to
`mlx_array_new_data` without checking the relationship between them:

- `src/core/mx.ts:574-594`

mlx-c receives no byte length for the source buffer. If the product of the
shape is larger than `data.length`, native code copies beyond the TypedArray's
allocation. A small reproduction using one float with shape `[4]` returned
additional values read outside the declared TypedArray.

The constructors also accept fractional, negative, non-finite, and overflowing
dimensions. Shape `[]` currently fails inside Bun FFI because a pointer cannot
be obtained from a zero-length shape buffer.

Recommended direction:

- Require finite, nonnegative integer dimensions.
- Check for safe product overflow.
- Require `product(shape) === data.length`.
- Handle scalar shape `[]` explicitly, or document that callers must use
  `scalar()` and reject it with a clear error.

### P1: `tidy()` has incorrect ownership semantics

The implementation is in `src/core/mx.ts:35-49`.

#### Cleanup is skipped when the callback throws

`ARENA` is restored in `finally`, but resource cleanup occurs after that block.
When `fn()` throws, control leaves the function after `finally`, so lines 45-48
never execute. A direct reproduction confirmed that an `MX` created before the
exception retained a live handle.

This undermines deterministic cleanup precisely on error paths, where a
long-lived service needs it most.

#### A nested `tidy()` can take ownership of an external array

The child scope adds every returned array to the parent arena:

```ts
if (parent) for (const k of keep) parent.add(k);
```

That includes arrays which were not created by the child. A reproduction using
an array created outside both scopes showed its handle being zeroed by the outer
`tidy()` merely because the inner scope returned it temporarily.

Only values owned by the child arena should be adopted by the parent. External
inputs must remain externally owned.

Recommended direction:

- Catch exceptions, free everything owned by the current arena, restore the
  parent, and rethrow.
- Adopt only values that belong to `cur`, not every value reachable from the
  callback result.
- Add regression cases for thrown callbacks and returned external inputs.

### P1: public batch generation does not free its KV cache

`generateBatch()` constructs a cache and returns without freeing its final
contents:

- `src/models/qwen-nn.ts:135-147`

Every intermediate step uses `stepTidy()`, but the last key/value pair from each
layer remains owned by the local cache. Once the function returns, cleanup is
left to `FinalizationRegistry`.

Repeated synchronous batch calls can therefore accumulate native memory until a
JavaScript GC happens. This is the exact lifetime pattern for which the project
correctly says `FinalizationRegistry` is insufficient.

Recommended direction: wrap the full generation body in `try/finally` and free
every remaining cache entry in `finally`, as `streamTokens()` already does.

The function should also validate that the batch is nonempty and every sequence
has the same length before flattening it into a shaped native array.

### P2: custom Metal dispatch leaks native resources per call

`metalKernel().apply()` creates several native objects:

- an input `mlx_vector_array` from `vecArray(inputs)`;
- an output `mlx_vector_array` returned through `outVec`;
- a kernel configuration;
- output array handles extracted from the vector.

See `src/ffi/kernel.ts:53-73`.

The input vector and output vector are never freed. Because the output vector
keeps references to its arrays, freeing the returned `MX` wrappers does not
release all native ownership. The configuration also leaks if application
fails, and the compiled kernel itself has no `free()` or `Symbol.dispose` path.

A direct test dispatched a simple 1,024-float copy kernel 1,000 times, evaluated
and freed every returned `MX`, and still grew MLX active memory by exactly
4.096 MB.

This affects EnCodec's LSTM, which dispatches the kernel once per time step per
layer.

Recommended direction:

- Store and free the input vector in `finally`.
- Free the output vector after extracting retained output handles.
- Always free the configuration in `finally`.
- Make the compiled kernel disposable and call
  `mlx_fast_metal_kernel_free`.
- Remove permanent JS-buffer retention once synchronous argument lifetime is
  established.

### P2: model-weight ownership is incomplete

`singleFileWeights()` returns an object whose native map must be closed with
`done()`:

- `src/io/loader.ts:86-90`

`Qwen2` consumes this object, creates individual weight handles, but neither
retains the `Weights` object nor calls `done()`:

- `src/models/qwen2.ts:50-90`

The raw safetensors map is therefore lost and can never be released. The
individual tensors retain their own MLX references, so calling `W.done()` after
successful construction would be safe and would match the Qwen3 and OLMoE
ownership pattern.

High-level MusicGen, Stable Diffusion, and Spark-TTS models intentionally retain
weight maps because they fetch weights during each forward pass. However, those
model objects expose no `free()` or `Symbol.dispose`, so an application cannot
deterministically unload a model. Once such a model becomes unreachable, its raw
map handles are not managed by `FinalizationRegistry` and may remain alive for
the rest of the process.

Recommended direction:

- Call `W.done()` in `Qwen2` after successful construction, using `try/finally`
  for partially constructed models.
- Give every assembled high-level model explicit disposal ownership.
- Make shared `Weights` ownership reference-counted or ensure exactly one
  top-level model object closes each shared source.
- Make `done()` idempotent.

### P2: generated wrappers permanently retain temporary JavaScript buffers

The generated preamble contains a global array:

- `tools/codegen.ts:228-237`

Every shape buffer, C string, input array, and vector buffer passed through
`kptr`, `cstr`, `array`, or `vec` is pushed into this array and retained for the
life of the module.

For operation arguments, the TypedArray local remains alive for the duration of
the synchronous FFI call. For `mlx_array_new_data`, mlx-c documents that the
input is copied. Permanent retention is therefore unnecessary and makes the raw
Qwen/GPT-2 paths grow JavaScript heap with each token and reshape.

The same pattern appears in:

- `src/io/loader.ts:13-18`
- `src/ffi/kernel.ts:25-28`

Recommended direction: keep buffers in local variables spanning the native call
and remove the module-global retention arrays. If a particular API truly borrows
a pointer after return, give that native object explicit ownership of the
specific buffer instead.

### P3: the generated `show()` helper returns the wrong runtime type

The helper is emitted as:

```ts
export function show(a: Arr): string {
  // ...
  return m.mlx_string_data(Number(s[0])) as unknown as string;
}
```

See `tools/codegen.ts:264-267`.

`mlx_string_data` is bound as a pointer return, so the function returns a number,
not a JavaScript string. A direct call confirmed `typeof show(a) === "number"`.
The temporary `mlx_string` is also never freed.

Recommended direction: decode the pointer with the runtime-neutral `cstring()`
helper and free the `mlx_string` in `finally`.

### P3: seed zero is silently ignored

`streamTokens()` seeds only when the number is truthy:

- `src/text/lm.ts:83`

```ts
if (opts.seed) seed(opts.seed);
```

Zero is a valid deterministic seed, but currently selects the existing global
RNG state instead.

Recommended direction:

```ts
if (opts.seed !== undefined) seed(opts.seed);
```

## Additional design observations

- `MX.h` and the `MX` constructor are public. A caller can wrap the same raw
  handle twice and cause double ownership, or mutate `h` directly. For a safer
  SDK, raw-handle construction should be internal or explicitly marked unsafe.
- Most core operations do not guard against use after `free()`. A JavaScript
  error naming the freed tensor would be preferable to handing a null handle to
  mlx-c.
- `valueAndGrad()` retains callbacks and native closures globally and exposes no
  disposal path. This is acceptable for one process-lifetime training function,
  but repeated construction leaks callback/closure resources.
- `treeUnflattenLike()` and `Adam.update()` do not verify matching tree
  structures or leaf counts. A mismatch currently becomes an indirect native or
  JavaScript failure rather than a useful optimizer error.
- The raw `src/models/qwen.ts` and `src/models/gpt2.ts` paths are useful parity
  demonstrations but do not implement the deterministic ownership discipline of
  the `MX`/`tidy()` model paths. They should remain clearly identified as
  validation/CLI implementations rather than the production SDK surface.

## Verdict

The core architectural claim is valid: this is a real TypeScript MLX runtime
over Apple's C API, and the model forward passes use the correct kinds of MLX
operations and layouts. The implementation demonstrates that Bun, Deno, and
Node can drive meaningful MLX inference and training without custom native
bindings.

It should not yet be described as a fully production-safe TypeScript SDK. The
native error boundary, host-buffer validation, `tidy()` ownership rules,
custom-kernel cleanup, and model disposal need to be corrected first. Those are
runtime-safety and lifetime issues, not doubts about whether the MLX computation
itself is real or numerically meaningful.
