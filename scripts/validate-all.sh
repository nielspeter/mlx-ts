#!/usr/bin/env bash
# Full validation suite: every TS path vs its MLX-Python / HF reference.
cd "$(dirname "$0")/.." || exit 1   # run from the repo root
mkdir -p models data checkpoints          # gitignored; absent in a fresh clone
pass=0; fail=0
ok(){ printf '  \033[32m✅ %s\033[0m\n' "$1"; pass=$((pass+1)); }
no(){ printf '  \033[31m❌ %s\033[0m  (%s)\n' "$1" "$2"; fail=$((fail+1)); }
genids(){ grep -E 'gen ids|generated:' | grep -oE '\[[-0-9, ]+\]' | tail -1; }
wline(){ grep '^W:'; }
fp(){ grep -oE 'sum *= *[-0-9.]+|sum_sq *= *[-0-9.]+' | tr -d ' ' | tr '\n' ' '; }
cmp_pair(){ # name, ts-output-file, py-output-file, extractor
  local t p; t=$($4 <"$2"); p=$($4 <"$3")
  if [ -n "$t" ] && [ "$t" = "$p" ]; then ok "$1"; else no "$1" "TS=$t PY=$p"; fi
}

# Which libmlxc is in play matters: the bundled prebuilds/ copy and Homebrew's
# are different MLX builds and do NOT produce identical numerics. Print it, so a
# parity failure is never mistaken for a code regression. MLXTS_LIB overrides.
echo "mlx-c: $(bun -e 'import {LIBMLXC} from "./src/ffi/native-lib.ts"; console.log(LIBMLXC)')"

echo "=== codegen ==="
bun tools/codegen.ts >/tmp/v_cg.txt 2>&1 && grep -q "wrappers" /tmp/v_cg.txt \
  && ok "codegen ($(grep -oE '[0-9]+ FFI entries' /tmp/v_cg.txt), $(grep -oE '[0-9]+ typed op wrappers' /tmp/v_cg.txt))" || no "codegen" "see /tmp/v_cg.txt"

echo "=== static analysis ==="
# Nothing compiles this project — Bun/Deno/Node all strip types without checking
# them — so tsc is the only thing that ever looks at the types.
bunx tsc --noEmit >/tmp/v_tsc.txt 2>&1 \
  && ok "tsc --noEmit (0 type errors)" \
  || no "typecheck" "$(grep -c 'error TS' /tmp/v_tsc.txt) errors, see /tmp/v_tsc.txt"

# Structural invariants (imports resolve, src/ stands alone, docs point at real
# files). Same script CI runs; needs no MLX.
bun tools/audit.ts >/tmp/v_audit.txt 2>&1 \
  && ok "$(tail -1 /tmp/v_audit.txt)" \
  || no "structural audit" "$(head -1 /tmp/v_audit.txt)"

# Distribution: pack, install into a throwaway project, and use it from all
# three runtimes. Nothing else in the suite sees what a consumer sees.
if bash scripts/verify-package.sh >/tmp/v_pkg.txt 2>&1; then
  ok "npm package installs and runs ($(grep -c ': ok' /tmp/v_pkg.txt)/3 runtimes)"
else
  no "npm package" "$(tail -2 /tmp/v_pkg.txt | tr '\n' ' ')"
fi

# The published surface: nothing else in the suite imports it, so a missing or
# broken export would otherwise go unnoticed until a consumer hit it.
NEXP=$(bun -e 'import * as m from "./src/index.ts";
  const need = ["MX","tidy","Tokenizer","generate","valueAndGrad","loadWhisper","loadSafetensors","backend"];
  const missing = need.filter((k) => !(k in m));
  if (missing.length) { console.error("missing: " + missing.join(", ")); process.exit(1); }
  console.log(Object.keys(m).length);' 2>/tmp/v_api.txt)
[ -n "$NEXP" ] && ok "public API src/index.ts loads ($NEXP exports)" || no "public API" "$(cat /tmp/v_api.txt | tail -1)"


# Completeness, not just presence — see tools/check-api.ts for why.
bun tools/check-api.ts >/tmp/v_apic.txt 2>&1 \
  && ok "$(tail -1 /tmp/v_apic.txt)" \
  || no "public API completeness" "$(tail -1 /tmp/v_apic.txt)"

echo "=== runtimes (same block fingerprint on Bun / Deno / Node) ==="
FP_BUN=$(bun validation/block-gen.ts 2>/dev/null | fp)
[ -n "$FP_BUN" ] && ok "bun: Qwen3 block ($FP_BUN)" || no "bun" "no fingerprint"
if command -v deno >/dev/null 2>&1; then
  FP=$(deno run --allow-all validation/block-gen.ts 2>/dev/null | fp)
  [ -n "$FP" ] && [ "$FP" = "$FP_BUN" ] && ok "deno: identical to Bun" || no "deno" "DENO=$FP BUN=$FP_BUN"
fi
if command -v node >/dev/null 2>&1; then
  FP=$(node validation/block-gen.ts 2>/dev/null | fp)
  [ -n "$FP" ] && [ "$FP" = "$FP_BUN" ] && ok "node: identical to Bun" || no "node" "NODE=$FP BUN=$FP_BUN"
fi
if [ -f models/model-q4.safetensors ]; then   # examples/ must run everywhere too
  for RT in "deno run --allow-all" "node"; do
    command -v ${RT%% *} >/dev/null 2>&1 || continue
    OUT=$($RT examples/chat.ts "What is 2+2?" 2>&1 | tail -1)
    case "$OUT" in *"tok/s"*) ok "${RT%% *}: examples/chat.ts runs" ;;
                   *) no "${RT%% *} examples/chat.ts" "$OUT" ;; esac
  done
fi

echo "=== examples (no model files needed) ==="
for EX in basics module train; do
  OUT=$(bun examples/$EX.ts 2>&1 | tail -1)
  [ -n "$OUT" ] && ok "examples/$EX.ts" || no "examples/$EX.ts" "$OUT"
done
# train.ts doubles as the regression test for Adam inside tidy(): its moment
# buffers used to be freed as scope-local intermediates on the next step.
bun examples/train.ts 2>&1 | tail -1 | grep -qE "loss 0\.[01]" \
  && ok "Adam converges inside tidy() (loss < 0.2)" || no "Adam in tidy" "did not converge"

echo "=== unit/self checks ==="
python3 reference/tok-reference.py >/dev/null 2>&1
bun tests/tok-test.ts 2>&1 | grep -q "11/11" && ok "tokenizer vs HF tokenizers (11/11)" || no "tokenizer" "parity"
if [ -f models/gpt2-tokenizer.json ]; then
  python3 reference/reference-gpt2-tok.py >/dev/null 2>&1
  bun tests/gpt2-tok-test.ts 2>&1 | grep -q "8/8" && ok "GPT-2 BPE encoder vs HF tokenizers (8/8)" || no "gpt2-tok" "parity"
fi
if [ -f data/input.txt ]; then   # tok_train: train BPE in native Rust -> our TS inference round-trips it
  VOCAB=2048 python3 reference/tok-train.py >/dev/null 2>&1
  bun tests/tok-train-test.ts 2>&1 | grep -q "4/4" && ok "BPE tokenizer training (Rust) -> TS inference (4/4)" || no "tok-train" "round-trip"
  if [ -f models/tokenizer-trained.json ]; then   # data pipeline: stream-encode -> memmap token shards -> base_train
    CORPUS=data/input.txt TOKENS=/tmp/vtok bun training/data-prep.ts >/dev/null 2>&1
    TOKENS=/tmp/vtok CKPT=/tmp/vstream.safetensors ITERS=80 N_EMBD=128 bun training/base-train.ts >/tmp/vstream.txt 2>&1
    { grep -q "^mmap " /tmp/vstream.txt && grep -q "CKPT roundtrip: OK" /tmp/vstream.txt; } && ok "streaming dataloader (data-prep -> mmap shards -> base_train)" || no "data-prep/mmap" "streaming"
  fi
  # base_train: pretrain on BPE tokens + save a checkpoint that reloads (safetensors writer)
  if [ -f models/tokenizer-trained.json ]; then
    ITERS=200 bun training/base-train.ts >/tmp/vb.txt 2>&1
    { grep -q "CKPT roundtrip: OK" /tmp/vb.txt && [ -f checkpoints/base-ckpt.safetensors ]; } && ok "base_train + checkpoint save/reload (BPE pretrain)" || no "base-train" "checkpoint"
    # chat_sft handoff: load the base checkpoint, SFT, then a separate CLI answers a trained Q
    if [ -f checkpoints/base-ckpt.safetensors ]; then
      ITERS=300 bun training/chat-sft.ts >/tmp/vc.txt 2>&1
      { [ -f checkpoints/chat-ckpt.safetensors ] && bun training/chat-ckpt.ts "What is the capital of France?" 2>&1 | grep -qi "paris"; } && ok "chat_sft from checkpoint -> chat CLI (pretrain->SFT->chat)" || no "chat-sft" "handoff"
      if [ -f checkpoints/chat-ckpt.safetensors ]; then   # chat_web: serve the checkpoint over the OpenAI API
        PORT=8123 bun examples/chat-web.ts >/tmp/vw.log 2>&1 & WPID=$!
        for i in $(seq 1 60); do curl -s localhost:8123/health >/dev/null 2>&1 && break; sleep 0.5; done
        ans=$(curl -s localhost:8123/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}' 2>/dev/null)
        kill $WPID 2>/dev/null
        echo "$ans" | grep -qi paris && ok "chat_web serves the SFT'd checkpoint (OpenAI API)" || no "chat-web" "endpoint"
      fi
    fi
  fi
fi
python3 reference/reference-chat.py >/dev/null 2>&1
bun tests/chat-test.ts 2>&1 | grep -q "4/4" && ok "chat template vs Python jinja2 (4/4)" || no "chat-template" "parity"
bun validation/spike-moe.ts 2>&1 | grep -q "match: true" && ok "MoE gather_qmm op (vs MLX)" || no "spike-moe" "match"
bun validation/spike-moe-layer.ts 2>&1 | grep -q "match: true" && ok "MoE full layer (vs MLX)" || no "spike-moe-layer" "match"
bun validation/spike-throughput.ts 2>&1 | grep -q "identical (sync == async): true" && ok "async-overlap == sync tokens" || no "spike-throughput" "tokens"
bun tests/stream-test.ts 2>&1 | grep -q "STREAM OK" && ok "public stream() == generate() (ids + text)" || no "stream-test" "parity"
python3 reference/reference-mel.py >/dev/null 2>&1
bun tests/audio-test.ts 2>&1 | grep -q "MEL OK" && ok "log-Mel front-end vs numpy FFT" || no "audio-test" "mel spectrogram parity"
python3 tests/gen-fixtures.py >/dev/null 2>&1
ncase=$(grep -o '"name"' tests/fixtures.json 2>/dev/null | wc -l | tr -d ' ')
bun test tests/ 2>&1 | grep -q " 0 fail" && ok "op binding parity vs MLX (${ncase} cases, bun test)" || no "lib tests" "binding parity (bun test tests/)"
bun tests/validate-prod.ts 2>&1 | grep -q "reproducible (same seed -> same ids): true" && ok "sampling reproducible + batching" || no "validate-prod" "sampling"

echo "=== synthetic parity (TS vs MLX Python) ==="
bun validation/block.ts >/tmp/v_t.txt 2>&1;        python3 reference/reference.py >/tmp/v_p.txt 2>&1;          cmp_pair "Qwen3 block fp32" /tmp/v_t.txt /tmp/v_p.txt fp
bun validation/block-gen.ts >/tmp/v_t.txt 2>&1;    cmp_pair "Qwen3 block (generated wrappers)" /tmp/v_t.txt /tmp/v_p.txt fp
bun validation/model-gen.ts >/tmp/v_t.txt 2>&1;    python3 reference/reference-decode.py >/tmp/v_p.txt 2>&1;   cmp_pair "KV-cache decode" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference/save-model.py >/dev/null 2>&1; bun validation/model-load.ts >/tmp/v_t.txt 2>&1;              cmp_pair "safetensors load + decode" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference/reference-quant.py >/tmp/v_p.txt 2>&1; bun validation/model-quant.ts >/tmp/v_t.txt 2>&1;     cmp_pair "4-bit quantized decode" /tmp/v_t.txt /tmp/v_p.txt genids
bun validation/spike-train.ts >/tmp/v_t.txt 2>&1; python3 reference/reference-train.py >/tmp/v_p.txt 2>&1;     cmp_pair "training (value_and_grad + SGD)" /tmp/v_t.txt /tmp/v_p.txt wline
# LoRA: identical start, both converge (training over a 4-bit base tracks to float tolerance, not bit-exact)
bun training/lora-train.ts >/tmp/v_t.txt 2>&1; python3 reference/reference-lora.py >/tmp/v_p.txt 2>&1
tf=$(grep 'final loss' /tmp/v_t.txt | grep -oE '[0-9.]+$'); pf=$(grep 'final loss' /tmp/v_p.txt | grep -oE '[0-9.]+$')
if [ "$(grep 'step  0' /tmp/v_t.txt)" = "$(grep 'step  0' /tmp/v_p.txt)" ] && awk "BEGIN{exit !($tf<0.01 && $pf<0.01)}"; then
  ok "LoRA fine-tune (Adam + cross_entropy) — converges, matches MLX (3.16 -> ${tf})"
else no "lora-train" "TS final=$tf PY final=$pf"; fi

# microGPT trained from scratch (real MLX autograd over FFI): identical init +
# data -> exact step-0 loss; both converge (training drifts, FINDINGS §6 #8).
if [ -f data/names.txt ]; then
  bun validation/spike-microgpt.ts >/tmp/v_t.txt 2>&1; python3 reference/reference-microgpt.py >/tmp/v_p.txt 2>&1
  t0=$(grep STEP0 /tmp/v_t.txt | grep -oE '[0-9.]+$'); p0=$(grep STEP0 /tmp/v_p.txt | grep -oE '[0-9.]+$')
  tf=$(grep FINAL /tmp/v_t.txt | grep -oE '[0-9.]+$'); pf=$(grep FINAL /tmp/v_p.txt | grep -oE '[0-9.]+$')
  if [ -n "$t0" ] && [ "$t0" = "$p0" ] && awk "BEGIN{exit !($tf<2.6 && $pf<2.6)}"; then
    ok "microGPT from scratch (forward+autograd+Adam) — step0 ${t0}=PY, converges (-> ${tf})"
  else no "spike-microgpt" "TS step0=$t0/final=$tf PY step0=$p0/final=$pf"; fi
fi

# nanoGPT: multi-layer char-level GPT, mini-batched, AdamW + cosine LR + grad clip.
# 100 iters for speed; identical init + batches -> exact match vs MLX-Python.
if [ -f data/input.txt ]; then
  ITERS=100 bun validation/spike-nanogpt.ts >/tmp/v_t.txt 2>&1; ITERS=100 python3 reference/reference-nanogpt.py >/tmp/v_p.txt 2>&1
  t0=$(grep STEP0 /tmp/v_t.txt | grep -oE '[0-9.]+$'); p0=$(grep STEP0 /tmp/v_p.txt | grep -oE '[0-9.]+$')
  tv=$(grep '^VAL' /tmp/v_t.txt | grep -oE '[0-9.]+$'); pv=$(grep '^VAL' /tmp/v_p.txt | grep -oE '[0-9.]+$')
  if [ -n "$t0" ] && [ "$t0" = "$p0" ] && [ -n "$tv" ] && awk "BEGIN{exit !($tv<3.0 && ($tv-$pv<0.05) && ($pv-$tv<0.05))}"; then
    ok "nanoGPT multi-layer from scratch (AdamW+clip+cosine) — step0 ${t0}=PY, val ${tv}≈PY"
  else no "spike-nanogpt" "TS step0=$t0/val=$tv PY step0=$p0/val=$pv"; fi
fi

echo "=== real models (TS vs MLX Python) ==="
# The strongest check available: Apple's OWN mlx-lm, not a reimplementation.
# Every other oracle here builds the forward pass by hand, so a shared
# misreading of the architecture would agree with us and both be wrong. Skipped
# when mlx-lm is not installed (pip install mlx-lm).
if python3 -c "import mlx_lm" >/dev/null 2>&1; then
  python3 reference/reference-mlxlm-qwen.py "The capital of France is" >/tmp/v_p.txt 2>&1
  bun src/models/qwen.ts "The capital of France is" >/tmp/v_t.txt 2>&1
  cmp_pair "real Qwen3-0.6B vs Apple's mlx-lm" /tmp/v_t.txt /tmp/v_p.txt genids
fi
python3 reference/reference-qwen.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun src/models/qwen.ts "The capital of France is" >/tmp/v_t.txt 2>&1;          cmp_pair "real Qwen3-0.6B bf16" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference/reference-qwen-q4.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun src/models/qwen-nn.ts "The capital of France is" >/tmp/v_t.txt 2>&1;     cmp_pair "real Qwen3-0.6B 4-bit (nn)" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference/reference-olmoe.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun src/models/olmoe.ts "The capital of France is" >/tmp/v_t.txt 2>&1;        cmp_pair "real OLMoE-1B-7B 4-bit (MoE)" /tmp/v_t.txt /tmp/v_p.txt genids
if [ -f models/gpt2-model.safetensors ]; then  # own temp files: the sharded-OLMoE check below reuses /tmp/v_p.txt
  python3 reference/reference-gpt2.py "The capital of France is" >/tmp/vg_p.txt 2>&1; bun src/models/gpt2.ts "The capital of France is" >/tmp/vg_t.txt 2>&1; cmp_pair "real GPT-2-124M (BPE + tied head)" /tmp/vg_t.txt /tmp/vg_p.txt genids
  # SFT: full fine-tune of GPT-2-124M, completion-masked loss -> step0 matches, both converge
  ITERS=120 bun training/sft.ts >/tmp/vs_t.txt 2>&1; ITERS=120 python3 reference/reference-sft.py >/tmp/vs_p.txt 2>&1
  s0=$(grep STEP0 /tmp/vs_t.txt | grep -oE '[0-9.]+$'); q0=$(grep STEP0 /tmp/vs_p.txt | grep -oE '[0-9.]+$')
  sf=$(grep FINAL /tmp/vs_t.txt | grep -oE '[0-9.]+$'); qf=$(grep FINAL /tmp/vs_p.txt | grep -oE '[0-9.]+$')
  if [ -n "$s0" ] && [ "$s0" = "$q0" ] && awk "BEGIN{exit !($sf<0.1 && $qf<0.1)}"; then
    ok "SFT GPT-2-124M (full FT, completion loss) — step0 ${s0}=PY, converges (-> ${sf})"
  else no "spike-sft" "TS step0=$s0/final=$sf PY step0=$q0/final=$qf"; fi
  # RL (GRPO): validate the advantage-weighted-NLL loss path on a fixed batch
  CHECK=1 bun training/rl.ts >/tmp/vr_t.txt 2>&1; python3 reference/reference-rl.py >/tmp/vr_p.txt 2>&1
  rt=$(grep RLLOSS /tmp/vr_t.txt); rp=$(grep RLLOSS /tmp/vr_p.txt)
  if [ -n "$rt" ] && [ "$rt" = "$rp" ]; then ok "RL GRPO loss path vs MLX Python (${rt})"; else no "rl" "TS=$rt PY=$rp"; fi
fi
# Whisper needs mlx_whisper for the oracle + the converted weights; run it only
# when both are present (skipped gracefully otherwise — not a hard failure).
PYW=""; for c in /tmp/wvenv/bin/python python3; do "$c" -c "import mlx_whisper" >/dev/null 2>&1 && { PYW=$c; break; }; done
if [ -n "$PYW" ] && [ -f models/whisper-tiny.safetensors ]; then
  "$PYW" reference-whisper.py >/dev/null 2>&1
  bun tests/whisper-test.ts 2>&1 | grep -q "WHISPER OK" && ok "Whisper-tiny encoder+decoder vs mlx_whisper" || no "whisper" "parity"
  if [ -f models/whisper-mel-filters-80.f32 ] && [ -f models/whisper-multilingual.tiktoken ] && [ -f /tmp/jfk.flac ]; then
    "$PYW" reference-whisper-transcribe.py >/dev/null 2>&1
    bun tests/whisper-transcribe-test.ts 2>&1 | grep -q "TRANSCRIBE OK" && ok "Whisper transcription token-exact vs mlx_whisper" || no "whisper-transcribe" "tokens"
  fi
fi

if [ -f models/model-olmoe-sharded/model.safetensors.index.json ]; then
  MX_SHARDED=models/model-olmoe-sharded/model.safetensors.index.json bun src/models/olmoe.ts "The capital of France is" >/tmp/v_t.txt 2>&1
  cmp_pair "OLMoE sharded streaming load" /tmp/v_t.txt /tmp/v_p.txt genids
fi

echo
printf 'TOTAL: \033[32m%d passed\033[0m, %s\n' "$pass" "$([ $fail -eq 0 ] && echo '0 failed' || printf '\033[31m%d failed\033[0m' $fail)"
exit $fail
