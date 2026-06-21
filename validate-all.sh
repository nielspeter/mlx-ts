#!/usr/bin/env bash
# Full validation suite: every TS path vs its MLX-Python / HF reference.
cd "$(dirname "$0")" || exit 1
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

echo "=== codegen ==="
bun codegen.ts >/tmp/v_cg.txt 2>&1 && grep -q "wrappers" /tmp/v_cg.txt \
  && ok "codegen ($(grep -oE '[0-9]+ FFI entries' /tmp/v_cg.txt), $(grep -oE '[0-9]+ typed op wrappers' /tmp/v_cg.txt))" || no "codegen" "see /tmp/v_cg.txt"

echo "=== unit/self checks ==="
python3 tok-reference.py >/dev/null 2>&1
bun tok-test.ts 2>&1 | grep -q "11/11" && ok "tokenizer vs HF tokenizers (11/11)" || no "tokenizer" "parity"
if [ -f gpt2-tokenizer.json ]; then
  python3 reference-gpt2-tok.py >/dev/null 2>&1
  bun gpt2-tok-test.ts 2>&1 | grep -q "8/8" && ok "GPT-2 BPE encoder vs HF tokenizers (8/8)" || no "gpt2-tok" "parity"
fi
if [ -f input.txt ]; then   # tok_train: train BPE in native Rust -> our TS inference round-trips it
  VOCAB=2048 python3 tok-train.py >/dev/null 2>&1
  bun tok-train-test.ts 2>&1 | grep -q "4/4" && ok "BPE tokenizer training (Rust) -> TS inference (4/4)" || no "tok-train" "round-trip"
  # base_train: pretrain on BPE tokens + save a checkpoint that reloads (safetensors writer)
  if [ -f tokenizer-trained.json ]; then
    ITERS=200 bun base-train.ts >/tmp/vb.txt 2>&1
    { grep -q "CKPT roundtrip: OK" /tmp/vb.txt && [ -f base-ckpt.safetensors ]; } && ok "base_train + checkpoint save/reload (BPE pretrain)" || no "base-train" "checkpoint"
  fi
fi
python3 reference-chat.py >/dev/null 2>&1
bun chat-test.ts 2>&1 | grep -q "4/4" && ok "chat template vs Python jinja2 (4/4)" || no "chat-template" "parity"
bun spike-moe.ts 2>&1 | grep -q "match: true" && ok "MoE gather_qmm op (vs MLX)" || no "spike-moe" "match"
bun spike-moe-layer.ts 2>&1 | grep -q "match: true" && ok "MoE full layer (vs MLX)" || no "spike-moe-layer" "match"
bun spike-throughput.ts 2>&1 | grep -q "identical (sync == async): true" && ok "async-overlap == sync tokens" || no "spike-throughput" "tokens"
bun stream-test.ts 2>&1 | grep -q "STREAM OK" && ok "public stream() == generate() (ids + text)" || no "stream-test" "parity"
python3 reference-mel.py >/dev/null 2>&1
bun audio-test.ts 2>&1 | grep -q "MEL OK" && ok "log-Mel front-end vs numpy FFT" || no "audio-test" "mel spectrogram parity"
python3 tests/gen-fixtures.py >/dev/null 2>&1
ncase=$(grep -o '"name"' tests/fixtures.json 2>/dev/null | wc -l | tr -d ' ')
bun test tests/ 2>&1 | grep -q " 0 fail" && ok "op binding parity vs MLX (${ncase} cases, bun test)" || no "lib tests" "binding parity (bun test tests/)"
bun validate-prod.ts 2>&1 | grep -q "reproducible (same seed -> same ids): true" && ok "sampling reproducible + batching" || no "validate-prod" "sampling"

echo "=== synthetic parity (TS vs MLX Python) ==="
bun block.ts >/tmp/v_t.txt 2>&1;        python3 reference.py >/tmp/v_p.txt 2>&1;          cmp_pair "Qwen3 block fp32" /tmp/v_t.txt /tmp/v_p.txt fp
bun block-gen.ts >/tmp/v_t.txt 2>&1;    cmp_pair "Qwen3 block (generated wrappers)" /tmp/v_t.txt /tmp/v_p.txt fp
bun model-gen.ts >/tmp/v_t.txt 2>&1;    python3 reference-decode.py >/tmp/v_p.txt 2>&1;   cmp_pair "KV-cache decode" /tmp/v_t.txt /tmp/v_p.txt genids
python3 save-model.py >/dev/null 2>&1; bun model-load.ts >/tmp/v_t.txt 2>&1;              cmp_pair "safetensors load + decode" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference-quant.py >/tmp/v_p.txt 2>&1; bun model-quant.ts >/tmp/v_t.txt 2>&1;     cmp_pair "4-bit quantized decode" /tmp/v_t.txt /tmp/v_p.txt genids
bun spike-train.ts >/tmp/v_t.txt 2>&1; python3 reference-train.py >/tmp/v_p.txt 2>&1;     cmp_pair "training (value_and_grad + SGD)" /tmp/v_t.txt /tmp/v_p.txt wline
# LoRA: identical start, both converge (training over a 4-bit base tracks to float tolerance, not bit-exact)
bun lora-train.ts >/tmp/v_t.txt 2>&1; python3 reference-lora.py >/tmp/v_p.txt 2>&1
tf=$(grep 'final loss' /tmp/v_t.txt | grep -oE '[0-9.]+$'); pf=$(grep 'final loss' /tmp/v_p.txt | grep -oE '[0-9.]+$')
if [ "$(grep 'step  0' /tmp/v_t.txt)" = "$(grep 'step  0' /tmp/v_p.txt)" ] && awk "BEGIN{exit !($tf<0.01 && $pf<0.01)}"; then
  ok "LoRA fine-tune (Adam + cross_entropy) — converges, matches MLX (3.16 -> ${tf})"
else no "lora-train" "TS final=$tf PY final=$pf"; fi

# microGPT trained from scratch (real MLX autograd over FFI): identical init +
# data -> exact step-0 loss; both converge (training drifts, FINDINGS §6 #8).
if [ -f names.txt ]; then
  bun spike-microgpt.ts >/tmp/v_t.txt 2>&1; python3 reference-microgpt.py >/tmp/v_p.txt 2>&1
  t0=$(grep STEP0 /tmp/v_t.txt | grep -oE '[0-9.]+$'); p0=$(grep STEP0 /tmp/v_p.txt | grep -oE '[0-9.]+$')
  tf=$(grep FINAL /tmp/v_t.txt | grep -oE '[0-9.]+$'); pf=$(grep FINAL /tmp/v_p.txt | grep -oE '[0-9.]+$')
  if [ -n "$t0" ] && [ "$t0" = "$p0" ] && awk "BEGIN{exit !($tf<2.6 && $pf<2.6)}"; then
    ok "microGPT from scratch (forward+autograd+Adam) — step0 ${t0}=PY, converges (-> ${tf})"
  else no "spike-microgpt" "TS step0=$t0/final=$tf PY step0=$p0/final=$pf"; fi
fi

# nanoGPT: multi-layer char-level GPT, mini-batched, AdamW + cosine LR + grad clip.
# 100 iters for speed; identical init + batches -> exact match vs MLX-Python.
if [ -f input.txt ]; then
  ITERS=100 bun spike-nanogpt.ts >/tmp/v_t.txt 2>&1; ITERS=100 python3 reference-nanogpt.py >/tmp/v_p.txt 2>&1
  t0=$(grep STEP0 /tmp/v_t.txt | grep -oE '[0-9.]+$'); p0=$(grep STEP0 /tmp/v_p.txt | grep -oE '[0-9.]+$')
  tv=$(grep '^VAL' /tmp/v_t.txt | grep -oE '[0-9.]+$'); pv=$(grep '^VAL' /tmp/v_p.txt | grep -oE '[0-9.]+$')
  if [ -n "$t0" ] && [ "$t0" = "$p0" ] && [ -n "$tv" ] && awk "BEGIN{exit !($tv<3.0 && ($tv-$pv<0.05) && ($pv-$tv<0.05))}"; then
    ok "nanoGPT multi-layer from scratch (AdamW+clip+cosine) — step0 ${t0}=PY, val ${tv}≈PY"
  else no "spike-nanogpt" "TS step0=$t0/val=$tv PY step0=$p0/val=$pv"; fi
fi

echo "=== real models (TS vs MLX Python) ==="
python3 reference-qwen.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun qwen.ts "The capital of France is" >/tmp/v_t.txt 2>&1;          cmp_pair "real Qwen3-0.6B bf16" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference-qwen-q4.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun qwen-nn.ts "The capital of France is" >/tmp/v_t.txt 2>&1;     cmp_pair "real Qwen3-0.6B 4-bit (nn)" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference-olmoe.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun olmoe.ts "The capital of France is" >/tmp/v_t.txt 2>&1;        cmp_pair "real OLMoE-1B-7B 4-bit (MoE)" /tmp/v_t.txt /tmp/v_p.txt genids
if [ -f gpt2-model.safetensors ]; then  # own temp files: the sharded-OLMoE check below reuses /tmp/v_p.txt
  python3 reference-gpt2.py "The capital of France is" >/tmp/vg_p.txt 2>&1; bun gpt2.ts "The capital of France is" >/tmp/vg_t.txt 2>&1; cmp_pair "real GPT-2-124M (BPE + tied head)" /tmp/vg_t.txt /tmp/vg_p.txt genids
  # SFT: full fine-tune of GPT-2-124M, completion-masked loss -> step0 matches, both converge
  ITERS=120 bun sft.ts >/tmp/vs_t.txt 2>&1; ITERS=120 python3 reference-sft.py >/tmp/vs_p.txt 2>&1
  s0=$(grep STEP0 /tmp/vs_t.txt | grep -oE '[0-9.]+$'); q0=$(grep STEP0 /tmp/vs_p.txt | grep -oE '[0-9.]+$')
  sf=$(grep FINAL /tmp/vs_t.txt | grep -oE '[0-9.]+$'); qf=$(grep FINAL /tmp/vs_p.txt | grep -oE '[0-9.]+$')
  if [ -n "$s0" ] && [ "$s0" = "$q0" ] && awk "BEGIN{exit !($sf<0.1 && $qf<0.1)}"; then
    ok "SFT GPT-2-124M (full FT, completion loss) — step0 ${s0}=PY, converges (-> ${sf})"
  else no "spike-sft" "TS step0=$s0/final=$sf PY step0=$q0/final=$qf"; fi
  # RL (GRPO): validate the advantage-weighted-NLL loss path on a fixed batch
  CHECK=1 bun rl.ts >/tmp/vr_t.txt 2>&1; python3 reference-rl.py >/tmp/vr_p.txt 2>&1
  rt=$(grep RLLOSS /tmp/vr_t.txt); rp=$(grep RLLOSS /tmp/vr_p.txt)
  if [ -n "$rt" ] && [ "$rt" = "$rp" ]; then ok "RL GRPO loss path vs MLX Python (${rt})"; else no "rl" "TS=$rt PY=$rp"; fi
fi
# Whisper needs mlx_whisper for the oracle + the converted weights; run it only
# when both are present (skipped gracefully otherwise — not a hard failure).
PYW=""; for c in /tmp/wvenv/bin/python python3; do "$c" -c "import mlx_whisper" >/dev/null 2>&1 && { PYW=$c; break; }; done
if [ -n "$PYW" ] && [ -f whisper-tiny.safetensors ]; then
  "$PYW" reference-whisper.py >/dev/null 2>&1
  bun whisper-test.ts 2>&1 | grep -q "WHISPER OK" && ok "Whisper-tiny encoder+decoder vs mlx_whisper" || no "whisper" "parity"
  if [ -f whisper-mel-filters-80.f32 ] && [ -f whisper-multilingual.tiktoken ] && [ -f /tmp/jfk.flac ]; then
    "$PYW" reference-whisper-transcribe.py >/dev/null 2>&1
    bun whisper-transcribe-test.ts 2>&1 | grep -q "TRANSCRIBE OK" && ok "Whisper transcription token-exact vs mlx_whisper" || no "whisper-transcribe" "tokens"
  fi
fi

if [ -f model-olmoe-sharded/model.safetensors.index.json ]; then
  MX_SHARDED=model-olmoe-sharded/model.safetensors.index.json bun olmoe.ts "The capital of France is" >/tmp/v_t.txt 2>&1
  cmp_pair "OLMoE sharded streaming load" /tmp/v_t.txt /tmp/v_p.txt genids
fi

echo
printf 'TOTAL: \033[32m%d passed\033[0m, %s\n' "$pass" "$([ $fail -eq 0 ] && echo '0 failed' || printf '\033[31m%d failed\033[0m' $fail)"
exit $fail
