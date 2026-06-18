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
python3 reference-chat.py >/dev/null 2>&1
bun chat-test.ts 2>&1 | grep -q "4/4" && ok "chat template vs Python jinja2 (4/4)" || no "chat-template" "parity"
bun spike-moe.ts 2>&1 | grep -q "match: true" && ok "MoE gather_qmm op (vs MLX)" || no "spike-moe" "match"
bun spike-moe-layer.ts 2>&1 | grep -q "match: true" && ok "MoE full layer (vs MLX)" || no "spike-moe-layer" "match"
bun spike-throughput.ts 2>&1 | grep -q "identical (sync == async): true" && ok "async-overlap == sync tokens" || no "spike-throughput" "tokens"
bun stream-test.ts 2>&1 | grep -q "STREAM OK" && ok "public stream() == generate() (ids + text)" || no "stream-test" "parity"
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

echo "=== real models (TS vs MLX Python) ==="
python3 reference-qwen.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun qwen.ts "The capital of France is" >/tmp/v_t.txt 2>&1;          cmp_pair "real Qwen3-0.6B bf16" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference-qwen-q4.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun qwen-nn.ts "The capital of France is" >/tmp/v_t.txt 2>&1;     cmp_pair "real Qwen3-0.6B 4-bit (nn)" /tmp/v_t.txt /tmp/v_p.txt genids
python3 reference-olmoe.py "The capital of France is" >/tmp/v_p.txt 2>&1; bun olmoe.ts "The capital of France is" >/tmp/v_t.txt 2>&1;        cmp_pair "real OLMoE-1B-7B 4-bit (MoE)" /tmp/v_t.txt /tmp/v_p.txt genids
if [ -f model-olmoe-sharded/model.safetensors.index.json ]; then
  MX_SHARDED=model-olmoe-sharded/model.safetensors.index.json bun olmoe.ts "The capital of France is" >/tmp/v_t.txt 2>&1
  cmp_pair "OLMoE sharded streaming load" /tmp/v_t.txt /tmp/v_p.txt genids
fi

echo
printf 'TOTAL: \033[32m%d passed\033[0m, %s\n' "$pass" "$([ $fail -eq 0 ] && echo '0 failed' || printf '\033[31m%d failed\033[0m' $fail)"
exit $fail
