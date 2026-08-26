// Does the MLX repo layout actually cover every weight the model asks for?
//
// jasonvassallo/mlx-musicgen-* stores the LM and T5 under mlx-audiogen's own
// names, which mlxName() rewrites from the Hugging Face ones. Getting one of
// those rewrites wrong is a `missing tensor` error 6 GB into a download, so
// this checks the mapping against the real safetensors headers — which are a
// range request each, no weights fetched.
//
//   bun validation/musicgen-mlx-layout.ts [repo]
import { mlxName } from "../src/models/musicgen.ts";

const repo = process.argv[2] ?? "jasonvassallo/mlx-musicgen-medium";

async function header(file: string): Promise<Set<string>> {
  const url = `https://huggingface.co/${repo}/resolve/main/${file}`;
  const head = await fetch(url, { headers: { range: "bytes=0-7" } });
  const n = Number(new DataView(await head.arrayBuffer()).getBigUint64(0, true));
  const res = await fetch(url, { headers: { range: `bytes=8-${8 + n - 1}` } });
  return new Set(Object.keys(JSON.parse(await res.text())).filter((k) => k !== "__metadata__"));
}

const cfg = await (await fetch(`https://huggingface.co/${repo}/resolve/main/config.json`)).json();
const d = cfg.decoder, t = cfg.text_encoder;

// Every name the model code requests, in the same spelling it uses.
const want: string[] = ["enc_to_dec_proj.weight", "enc_to_dec_proj.bias"];
const lm = "decoder.model.decoder";
want.push(`${lm}.embed_positions.weights`, `${lm}.layer_norm.weight`, `${lm}.layer_norm.bias`);
for (let k = 0; k < d.num_codebooks; k++)
  want.push(`${lm}.embed_tokens.${k}.weight`, `decoder.lm_heads.${k}.weight`);
for (let l = 0; l < d.num_hidden_layers; l++) {
  const p = `${lm}.layers.${l}`;
  for (const nrm of ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"])
    want.push(`${p}.${nrm}.weight`, `${p}.${nrm}.bias`);
  for (const att of ["self_attn", "encoder_attn"])
    for (const proj of ["q_proj", "k_proj", "v_proj", "out_proj"])
      want.push(`${p}.${att}.${proj}.weight`);
  want.push(`${p}.fc1.weight`, `${p}.fc2.weight`);
}
const te = "text_encoder";
want.push(`${te}.shared.weight`, `${te}.encoder.final_layer_norm.weight`,
          `${te}.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight`);
for (let b = 0; b < t.num_layers; b++) {
  const p = `${te}.encoder.block.${b}`;
  for (const q of ["q", "k", "v", "o"]) want.push(`${p}.layer.0.SelfAttention.${q}.weight`);
  want.push(`${p}.layer.0.layer_norm.weight`, `${p}.layer.1.layer_norm.weight`,
            `${p}.layer.1.DenseReluDense.wi.weight`, `${p}.layer.1.DenseReluDense.wo.weight`);
}

const [lmKeys, t5Keys] = await Promise.all([header("decoder.safetensors"), header("t5.safetensors")]);
const missing = want.filter((n) => {
  const into = n.startsWith(`${te}.`) ? t5Keys : lmKeys;
  return !into.has(mlxName(n));
});

console.log(`  repo    : ${repo}`);
console.log(`  layers  : ${d.num_hidden_layers} LM / ${t.num_layers} T5, ${d.num_codebooks} codebooks`);
console.log(`  checked : ${want.length} names -> ${lmKeys.size} LM + ${t5Keys.size} T5 tensors`);
if (missing.length) {
  console.log(`  missing : ${missing.length}`);
  for (const n of missing.slice(0, 12)) console.log(`      ${n}  ->  ${mlxName(n)}`);
}
console.log(`  verdict : ${missing.length === 0 ? "OK" : "FAIL"}`);
