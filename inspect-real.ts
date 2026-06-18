// Bonus: load a REAL mlx-community model shard from the HF cache and enumerate
// its tensors from TypeScript — proof that mlx_load handles genuine model files.
//   bun inspect-real.ts <path-to.safetensors>

import { loadSafetensors, entries } from "./loader.ts";

const path = process.argv[2];
if (!path) { console.error("usage: bun inspect-real.ts <file.safetensors>"); process.exit(1); }

const w = loadSafetensors(path);
const all = entries(w);
console.log(`loaded ${path.split("/").pop()} — ${all.length} tensors`);
for (const e of all.slice(0, 8)) console.log(`  ${e.name}  [${e.shape.join(", ")}]`);
console.log(`  … and ${Math.max(0, all.length - 8)} more`);
