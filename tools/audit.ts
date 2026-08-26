// Structural audit: the checks that kept finding real bugs this week, made
// permanent. Needs no MLX, no weights, no GPU — so CI can run it anywhere.
//
//   bun tools/audit.ts
//
// What it enforces:
//   1. every relative import resolves
//   2. src/ imports nothing outside src/ (a published package must stand alone)
//   3. no file outside src/ is imported across a directory boundary
//      (that is what library code in the wrong place looks like)
//   4. every file path named in the docs exists
import { readFileSync, existsSync } from "node:fs";
import { execSync } from "node:child_process";
import { dirname, join, normalize, relative } from "node:path";

const sh = (cmd: string) => execSync(cmd, { encoding: "utf8" }).trim();
const tracked = sh("git ls-files").split("\n");
const code = tracked.filter((f) => /\.(ts|mjs)$/.test(f));
const docs = tracked.filter((f) => /\.md$/.test(f));

const fails: string[] = [];
const fail = (rule: string, detail: string) => fails.push(`[${rule}] ${detail}`);

// A computed dynamic import (src/ffi picks its backend this way) is not statically resolvable.
const IMPORT = /(?:from\s+"([^"]+)"|import\(\s*"([^"]+)"\))/g;
const edges: Array<[string, string]> = [];
for (const f of code) {
  // Blank out template literals first: tools/codegen.ts embeds the preamble it
  // emits into src/ffi/, whose imports are relative to *there*, not to here.
  const src = readFileSync(f, "utf8").replace(/`(?:[^`\\]|\\.)*`/gs, "``");
  for (const mo of src.matchAll(IMPORT)) {
    const spec = mo[1] ?? mo[2];
    if (!spec?.startsWith(".")) continue;
    const target = normalize(join(dirname(f), spec));
    edges.push([f, target]);
    if (!existsSync(target)) fail("1 broken-import", `${f} -> ${spec}`);
  }
}

// 5. Every imported file must be TRACKED, not merely present on disk. The audit
// otherwise reads a working tree that a fresh clone will not have: `.gitignore`
// carried an unanchored `models/`, which matches any directory of that name at
// any depth — so `src/models/load.ts` was silently skipped by `git add` and the
// pushed repo could not typecheck, while everything passed locally.
const GENERATED = new Set(["src/ffi/generated.ts"]);   // emitted by tools/codegen.ts
const trackedSet = new Set(tracked);
for (const [from, to] of edges) {
  if (!existsSync(to) || trackedSet.has(to) || GENERATED.has(to)) continue;
  fail("5 untracked-import", `${from} imports ${to}, which git does not track (ignored?)`);
}

const top = (p: string) => p.split("/")[0];
for (const [from, to] of edges) {
  if (top(from) === "src" && top(to) !== "src") {
    fail("2 src-not-self-contained", `${from} imports ${to}`);
  }
  if (top(from) !== top(to) && top(to) !== "src") {
    fail("3 cross-dir-import", `${from} imports ${to} (library code in the wrong place?)`);
  }
}

// Paths named in prose or in `bun x/y.ts` commands, resolved relative to the doc.
const EXTERNAL = new Set(["qwen3.py", "runcpu.sh", "speedrun.sh", "runs/runcpu.sh"]);
const TOPDIRS = new Set(tracked.filter((f) => f.includes("/")).map(top));
for (const d of docs) {
  const text = readFileSync(d, "utf8");
  const named = new Set<string>();
  for (const mo of text.matchAll(/`([A-Za-z0-9_./-]+\.(?:ts|py|sh|md|mjs|html))`/g)) named.add(mo[1]!);
  for (const mo of text.matchAll(/\b(?:bun|node|python3|bash|deno run(?: --allow-all)?)\s+([A-Za-z0-9_./-]+\.(?:ts|py|sh|mjs))/g)) named.add(mo[1]!);
  for (const p of named) {
    if (EXTERNAL.has(p) || (!p.includes("/") && /^(README|AGENTS)\.md$/.test(p))) continue;
    const candidate = TOPDIRS.has(top(p)) ? p : normalize(join(dirname(d), p));
    if (!existsSync(candidate)) fail("4 broken-doc-path", `${d} names ${p}`);
  }
}

if (fails.length) {
  console.error(`audit: ${fails.length} problem(s)\n` + fails.map((f) => `  ${f}`).join("\n"));
  process.exit(1);
}
console.log(`audit: clean — ${code.length} source files, ${edges.length} imports, ${docs.length} docs`);
