/* Syntax-check every inline <script> block in demo/index.html and flag duplicate
   top-level definitions (the failure mode for edits to large single-file apps). */
const fs = require("fs"), vm = require("vm");
const p = process.argv[2] || "C:/Users/user/Desktop/NoduleDetect_Benchmark/demo/index.html";
console.log("checking:", p);
const html = fs.readFileSync(p, "utf8");

// only blocks WITH content (skip <script src=...></script>)
const blocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
console.log("inline script blocks:", blocks.length);
let ok = true;
blocks.forEach((b, i) => {
  try { new vm.Script(b); console.log(`  block ${i}: SYNTAX OK (${b.length} chars)`); }
  catch (e) { ok = false; console.log(`  block ${i}: SYNTAX ERROR -> ${e.message}`); }
});

const src = blocks.join("\n");
for (const kind of ["function", "const", "let"]) {
  const re = kind === "function" ? /^function\s+([A-Za-z_$][\w$]*)\s*\(/gm
           : new RegExp("^" + kind + "\\s+([A-Za-z_$][\\w$]*)\\s*=", "gm");
  const c = {}; let m;
  while ((m = re.exec(src))) c[m[1]] = (c[m[1]] || 0) + 1;
  const dup = Object.entries(c).filter(([, n]) => n > 1);
  console.log(`  duplicate ${kind}:`, dup.length ? dup : "none");
}

// referential sanity: every getElementById in JS must exist as an id in the HTML
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
const used = new Set([...src.matchAll(/getElementById\(['"]([^'"]+)['"]\)/g)].map(m => m[1]));
const missing = [...used].filter(u => !ids.has(u));
console.log("  ids referenced but not present:", missing.length ? missing : "none");
console.log(ok && !missing.length ? "\nRESULT: OK" : "\nRESULT: PROBLEMS");