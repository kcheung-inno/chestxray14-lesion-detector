#!/usr/bin/env bash
# Pre-publication sanity check: confirm the repository is safe to push.
#
# Checks three things:
#   1. We are actually inside a git repository (see the trap below).
#   2. No large binaries / caches / secrets would be committed.
#   3. The attribution that the NIH licence *requires* is present in the docs.
#
# THE TRAP THIS SCRIPT EXISTS TO AVOID
# ------------------------------------
# `git check-ignore` exits 128 when run outside a repository. The natural idiom
#
#     if git check-ignore -q "$path"; then echo ignored; else echo PUBLISHED; fi
#
# treats that error as "not ignored" and therefore reports EVERY path - including
# ones that must never be published - as safe to publish. A guard that fails open
# on exactly the thing it guards is worse than no guard, so step 1 hard-fails
# rather than assuming.
cd "$(dirname "$0")/.." || exit 1

fail=0

# --- 1. must be inside a git repository -------------------------------------
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "FATAL: not a git repository - cannot verify anything." >&2
  echo "       Run 'git init' first, then re-run this script." >&2
  exit 2
fi
echo "step 1: git repository detected"

# --- 2. nothing bulky, cached or secret should be publishable ---------------
# Patterns that must NOT appear among the files git would commit.
FORBIDDEN=(
  '\.pth$' '\.ckpt$' '\.onnx$' '\.safetensors$'
  '^\.venv/' '^node_modules/' '__pycache__' '\.pyc$'
  '\.log$' '^hf/' '^\.hfhome/' '^data/nih/_hf/'
  '^third_party/'
  '^\.env' '\.pem$' 'id_rsa'
)

TMPLIST="$(mktemp)" || TMPLIST="./.publishable_check.tmp"
# untracked-but-not-ignored files, i.e. what a first commit would actually contain
git ls-files --others --exclude-standard > "$TMPLIST"
# plus anything already tracked
git ls-files >> "$TMPLIST"
trap 'rm -f "$TMPLIST"' EXIT
echo "  (candidate files: $(wc -l < "$TMPLIST"))"

echo
echo "step 2: scanning publishable files for bulky / secret artefacts"
for pat in "${FORBIDDEN[@]}"; do
  hits=$(grep -E "$pat" "$TMPLIST" | head -5)
  if [ -n "$hits" ]; then
    echo "  FAIL  pattern [$pat] would be published:"
    echo "$hits" | sed 's/^/          /'
    fail=1
  fi
done
[ "$fail" -eq 0 ] && echo "  ok    no weights, caches, logs or secrets publishable"

# --- 3. required attribution is present -------------------------------------
echo
echo "step 3: checking the attribution the licence requires"
for f in docs/DATA_CARD.md docs/MODEL_CARD.md docs/HF_MODEL_CARD.md; do
  [ -f "$f" ] || { echo "  FAIL  missing $f"; fail=1; }
done
# the three required elements: the link, the citation, the acknowledgement
chk() {
  if grep -rqs "$1" docs/ ; then echo "  ok    $2"; else echo "  FAIL  missing: $2"; fail=1; fi
}
chk "nihcc.app.box.com" "link to the NIH download site"
chk "Wang" "citation of Wang et al. CVPR 2017"
chk "NIH Clinical Center" "acknowledgement of the NIH Clinical Center"
chk "10.1109/CVPR.2017.369" "DOI of the dataset paper"

# --- 4. data/ and outputs/ must contain only this project's artefacts ---------
# Why this step exists: a leaked artefact need not contain any tell-tale keyword.
# A detections JSON full of image ids and predicted boxes from a research-only
# corpus contains no occurrence of that corpus's name, so a keyword scan passes it
# happily. Naming is the reliable signal here, not content.
echo
echo "step 4: checking data/ and outputs/ hold only this project's artefacts"
stray=$(grep -E '^(data|outputs)/' "$TMPLIST" \
        | grep -vE '(^data/nih/|_nih[/._])' |
        grep -vE '/$')
if [ -n "$stray" ]; then
  echo "  FAIL  unexpected artefacts under data/ or outputs/:"
  echo "$stray" | head -10 | sed 's/^/          /'
  fail=1
else
  echo "  ok    data/ and outputs/ contain only NIH artefacts"
fi

# --- 5. licensing artefacts present ------------------------------------------
# Three different owners, three different licences: our code (Apache-2.0), the NIH
# images (dataset terms), and the vendored onnxruntime-web binaries (MIT, notice
# required). All three notices must ship, or the release is out of compliance.
echo
echo "step 5: checking the licensing artefacts are present"
for f in LICENSE NOTICE DATA_LICENSE.md THIRD-PARTY-NOTICES.md LICENSING.md demo/ort/LICENSE CITATION.cff; do
  if [ -f "$f" ]; then echo "  ok    $f"; else echo "  FAIL  missing $f"; fail=1; fi
done
# the MIT notice must actually name the copyright holder, not just exist
if ! grep -qs "Microsoft Corporation" demo/ort/LICENSE; then
  echo "  FAIL  demo/ort/LICENSE does not carry the Microsoft copyright notice"
  fail=1
fi
# We must not claim a licence we do not have over the NIH images. The provider
# applies no named licence, so an assertion of CC0 or CC BY over the *dataset* is
# wrong. Match affirmative phrasing only - DATA_LICENSE.md legitimately *mentions*
# CC0 in a note warning against exactly that misreading, and a bare substring test
# would flag that note.
# Match only affirmative phrasing. Deliberately NOT a bare "CC0" or "as CC0": the
# warning note below legitimately reads "such as CC0 or CC BY 4.0", and the phrase
# "such as CC0" would trip a looser pattern.
if grep -qE "CC0 1\.0|CC0 Universal|under CC0|CC0 licence|CC0 license" DATA_LICENSE.md; then
  echo "  FAIL  DATA_LICENSE.md asserts CC0, which the dataset provider does not apply"
  fail=1
fi

# --- 6. every script the README advertises actually exists -------------------
# Found a real instance of this: the pipeline list still named verify_onnx.py
# after that script had been removed. Documentation that promises files which do
# not exist is worse than no documentation.
echo
echo "step 6: checking README's pipeline list resolves to real files"
# Extract the contents of the FIRST fenced code block after "## Pipeline".
# NOT `sed -n '/## Pipeline/,/^```$/p'` - that range closes on the OPENING fence,
# yielding an empty block and a check that always passes. That bug shipped in the
# first version of this step and the negative control caught it.
missing=""
while IFS= read -r name; do
  [ -z "$name" ] && continue
  if [ ! -e "scripts/$name" ]; then missing="$missing $name"; fi
done < <(awk '
  /^## Pipeline/ { seen = 1; next }
  seen && /^```/ { if (inblock) exit; inblock = 1; next }
  inblock { print }
' README.md | grep -oE '^[a-z0-9_]+\.(py|sh|js)' | sort -u)
if [ -n "$missing" ]; then
  echo "  FAIL  README names scripts that do not exist:$missing"
  fail=1
else
  echo "  ok    all scripts named in README exist"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: OK - repository is safe to publish"
else
  echo "RESULT: PROBLEMS FOUND - fix the above before publishing"
fi
exit "$fail"
