"""One-off uploader for the HuggingFace model repo. Token comes from the environment
and is never written to disk or into the repo."""
import os, sys
from huggingface_hub import HfApi, create_repo

# run from the repo root regardless of the caller's cwd - every path below is relative
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TOKEN = os.environ["HF_TOKEN"]
REPO = "kcheung-inno/nih-chestxray14-rfdetr-small"
api = HfApi(token=TOKEN)

# fail fast and loudly rather than half-uploading
REQUIRED = {
    "docs/HF_MODEL_CARD.md": None,
    "demo/model_nih/rfdetr_nih.onnx": None,
    "LICENSE": None,
    "DATA_LICENSE.md": None,
}
missing = [p for p in REQUIRED if not os.path.exists(p)]
if missing:
    sys.exit("missing required file(s): " + ", ".join(missing))

print("authenticated as:", api.whoami()["name"])

url = create_repo(REPO, repo_type="model", exist_ok=True, private=False, token=TOKEN)
print("repo:", url)

# model card first, so the repo always shows its licence
api.upload_file(
    path_or_fileobj="docs/HF_MODEL_CARD.md",
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
    token=TOKEN,
)
print("uploaded README.md (model card)")

# the ONNX graph (~117 MB, goes via LFS)
api.upload_file(
    path_or_fileobj="demo/model_nih/rfdetr_nih.onnx",
    path_in_repo="rfdetr_nih.onnx",
    repo_id=REPO,
    repo_type="model",
    token=TOKEN,
)
print("uploaded rfdetr_nih.onnx")

# and a copy of the licence so the repo is self-contained legally
api.upload_file(
    path_or_fileobj="LICENSE",
    path_in_repo="LICENSE",
    repo_id=REPO,
    repo_type="model",
    token=TOKEN,
)
print("uploaded LICENSE")

api.upload_file(
    path_or_fileobj="DATA_LICENSE.md",
    path_in_repo="DATA_LICENSE.md",
    repo_id=REPO,
    repo_type="model",
    token=TOKEN,
)
print("uploaded DATA_LICENSE.md")

print("files now in repo:", sorted(api.list_repo_files(REPO, repo_type="model")))
