"""Republish the HuggingFace model repo with a single clean commit.

WARNING - DESTRUCTIVE. This deletes the HuggingFace repo and recreates it. Any
stars, comments, discussions, community branches or other collaborators' commits
on the existing repo are lost, and the URL is briefly unavailable. It only
reproduces the files listed in FILES below, from this working tree.

Why recreate rather than force-push: the repo carries upload history (initial
upload, then a model-card rewrite). Deleting and re-uploading once gives a
one-commit history, which is quicker than cloning the 122 MB LFS object back down
just to push it again.

Usage:  HF_TOKEN=... python scripts/republish_hf.py
"""
import os, sys
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TOKEN = os.environ["HF_TOKEN"]
REPO = "kcheung-inno/nih-chestxray14-rfdetr-small"
api = HfApi(token=TOKEN)
print("authenticated as:", api.whoami()["name"])

FILES = [
    ("docs/HF_MODEL_CARD.md", "README.md"),
    ("demo/model_nih/rfdetr_nih.onnx", "rfdetr_nih.onnx"),
    ("LICENSE", "LICENSE"),
    ("LICENSING.md", "LICENSING.md"),
    ("DATA_LICENSE.md", "DATA_LICENSE.md"),
]
missing = [src for src, _ in FILES if not os.path.exists(src)]
if missing:
    sys.exit("missing local file(s): " + ", ".join(missing))

print("deleting existing repo (history reset)...")
try:
    api.delete_repo(REPO, repo_type="model", missing_ok=True)
    print("  deleted")
except Exception as e:
    sys.exit(f"could not delete: {e}")

print("recreating...")
create_repo(REPO, repo_type="model", private=False, exist_ok=True, token=TOKEN)

print("uploading as a single commit...")
# upload_folder needs a directory, so stage the files first
import shutil, tempfile
stage = tempfile.mkdtemp(prefix="hf_stage_")
for src, dst in FILES:
    shutil.copy2(src, os.path.join(stage, dst))
api.upload_folder(
    repo_id=REPO,
    repo_type="model",
    folder_path=stage,
    commit_message="Initial release",
    commit_description=(
        "RF-DETR-S fine-tuned for 8-class lesion localisation on the box-annotated subset "
        "of NIH ChestX-ray14, exported to ONNX at 576x576. Research use only - not a medical device."
    ),
)
shutil.rmtree(stage, ignore_errors=True)

info = api.model_info(REPO, files_metadata=False)
print("  private :", info.private)
print("  license :", (info.cardData or {}).get("license"))
print("  files   :", sorted(api.list_repo_files(REPO, repo_type="model")))

commits = api.list_repo_commits(REPO, repo_type="model")
print("  commits :", len(commits), "->", [c.title for c in commits])
