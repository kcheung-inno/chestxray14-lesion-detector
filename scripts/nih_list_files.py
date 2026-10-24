"""Cache the full file list of the NIH HF mirror (100k files) so we can map each
filename from BBox_List_2017.csv to its images_XXX/ folder."""
import json, os
from huggingface_hub import HfApi

REPO = "iraqigold/nih-chest-xray-14-flat"
OUT = "data/nih/file_index.json"
os.makedirs("data/nih", exist_ok=True)

if os.path.exists(OUT):
    d = json.load(open(OUT))
    print(f"cached: {len(d)} entries")
else:
    api = HfApi()
    files = []
    for e in api.list_repo_tree(repo_id=REPO, repo_type="dataset", recursive=True):
        p = getattr(e, "path", None)
        if p:
            files.append(p)
    print(f"listed {len(files)} paths")
    png = [f for f in files if f.lower().endswith(".png")]
    print(f"png files: {len(png)}")
    d = {os.path.basename(f): f for f in png}
    json.dump(d, open(OUT, "w"))
    print(f"wrote {OUT}")

# sanity: how many of the boxed images can we locate?
import csv
rows = list(csv.DictReader(open("data/nih/BBox_List_2017.csv", encoding="utf-8-sig")))
names = {r["Image Index"] for r in rows}
found = [n for n in names if n in d]
print(f"boxed images: {len(names)}  locatable in mirror: {len(found)}")
missing = sorted(set(names) - set(found))
if missing:
    print("  missing examples:", missing[:5])

# folder distribution
from collections import Counter
print("folder split:", dict(Counter(p.split('/')[0] for p in list(d.values())[:20000])))
print("example mapping:", dict(list(d.items())[:3]))