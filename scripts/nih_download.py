"""Download the selected NIH images (per-file, with retries)."""
import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

REPO = "iraqigold/nih-chest-xray-14-flat"
DEST = "data/nih/images"
os.makedirs(DEST, exist_ok=True)

sel = json.load(open("data/nih/selection.json"))
items = [(s, e) for s in ("train", "val") for e in sel[s]]
todo = [(s, e) for s, e in items if not os.path.exists(os.path.join(DEST, e["file"]))]
print(f"total={len(items)} present={len(items)-len(todo)} to_fetch={len(todo)}", flush=True)

def fetch(item):
    s, e = item
    return hf_hub_download(REPO, e["path"], repo_type="dataset", local_dir="data/nih/_hf")

fails = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, it): it for it in todo}
    for i, fut in enumerate(as_completed(futs), 1):
        it = futs[fut]
        try:
            p = fut.result()
            dst = os.path.join(DEST, it[1]["file"])
            if not os.path.exists(dst):
                os.replace(p, dst)          # flatten into images/ for convenience
        except Exception as e:
            fails.append(it)
            if len(fails) <= 5:
                print("  fail", it[1]["file"], type(e).__name__, flush=True)
        if i % 200 == 0:
            print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s  fails={len(fails)}", flush=True)

have = sum(1 for _, e in items if os.path.exists(os.path.join(DEST, e["file"])))
print(f"\nFINAL present={have}/{len(items)} fails={len(fails)} in {time.time()-t0:.0f}s", flush=True)

# size / resolution check on a few
from PIL import Image
sample = [e["file"] for _, e in items[:6] if os.path.exists(os.path.join(DEST, e["file"]))]
for f in sample:
    p = os.path.join(DEST, f)
    with Image.open(p) as im:
        print(f"  {f}  {im.size} {im.mode}  {os.path.getsize(p)/1e3:.0f} KB", flush=True)
tot = sum(os.path.getsize(os.path.join(DEST, f)) for f in os.listdir(DEST))
print(f"images dir total: {tot/1e9:.2f} GB", flush=True)
print("DONE", flush=True)