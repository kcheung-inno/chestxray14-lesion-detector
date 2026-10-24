"""Robustly fetch RF-DETR pretrained weights (GCS drops connections mid-transfer;
resume with Range + retry instead of letting the library fail)."""
import os, time, requests

DEST = os.path.expanduser("~/.roboflow/models")
os.makedirs(DEST, exist_ok=True)

JOBS = [
    ("nano",  "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
              os.path.join(DEST, "rf-detr-nano.pth")),
    ("small", "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
              os.path.join(DEST, "rf-detr-small.pth")),
]

def expected_size(url):
    r = requests.head(url, timeout=30, allow_redirects=True)
    return int(r.headers.get("Content-Length", 0))

for name, url, path in JOBS:
    try:
        total = expected_size(url)
    except Exception as e:
        print(f"{name}: HEAD failed {type(e).__name__}", flush=True); continue
    print(f"{name}: expected {total/1e6:.1f} MB -> {path}", flush=True)
    if os.path.exists(path) and os.path.getsize(path) == total:
        print(f"{name}: already complete, skipping", flush=True); continue
    for attempt in range(1, 9):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            t = time.time()
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                mode = "ab" if have else "wb"
                with open(path, mode) as f:
                    for chunk in r.iter_content(1 << 18):
                        f.write(chunk)
            now = os.path.getsize(path)
            print(f"{name}: attempt {attempt} -> {now/1e6:.1f}/{total/1e6:.1f} MB "
                  f"({(now-have)/1e6/max(time.time()-t,1e-6):.2f} MB/s)", flush=True)
        except Exception as e:
            now = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"{name}: attempt {attempt} broke at {now/1e6:.1f} MB ({type(e).__name__}), resuming", flush=True)
            time.sleep(2)
    final = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"{name}: FINAL {final/1e6:.1f}/{total/1e6:.1f} MB {'OK' if final == total else 'INCOMPLETE'}", flush=True)
print("DONE", flush=True)