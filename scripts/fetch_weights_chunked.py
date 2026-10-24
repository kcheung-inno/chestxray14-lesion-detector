"""Fetch RF-DETR pretrained weights in bounded chunks.
A single long connection repeatedly died at the same offset (~313 MB), which looks like a
transfer-size limit on this network; small ranged chunks avoid it."""
import os, time, requests

DEST = os.path.expanduser("~/.roboflow/models")
os.makedirs(DEST, exist_ok=True)
CHUNK = 8 << 20  # 8 MB

JOBS = [
    ("nano",  "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
              os.path.join(DEST, "rf-detr-nano.pth")),
    ("small", "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
              os.path.join(DEST, "rf-detr-small.pth")),
]

def size_of(url):
    r = requests.head(url, timeout=30, allow_redirects=True)
    return int(r.headers.get("Content-Length", 0))

for name, url, path in JOBS:
    try:
        total = size_of(url)
    except Exception as e:
        print(f"{name}: HEAD failed {type(e).__name__}", flush=True); continue
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have > total:                      # corrupt from a previous partial run
        os.remove(path); have = 0
    print(f"{name}: {total/1e6:.1f} MB target, have {have/1e6:.1f} MB", flush=True)
    if have == total:
        print(f"{name}: complete", flush=True); continue
    with open(path, "r+b" if have else "wb") as f:
        pos = have
        f.seek(pos)
        stall = 0
        while pos < total and stall < 12:
            end = min(pos + CHUNK - 1, total - 1)
            try:
                t = time.time()
                r = requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, timeout=45)
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                data = r.content
                if not data:
                    raise RuntimeError("empty body")
                f.write(data); f.flush()
                pos += len(data)
                if pos % (64 << 20) < CHUNK:
                    print(f"  {name}: {pos/1e6:.0f}/{total/1e6:.0f} MB "
                          f"({len(data)/1e6/max(time.time()-t,1e-6):.1f} MB/s)", flush=True)
            except Exception as e:
                stall += 1
                print(f"  {name}: chunk at {pos/1e6:.1f} MB failed ({type(e).__name__}), retry {stall}", flush=True)
                time.sleep(1.5)
    final = os.path.getsize(path)
    print(f"{name}: FINAL {final/1e6:.1f}/{total/1e6:.1f} MB {'OK' if final == total else 'INCOMPLETE'}", flush=True)
print("DONE", flush=True)