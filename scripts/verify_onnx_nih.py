"""Determine the ONNX class-index convention for the NIH model by comparing against the
PyTorch model's own class_id output on a real image. Must not guess: the demo labels boxes
with class names, and an off-by-one would print the wrong finding."""
import json, os
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

ONNX = "demo/model_nih/rfdetr_nih.onnx"
RES = 576
MEAN = np.array([0.485,0.456,0.406], np.float32)
STD = np.array([0.229,0.224,0.225], np.float32)

man = json.load(open("demo/nih_manifest.json"))
names = man["names"]
print("manifest class names:", names)

# pick a demo image that HAS a Nodule box
target = next(m for m in man["images"] if any(b["name"] == "Nodule" for b in m["boxes"]))
gt_names = sorted({b["name"] for b in target["boxes"]})
print(f"\nimage: {target['file']}  GT boxes: {gt_names}")

pil = Image.open("demo/nih_images/" + target["file"]).convert("RGB")

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
print("\nONNX outputs:")
for o in sess.get_outputs():
    print(f"  {o.name} shape={o.shape}")

a = np.asarray(pil.resize((RES, RES), Image.BILINEAR), np.float32) / 255
a = ((a - MEAN) / STD).transpose(2, 0, 1)[None]
dets, labels = sess.run(None, {"input": a.astype(np.float32)})
NC = labels.shape[2]
print(f"\nlabels shape {labels.shape} -> {NC} class logits per query")

def report(sig, tag):
    order = np.argsort(-sig)[:6]
    print(f"\n--- top-6 by {tag} ---")
    for j in order:
        cx, cy, w, h = dets[0, j]
        print(f"  score={sig[j]:.3f} idx={j} box_norm=[{cx:.3f},{cy:.3f},{w:.3f},{h:.3f}]")

sig = 1 / (1 + np.exp(-labels[0]))
report(sig.max(axis=1), "max-sigmoid across classes (query level)")

# per-class max, so we can see which index carries a real detection
print("\n--- per-index max sigmoid over all 300 queries ---")
for c in range(NC):
    print(f"  index {c:2d} (name '{names.get(str(c)) or names.get(str(c+1)) or '?'}'): max={sig[:,c].max():.3f}")

print("\n--- PyTorch reference (class ids it reports) ---")
import rfdetr
m = rfdetr.RFDETRSmall(pretrain_weights="outputs/rfdetr_nih/checkpoint_best_ema.pth")
out = m.predict(pil, threshold=0.25)
print(f"  n detections: {len(out.xyxy)}")
cid = getattr(out, "class_id", None)
for i in range(min(len(out.xyxy), 8)):
    b = out.xyxy[i]
    print(f"  score={float(out.confidence[i]):.3f} class_id={None if cid is None else int(cid[i])} "
          f"box=[{float(b[0]):.0f},{float(b[1]):.0f},{float(b[2]-b[0]):.0f},{float(b[3]-b[1]):.0f}]")
print("\nGT boxes:", [(b["name"], round(b["x"]), round(b["y"]), round(b["w"]), round(b["h"]))
                      for b in target["boxes"]])