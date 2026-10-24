"""Build the NIH ChestX-ray14 detection dataset.

Positives : the 880 images in BBox_List_2017.csv (984 radiologist-drawn boxes, 8 findings).
Negatives : images labelled "No Finding" in Data_Entry_2017.csv (sampled).
Split     : BY PATIENT, so the same patient never appears in both train and val.

Why multi-class rather than nodule-only: only 164 of the 984 boxes are Nodule/Mass
(79 Nodule + 85 Mass) across ~150 images. Training on all 8 findings uses ~6x more box
supervision and yields a better localiser; Nodule/Mass is reported as its own class.
"""
import csv, json, os, random
from collections import Counter, defaultdict

SEED = 3407
N_NEG = 900
VAL_FRAC = 0.2
OUT = "data/nih"

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
CLS = {f: i + 1 for i, f in enumerate(FINDINGS)}   # 1-indexed for COCO

# ---- positives: boxes ----
raw = list(csv.reader(open(f"{OUT}/BBox_List_2017.csv", encoding="utf-8-sig")))
hdr, rows = raw[0], raw[1:]
boxes = defaultdict(list)
for r in rows:
    if len(r) < 6 or not r[0].strip():
        continue
    img, finding = r[0].strip(), r[1].strip()
    if finding not in CLS:
        continue
    x, y, w, h = (float(r[2]), float(r[3]), float(r[4]), float(r[5]))
    boxes[img].append({"cls": CLS[finding], "finding": finding, "x": x, "y": y, "w": w, "h": h})

# ---- metadata: patient id + labels ----
meta = {}
for r in csv.DictReader(open(f"{OUT}/Data_Entry_2017.csv", encoding="utf-8-sig")):
    meta[r["Image Index"]] = {"patient": r["Patient ID"], "labels": r["Finding Labels"]}

pos_imgs = sorted(boxes)
no_finding = [k for k, v in meta.items() if v["labels"].strip() == "No Finding"]

# ---- sample negatives (deterministic) ----
rnd = random.Random(SEED)
neg_imgs = rnd.sample(no_finding, min(N_NEG, len(no_finding)))

# ---- split BY PATIENT ----
all_imgs = pos_imgs + neg_imgs
patients = sorted({meta[i]["patient"] for i in all_imgs if i in meta})
rnd.shuffle(patients)
val_patients = set(patients[:max(1, int(len(patients) * VAL_FRAC))])

def split_of(img):
    p = meta.get(img, {}).get("patient")
    return "val" if p in val_patients else "train"

summary = {}
for name, imgs in (("pos", pos_imgs), ("neg", neg_imgs)):
    c = Counter(split_of(i) for i in imgs)
    summary[name] = dict(c)
print("patients:", len(patients), "| val patients:", len(val_patients))
print("positives by split:", dict(Counter(split_of(i) for i in pos_imgs)))
print("negatives by split:", dict(Counter(split_of(i) for i in neg_imgs)))

# per-split box counts (train/val) for the report
for sp in ("train", "val"):
    bc = Counter()
    for i in pos_imgs:
        if split_of(i) == sp:
            for b in boxes[i]:
                bc[b["finding"]] += 1
    print(f"{sp} boxes: total={sum(bc.values())}  nodule={bc['Nodule']}  mass={bc['Mass']}")

findex = json.load(open(f"{OUT}/file_index.json"))
sel = {"train": [], "val": []}
for i in all_imgs:
    sp = split_of(i)
    sel[sp].append({"file": i, "path": findex[i],
                    "positive": i in boxes,
                    "boxes": boxes.get(i, []),
                    "patient": meta.get(i, {}).get("patient"),
                    "labels": meta.get(i, {}).get("labels")})
json.dump(sel, open(f"{OUT}/selection.json", "w"), indent=1)
json.dump({"findings": FINDINGS, "classes": CLS}, open(f"{OUT}/classes.json", "w"), indent=1)
print(f"\nselected: train={len(sel['train'])} val={len(sel['val'])} "
      f"total={len(sel['train'])+len(sel['val'])}")