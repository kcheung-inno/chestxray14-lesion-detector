"""Build all three dataset layouts for NIH ChestX-ray14 from data/nih/selection.json:
  - data/coco_nih/instances_{train,val}.json          (canonical COCO)
  - data/rfdetr_nih/{train,valid,test}/_annotations.coco.json  (RF-DETR / Roboflow)
  - data/dfine_nih/{train,val}.json                   (D-FINE, 0-indexed category_id)
8 classes (the 8 findings with boxes). Test split = val, as before.
"""
import json, os, shutil

NIH = "data/nih"
IMG = f"{NIH}/images"
COCO = "data/coco_nih"
RFD = "data/rfdetr_nih"
DFI = "data/dfine_nih"
for d in (COCO, RFD, DFI):
    os.makedirs(d, exist_ok=True)

sel = json.load(open(f"{NIH}/selection.json"))
classes = json.load(open(f"{NIH}/classes.json"))
FINDINGS = classes["findings"]
CLS = classes["classes"]                      # name -> 1-indexed id
CATS = [{"id": CLS[f], "name": f, "supercategory": "finding"} for f in FINDINGS]

def build(split):
    imgs, anns = [], []
    aid = 1
    for i, e in enumerate(sel[split], 1):
        p = os.path.join(IMG, e["file"])
        if not os.path.exists(p):
            continue
        from PIL import Image
        with Image.open(p) as im:
            W, H = im.size
        imgs.append({"id": i, "file_name": e["file"], "width": W, "height": H,
                     "patient_id": e.get("patient")})
        for b in e["boxes"]:
            x = max(0.0, b["x"]); y = max(0.0, b["y"])
            w = min(b["w"], W - x); h = min(b["h"], H - y)
            if w < 1 or h < 1:
                continue
            anns.append({"id": aid, "image_id": i, "category_id": b["cls"],
                         "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0})
            aid += 1
    return imgs, anns

doc = {}
for split in ("train", "val"):
    imgs, anns = build(split)
    doc[split] = {"images": imgs, "annotations": anns}
    c = {"info": {"description": "NIH ChestX-ray14 box-annotated subset (BBox_List_2017)",
                  "attribution": "NIH Clinical Center; Wang et al., CVPR 2017; "
                                 "https://nihcc.app.box.com/v/ChestXray-NIHCC"},
         "licenses": [{"id": 1, "name": "No restrictions on use (NIH); attribution required", "url": ""}],
         "categories": CATS, "images": imgs, "annotations": anns}
    json.dump(c, open(f"{COCO}/instances_{split}.json", "w"))
    npos = len({a["image_id"] for a in anns})
    print(f"COCO {split}: {len(imgs)} images ({npos} with boxes), {len(anns)} boxes")

# ---- Roboflow layout for RF-DETR (hardlink images) ----
for src_split, dst_split in (("train", "train"), ("val", "valid"), ("val", "test")):
    outdir = f"{RFD}/{dst_split}"
    os.makedirs(outdir, exist_ok=True)
    remap = {}
    images = []
    for i, im in enumerate(doc[src_split]["images"], 1):
        src_png = os.path.join(IMG, im["file_name"])
        dst_png = os.path.join(outdir, im["file_name"])
        if not os.path.exists(dst_png):
            try:
                os.link(src_png, dst_png)
            except OSError:
                shutil.copy2(src_png, dst_png)
        remap[im["id"]] = i
        images.append({"id": i, "file_name": im["file_name"],
                       "width": im["width"], "height": im["height"]})
    anns = [{"id": j, "image_id": remap[a["image_id"]], "category_id": a["category_id"],
             "bbox": a["bbox"], "area": a["area"], "iscrowd": 0}
            for j, a in enumerate(doc[src_split]["annotations"], 1)]
    json.dump({"info": {"description": "NIH ChestX-ray14 box subset"},
               "licenses": [{"id": 1, "name": "NIH - no restrictions, attribution required", "url": ""}],
               "categories": CATS, "images": images, "annotations": anns},
              open(f"{outdir}/_annotations.coco.json", "w"))
    print(f"rfdetr {dst_split}: {len(images)} images, {len(anns)} boxes")

# ---- D-FINE (0-indexed category_id) ----
for split in ("train", "val"):
    c = json.load(open(f"{COCO}/instances_{split}.json"))
    for a in c["annotations"]:
        a["category_id"] -= 1
    c["categories"] = [{"id": CLS[f] - 1, "name": f, "supercategory": "finding"} for f in FINDINGS]
    json.dump(c, open(f"{DFI}/{split}.json", "w"))
    print(f"dfine {split}: {len(c['images'])} images, {len(c['annotations'])} boxes")

print("\nlayout ready: COCO, RF-DETR, D-FINE")
print("NOTE: negative images (no boxes) are included so the model can learn to reject.")