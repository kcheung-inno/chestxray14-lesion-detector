"""Stage the NIH demo images: held-out (val) images with their radiologist gold-standard
boxes from BBox_List_2017.csv.

NIH ChestX-ray14: "There are no restrictions on the use of the NIH chest x-ray images"
(attribution required). These images MAY be redistributed.
"""
import json, os, shutil

NIH = "data/nih"
DST = "demo/nih_images"
os.makedirs(DST, exist_ok=True)

sel = json.load(open(f"{NIH}/selection.json"))
classes = json.load(open(f"{NIH}/classes.json"))
ID2NAME = {v: k for k, v in classes["classes"].items()}

val = sel["val"]

# Deliberately a SPREAD, not a cherry-pick. The model is uneven across findings
# (Cardiomegaly AP 0.395 vs Nodule 0.062), so the demo should show both the classes it
# handles and the ones it does not, including the target finding at its honest strength.
PLAN = [("Cardiomegaly", 3), ("Mass", 2), ("Nodule", 3), ("Pneumothorax", 1)]
picks, used = [], set()
for finding, n in PLAN:
    got = 0
    for e in val:
        if got >= n:
            break
        if e["file"] in used:
            continue
        if any(b["finding"] == finding for b in e["boxes"]):
            picks.append(e); used.add(e["file"]); got += 1
    print(f"  picked {got}/{n} with {finding}")
for e in val:                                  # then negatives
    if len([p for p in picks]) >= 11:
        break
    if not e["boxes"] and e["file"] not in used:
        picks.append(e); used.add(e["file"])
        if len(picks) >= 11:
            break
print(f"val pool: {len(val)} images; staged {len(picks)}")

manifest = {
    "model": "RF-DETR-S trained on NIH ChestX-ray14 (8-class lesion detection)",
    "dataset": "NIH ChestX-ray14 box-annotated subset (BBox_List_2017)",
    "licence": "NIH ChestX-ray14: no restrictions on use; attribution required "
               "(NIH Clinical Center; Wang et al., CVPR 2017; https://nihcc.app.box.com/v/ChestXray-NIHCC)",
    "split": "val (held out; split by patient)",
    "classes": classes["classes"],
    "names": {str(v): k for k, v in classes["classes"].items()},
    "images": [],
}
for e in picks:
    src = os.path.join(NIH, "images", e["file"])
    if not os.path.exists(src):
        continue
    shutil.copyfile(src, os.path.join(DST, e["file"]))
    manifest["images"].append({
        "file": e["file"], "positive": bool(e["boxes"]),
        "boxes": [{"cls": b["cls"], "name": b["finding"],
                   "x": round(b["x"], 1), "y": round(b["y"], 1),
                   "w": round(b["w"], 1), "h": round(b["h"], 1)} for b in e["boxes"]],
    })

json.dump(manifest, open("demo/nih_manifest.json", "w"), indent=2)
print(f"staged {len(manifest['images'])} images")
for m in manifest["images"]:
    names = sorted({b["name"] for b in m["boxes"]})
    print(f"  {m['file']}  boxes={len(m['boxes'])}  {names if names else 'no boxes'}")
tot = sum(os.path.getsize(os.path.join(DST, m["file"])) for m in manifest["images"])
print(f"total: {tot/1e6:.1f} MB")