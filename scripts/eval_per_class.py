"""Per-class AP for a COCO detections file, so the Nodule/Mass numbers are visible
rather than hidden inside an 8-class average.

Usage: python scripts/eval_per_class.py --gt <coco> --det <detections> --tag <name>
"""
import argparse, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--det", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--outdir", default="outputs/eval")
args = ap.parse_args()

coco_gt = COCO(args.gt)
coco_dt = coco_gt.loadRes(args.det)
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate(); ev.accumulate()

cats = sorted(coco_gt.getCatIds())
names = {c["id"]: c["name"] for c in coco_gt.loadCats(cats)}

# how many GT boxes per class (needed to judge how trustworthy each AP is)
gt_counts = {c: 0 for c in cats}
for a in coco_gt.dataset["annotations"]:
    gt_counts[a["category_id"]] += 1

rows = []
for i, c in enumerate(cats):
    # [T, R, K, A, M] -> all IoU thresholds, all recall, this class, all areas, maxDets=100
    p = ev.eval["precision"][:, :, i, 0, 2]
    p = p[p > -1]
    ap5095 = float(np.mean(p)) if p.size else float("nan")
    p50 = ev.eval["precision"][0, :, i, 0, 2]; p50 = p50[p50 > -1]
    p75 = ev.eval["precision"][5, :, i, 0, 2]; p75 = p75[p75 > -1]
    rows.append({
        "class": names[c], "gt_boxes": gt_counts[c],
        "AP_50_95": round(ap5095, 4) if p.size else None,
        "AP_50": round(float(np.mean(p50)), 4) if p50.size else None,
        "AP_75": round(float(np.mean(p75)), 4) if p75.size else None,
    })

print(f"=== per-class AP — {args.tag} ===")
print(f"{'class':16s} {'GT boxes':>9} {'AP@50:95':>10} {'AP@50':>8} {'AP@75':>8}")
for r in sorted(rows, key=lambda r: -(r["AP_50_95"] or 0)):
    print(f"{r['class']:16s} {r['gt_boxes']:>9} "
          f"{('-' if r['AP_50_95'] is None else format(r['AP_50_95'],'.4f')):>10} "
          f"{('-' if r['AP_50'] is None else format(r['AP_50'],'.4f')):>8} "
          f"{('-' if r['AP_75'] is None else format(r['AP_75'],'.4f')):>8}")

out = f"{args.outdir}/per_class_{args.tag}.json"
json.dump({"tag": args.tag, "gt": args.gt, "det": args.det, "per_class": rows}, open(out, "w"), indent=2)
print(f"\nwrote {out}")

for r in rows:
    if r["class"] in ("Nodule", "Mass"):
        print(f"\n** {r['class']}: AP@50:95 = {r['AP_50_95']} on {r['gt_boxes']} GT boxes **")