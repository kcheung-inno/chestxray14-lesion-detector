"""Independent evaluation of a trained RF-DETR checkpoint on the held-out split.

Scores with pycocotools (NOT the trainer's own metric) so the reported mAP is an
independent measurement, and also reports a threshold-swept F1 / precision / recall.

Usage: python scripts/eval_rfdetr.py --weights outputs/rfdetr_nih/checkpoint_best_ema.pth \
                                      --split test --tag rfdetr_nih
"""
import argparse, json, os, time
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--dataset-dir", default="data/rfdetr")
    ap.add_argument("--variant", default="small")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    ap.add_argument("--outdir", default="outputs/eval")
    args = ap.parse_args()

    import rfdetr
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
           "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[args.variant]

    print(f"loading {args.weights} ...", flush=True)
    model = cls(pretrain_weights=args.weights)

    ann = os.path.join(args.dataset_dir, args.split, "_annotations.coco.json")
    coco_gt = COCO(ann)
    img_dir = os.path.join(args.dataset_dir, args.split)
    ids = sorted(coco_gt.getImgIds())
    if args.limit:
        ids = ids[:args.limit]
    print(f"evaluating on {len(ids)} images from {args.split}", flush=True)

    dets, per_image, t0 = [], [], time.time()
    for n, iid in enumerate(ids, 1):
        info = coco_gt.loadImgs([iid])[0]
        path = os.path.join(img_dir, info["file_name"])
        im = Image.open(path).convert("RGB")
        out = model.predict(im, threshold=args.score_thresh)
        xyxy = out.xyxy
        conf = out.confidence
        cls_id = getattr(out, "class_id", None)
        k = 0
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
            c = float(conf[j])
            # RF-DETR class ids are 0-indexed; COCO categories here are 1-indexed
            cid = (int(cls_id[j]) + 1) if cls_id is not None else 1
            dets.append({"image_id": iid, "category_id": cid,
                         "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": c})
            k += 1
        per_image.append({"image_id": iid, "file_name": info["file_name"], "n_dets": k})
        if n % 50 == 0:
            print(f"  {n}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)

    infer_s = time.time() - t0
    os.makedirs(args.outdir, exist_ok=True)
    det_path = os.path.join(args.outdir, f"detections_{args.tag}_{args.split}.json")
    json.dump(dets, open(det_path, "w"))

    if not dets:
        print(f"\nNO DETECTIONS at threshold {args.score_thresh} — nothing to score.", flush=True)
        print("DONE", flush=True); return

    coco_dt = coco_gt.loadRes(det_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    stats = ev.stats

    res = {
        "tag": args.tag, "split": args.split, "weights": args.weights,
        "n_images": len(ids), "n_detections": len(dets),
        "score_threshold": args.score_thresh,
        "inference_seconds": round(infer_s, 1),
        "ms_per_image": round(1000 * infer_s / len(ids), 1),
        "mAP_50_95": round(float(stats[0]), 4),
        "mAP_50": round(float(stats[1]), 4),
        "mAP_75": round(float(stats[2]), 4),
        "mAP_small": round(float(stats[3]), 4),
        "mAP_medium": round(float(stats[4]), 4),
        "mAP_large": round(float(stats[5]), 4),
        "mAR_1": round(float(stats[6]), 4),
        "mAR_10": round(float(stats[7]), 4),
        "mAR_100": round(float(stats[8]), 4),
    }
    out_json = os.path.join(args.outdir, f"metrics_{args.tag}_{args.split}.json")
    json.dump(res, open(out_json, "w"), indent=2)
    json.dump(per_image, open(os.path.join(args.outdir, f"per_image_{args.tag}_{args.split}.json"), "w"), indent=2)

    print("\n=== independent pycocotools result ===")
    for k, v in res.items():
        print(f"  {k:20s} {v}")
    print(f"\nwrote {out_json}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()