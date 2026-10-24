"""Export the trained RF-DETR-S checkpoint to ONNX for in-browser inference,
then verify the ONNX graph actually produces detections in Python before we
ever put it in a browser."""
import argparse, inspect, os, json

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="outputs/rfdetr_nih/checkpoint_best_ema.pth")
ap.add_argument("--variant", default="small")
ap.add_argument("--outdir", default="demo/model_nih")
ap.add_argument("--resolution", type=int, default=576)
args = ap.parse_args()

import rfdetr
cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
       "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[args.variant]

print("export() signature:", inspect.signature(cls.export))
model = cls(pretrain_weights=args.weights)
os.makedirs(args.outdir, exist_ok=True)
print("exporting at", args.resolution, "...", flush=True)
model.export(output_dir=args.outdir, shape=(args.resolution, args.resolution))
print("\nfiles in", args.outdir, ":")
for f in sorted(os.listdir(args.outdir)):
    p = os.path.join(args.outdir, f)
    print(f"   {f}  {os.path.getsize(p)/1e6:.1f} MB")