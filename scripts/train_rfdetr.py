"""Train an RF-DETR detector on a Roboflow-layout COCO dataset.

Usage:
  python scripts/train_rfdetr.py --variant small --epochs 30 --batch 2 --resolution 576 \
      --dataset-dir data/rfdetr_nih --output outputs/rfdetr_nih

Dataset: <dataset-dir>/{train,valid,test}/_annotations.coco.json (Roboflow COCO layout).
The class count is read from the dataset, so the same script works for any number of findings.
"""
import argparse, json, os, time, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="small", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=576)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset-dir", default="data/rfdetr")
    args = ap.parse_args()

    import rfdetr
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
           "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[args.variant]

    print(f"instantiating RFDETR-{args.variant} ...", flush=True)
    model = cls()
    mc = getattr(model, "model_config", None)
    if mc is not None:
        print(f"model_config: resolution={getattr(mc,'resolution',None)} "
              f"patch_size={getattr(mc,'patch_size',None)} "
              f"num_windows={getattr(mc,'num_windows',None)}", flush=True)

    os.makedirs(args.output, exist_ok=True)
    run_cfg = {
        "variant": args.variant, "epochs": args.epochs, "batch_size": args.batch,
        "grad_accum_steps": args.grad_accum, "resolution": args.resolution, "lr": args.lr,
        "dataset_dir": args.dataset_dir, "output_dir": args.output,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(run_cfg, open(os.path.join(args.output, "run_config.json"), "w"), indent=2)
    print("run_config:", json.dumps(run_cfg), flush=True)

    t0 = time.time()
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        resolution=args.resolution,
        lr=args.lr,
        output_dir=args.output,
        num_workers=args.workers,
        tensorboard=False,
        wandb=False,
        run_test=False,
        progress_bar=True,
    )
    dt = time.time() - t0
    run_cfg["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_cfg["wall_seconds"] = round(dt, 1)
    run_cfg["seconds_per_epoch"] = round(dt / max(args.epochs, 1), 1)
    json.dump(run_cfg, open(os.path.join(args.output, "run_config.json"), "w"), indent=2)
    print(f"\nTRAINED in {dt/60:.1f} min ({dt/max(args.epochs,1):.1f} s/epoch)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()