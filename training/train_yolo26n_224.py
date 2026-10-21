"""Train and validate the 224-input YOLO26n model on an authorized dataset.

Configure training/dataset.example.yaml before running. The intended split has
880 training and 220 validation images with matching labels. A second model
may be passed through --models only for a same-split controlled comparison.
Results are written to runs_yolo26/ under the repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "dataset.example.yaml"
PROJECT = ROOT / "runs_yolo26"


def train_one(name: str, weights: Path, args) -> dict:
    print("=" * 78)
    print(f"=== {name}  ({weights.name})  imgsz={args.imgsz}  epochs={args.epochs}")
    print("=" * 78, flush=True)
    if not weights.exists():
        print(f"!! missing weights {weights}", flush=True)
        return {"name": name, "error": "missing weights"}

    model = YOLO(str(weights))
    t0 = time.time()
    device = 0 if torch.cuda.is_available() else "cpu"
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,          # Windows: 0 avoids the dataloader multiprocessing traps
        project=str(PROJECT),
        name=f"{name}_{args.imgsz}",
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        val=True,
        plots=True,
        # medical ultrasound: NO mosaic (it stitches unrelated anatomy together),
        # gentle geometric augmentation with preserved ultrasound orientation.
        mosaic=0.0,
        close_mosaic=0,
        fliplr=0.5,
        flipud=0.0,         # ultrasound frames are not up/down symmetric
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,          # keep it grayscale-faithful
        hsv_s=0.0,
        hsv_v=0.2,
        amp=True,
        verbose=True,
    )
    dt = time.time() - t0

    # re-run val() on the best weights so we get a clean metrics dict to record
    best = PROJECT / f"{name}_{args.imgsz}" / "weights" / "best.pt"
    rec = {"name": name, "weights": str(weights), "imgsz": args.imgsz,
           "epochs_requested": args.epochs, "seconds": round(dt, 1),
           "best_pt": str(best), "best_pt_exists": best.exists()}
    if best.exists():
        m = YOLO(str(best)).val(data=str(DATA), imgsz=args.imgsz,
                                batch=args.batch, device=device,
                                workers=0, plots=True,
                                project=str(PROJECT),
                                name=f"{name}_{args.imgsz}_val")
        try:
            rec["metrics"] = {k: float(v) for k, v in m.results_dict.items()}
        except Exception:
            rec["metrics"] = {"map50": float(m.box.map50), "map": float(m.box.map)}
        rec["speed_ms"] = {k: float(v) for k, v in getattr(m, "speed", {}).items()}
    # also read the best mAP50 out of results.csv (covers the early-stop case)
    csvp = PROJECT / f"{name}_{args.imgsz}" / "results.csv"
    if csvp.exists():
        import csv as _csv
        best_m, best_ep = None, None
        with csvp.open() as fh:
            for row in _csv.DictReader(fh):
                key = [c for c in row if "mAP50(B)" in c]
                if not key:
                    continue
                try:
                    v = float(row[key[0]])
                except ValueError:
                    continue
                if best_m is None or v > best_m:
                    best_m, best_ep = v, row.get("epoch")
        rec["best_map50_train"] = best_m
        rec["best_map50_epoch"] = best_ep
    print(f"--- {name} done in {dt/60:.1f} min: {json.dumps(rec.get('metrics', {}), indent=None)}",
          flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo26n"],
                    help="which model to train (default: yolo26n)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)
    print(f"data   : {DATA}")
    print(f"project: {PROJECT}")
    for p in (DATA, *(ROOT / f"{name}.pt" for name in args.models)):
        print(f"  {'OK ' if p.exists() else 'MISSING'} {p}")
        if not p.exists():
            raise FileNotFoundError(f"required training input missing: {p}")

    PROJECT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.models:
        w = ROOT / f"{name}.pt"
        records.append(train_one(name, w, args))
        # write incrementally so a crash later still leaves the earlier results
        (PROJECT / "summary.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print("SUMMARY  (imgsz=%d, %s)" % (args.imgsz, DATA.name))
    print("=" * 78)
    hdr = f"{'model':<12}{'best mAP50':>12}{'mAP50-95':>11}{'precision':>11}{'recall':>9}{'infer ms':>10}"
    print(hdr)
    for r in records:
        m = r.get("metrics", {})
        g = lambda *ks: next((m[k] for k in ks if k in m), float("nan"))
        print(f"{r['name']:<12}{g('metrics/mAP50(B)','map50'):>12.4f}"
              f"{g('metrics/mAP50-95(B)','map'):>11.4f}"
              f"{g('metrics/precision(B)'):>11.4f}{g('metrics/recall(B)'):>9.4f}"
              f"{r.get('speed_ms', {}).get('inference', float('nan')):>10.2f}")
    print(f"\nwrote {PROJECT/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
