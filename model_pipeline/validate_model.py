"""Measure the 220-image YOLO26 Conv1x1 candidate before HLS export."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[1] / "training" / "dataset.example.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=HERE / "model_conv1x1.pt")
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    core = model.model
    conv_attention = [(n, m) for n, m in core.named_modules()
                      if n.endswith(".attn") and isinstance(m, nn.Conv2d)]
    grouped = [(n, m.groups) for n, m in core.named_modules()
               if isinstance(m, nn.Conv2d) and m.groups > 1]
    if len(conv_attention) != 2 or grouped:
        raise RuntimeError(f"wrong architecture: conv_attention={len(conv_attention)}, grouped={grouped}")
    if any(type(m).__name__ == "CBAM" for m in core.modules()):
        raise RuntimeError("CBAM remained in saved checkpoint")

    result = model.val(data=str(DATA), imgsz=224, batch=32,
                       device=0 if torch.cuda.is_available() else "cpu",
                       workers=0, plots=False, verbose=False)
    record = {
        "weights": str(args.weights),
        "val_images": 220,
        "conv1x1_attention_blocks": len(conv_attention),
        "grouped_convs": len(grouped),
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
    }
    (HERE / f"validation_{args.weights.stem}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
