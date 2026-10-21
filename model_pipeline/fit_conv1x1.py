"""Fit a single 1x1 residual conv to each trained CBAM on real train images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import cbam_pl  # noqa: F401 -- checkpoint unpickling


HERE = Path(__file__).resolve().parent
TRAIN = Path(__file__).resolve().parents[1] / "datasets" / "carotid" / "images" / "train"
TARGETS = ("model.10.m.0.attn", "model.22.m.0.1.attn")


def letterbox(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path))
    h, w = bgr.shape[:2]
    scale = min(224 / w, 224 / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = np.full((224, 224, 3), 114, np.uint8)
    x, y = (224 - nw) // 2, (224 - nh) // 2
    rgb[y:y + nh, x:x + nw] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--src", type=Path, default=HERE / "model_cbam_source.pt")
    parser.add_argument("--conv-src", type=Path, default=HERE / "model_conv1x1.pt")
    parser.add_argument("--train-images", type=Path, default=TRAIN)
    parser.add_argument("--dst", type=Path, default=HERE / "model_conv1x1_fit.pt")
    args = parser.parse_args()
    src = torch.load(args.src, map_location="cpu", weights_only=False)
    model = src["model"].float().eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    stats = {p: {"xx": torch.zeros(129, 129, dtype=torch.float64),
                 "yx": torch.zeros(128, 129, dtype=torch.float64)} for p in TARGETS}

    def hook(path):
        def collect(_module, inputs, output):
            x = inputs[0].detach().double().cpu().permute(1, 0, 2, 3).reshape(128, -1)
            y = output.detach().double().cpu().permute(1, 0, 2, 3).reshape(128, -1)
            xa = torch.cat((x, torch.ones(1, x.shape[1], dtype=torch.float64)))
            stats[path]["xx"] += xa @ xa.T
            stats[path]["yx"] += y @ xa.T
        return collect

    handles = [model.get_submodule(p).register_forward_hook(hook(p)) for p in TARGETS]
    files = sorted(args.train_images.glob("*.png"),
                   key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
    if not files:
        raise FileNotFoundError(f"no PNG training images in {args.train_images}")
    picks = [files[round(i * (len(files) - 1) / max(args.images - 1, 1))]
             for i in range(args.images)]
    with torch.no_grad():
        for i, path in enumerate(picks, 1):
            model(letterbox(path).to(device))
            if i % 16 == 0: print(f"captured {i}/{len(picks)}", flush=True)
    for h in handles: h.remove()

    ckpt = torch.load(args.conv_src, map_location="cpu", weights_only=False)
    target = ckpt["model"].float().eval()
    for path in TARGETS:
        xx, yx = stats[path]["xx"], stats[path]["yx"]
        ridge = xx.diag().mean() * 1e-5
        xx[:128, :128] += torch.eye(128, dtype=torch.float64) * ridge
        beta = torch.linalg.solve(xx, yx.T).T.float()
        parent = target.get_submodule(path.rsplit(".", 1)[0])
        conv = nn.Conv2d(128, 128, 1, bias=True)
        with torch.no_grad():
            conv.weight.copy_(beta[:, :128, None, None])
            conv.bias.copy_(beta[:, 128])
        conv.eval()
        parent.attn = conv
        print(f"{path}: residual fit mean|W|={conv.weight.abs().mean().item():.5g}")
    ckpt["model"] = target
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"saved {args.dst}")


if __name__ == "__main__":
    main()
