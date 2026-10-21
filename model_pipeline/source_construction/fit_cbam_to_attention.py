"""Fit CBAM to the trained C2PSA attention -- distillation, not retraining.

THE IDEA
--------
Phase A (head depthwise -> standard 3x3 conv, diagonal init) was proven exactly
lossless, so the trained checkpoint's features survive it untouched.  Only the
attention had to go, because `q^T k` / softmax / matmul are inexpressible in the
accelerator's op set.

Rather than retrain the network -- which throws away the trained features and
costs 150 epochs -- fit ONLY the two CBAM blocks to reproduce what the original
attention produced.  Everything else stays frozen at the trained values.

This works because CBAM occupies the same slot: `PSABlock.forward` is
`x = x + self.attn(x)` for both the original Attention and the replacement, so
the fit target is literally the original module's raw output, with no reshaping
or rescaling needed.

WHAT THE FIT CAN AND CANNOT DO
-----------------------------
The original is `proj(v @ softmax(q^T k / sqrt(d)) + pe(v))`: content-dependent,
spatially global, with a learned positional term.  CBAM is a channel gate times a
spatial gate, `x * ca * sa`.  These are different function families, so the fit
is an approximation and the honest measure of success is downstream mAP, not the
fitting loss.  The loss is still worth watching: if CBAM cannot get near the
target even on the training pairs, the architecture is the problem, not the
optimisation.

Inputs for the fit come from the REAL distribution: images are pushed through the
original model and the two attention modules' actual (input, output) pairs are
captured with forward hooks.  Fitting random tensors would train the gate on
activations the network never produces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbam_pl import CBAM  # noqa: E402,F401  (unpickling the pl_only checkpoint)

ROOT = Path(__file__).resolve().parents[2]

ATTN_PATHS = ["model.10.m.0.attn", "model.22.m.0.1.attn"]
IMGSZ = 224


def letterbox_rgb(bgr: np.ndarray) -> np.ndarray:
    """Same geometry the deployment and training use: pad 114, min-scale."""
    h, w = bgr.shape[:2]
    sc = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(w * sc + 0.5), int(h * sc + 0.5)
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    c = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    x0, y0 = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    c[y0:y0 + nh, x0:x0 + nw] = r
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=240)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--orig", type=Path, default=ROOT / "runs_yolo26/yolo26n_224/weights/best.pt")
    ap.add_argument("--pl-only", type=Path, default=ROOT / "build/model_source/pl_only.pt")
    ap.add_argument("--train-images", type=Path, default=ROOT / "datasets/carotid/images/train")
    ap.add_argument("--dst", type=Path, default=ROOT / "build/model_source/model_cbam_source.pt")
    args = ap.parse_args()

    for p in (args.orig, args.pl_only):
        if not p.is_file():
            raise FileNotFoundError(p)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. capture the real attention inputs and outputs ---------------
    orig = torch.load(str(args.orig), map_location="cpu", weights_only=False)["model"]
    orig = orig.float().eval().to(dev)

    caps = {p: {"in": [], "out": []} for p in ATTN_PATHS}

    def mk_hook(path):
        def hook(mod, inp, out):
            caps[path]["in"].append(inp[0].detach().float().cpu())
            caps[path]["out"].append(out.detach().float().cpu())
        return hook

    handles = []
    for p in ATTN_PATHS:
        mod = orig.get_submodule(p)
        handles.append(mod.register_forward_hook(mk_hook(p)))
        print(f"hooked {p}: {type(mod).__name__}")

    files = sorted(args.train_images.glob("*.png")) + sorted(args.train_images.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no images in {args.train_images}")
    step = max(1, len(files) // args.images)
    files = files[::step][:args.images]
    print(f"capturing from {len(files)} images ...", flush=True)
    with torch.no_grad():
        for f in files:
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            orig(to_tensor(letterbox_rgb(bgr)).to(dev))
    for h in handles:
        h.remove()
    for p in ATTN_PATHS:
        caps[p]["in"] = torch.cat(caps[p]["in"]).float()
        caps[p]["out"] = torch.cat(caps[p]["out"]).float()
        print(f"  {p}: {tuple(caps[p]['in'].shape)} -> "
              f"{tuple(caps[p]['out'].shape)}   "
              f"|out| mean {caps[p]['out'].abs().mean():.4f}")

    # ---- 2. fit each CBAM to its captured pairs ------------------------
    ck = torch.load(str(args.pl_only), map_location="cpu", weights_only=False)
    model = ck["model"].float().eval()

    report = {}
    for p in ATTN_PATHS:
        blk = model.get_submodule(p.rsplit(".", 1)[0])
        cbam = blk.attn
        assert isinstance(cbam, CBAM), f"{p} is {type(cbam).__name__}, not CBAM"
        cbam = cbam.float().to(dev)
        blk.attn = cbam

        X = caps[p]["in"].to(dev)
        Y = caps[p]["out"].to(dev)
        n = X.shape[0]
        ntr = int(n * 0.9)
        Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

        # freeze everything in the CBAM except the learned convs' parameters,
        # and freeze the BN running stats so eval-time behaviour is stable
        for m in cbam.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                for t in m.parameters():
                    t.requires_grad_(True)
        params = [t for t in cbam.parameters() if t.requires_grad]

        opt = torch.optim.Adam(params, lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

        # baseline: the frozen init's error, for reference
        cbam.eval()
        with torch.no_grad():
            base = ((cbam(Xte) - Yte) ** 2).mean().item()
        print(f"\n{p}: {len(params)} tensors, init MSE {base:.6f}  "
              f"(|Y| mean {Y.abs().mean():.4f})", flush=True)

        cbam.train()
        for m in cbam.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()          # keep BN deterministic during the fit
        for it in range(args.steps):
            idx = torch.randint(0, ntr, (args.batch,), device=dev)
            pred = cbam(Xtr[idx])
            loss = ((pred - Ytr[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (it + 1) % 250 == 0:
                cbam.eval()
                with torch.no_grad():
                    te = ((cbam(Xte) - Yte) ** 2).mean().item()
                cbam.train()
                for m in cbam.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()
                print(f"    step {it + 1:>5}  train {loss.item():.6f}  "
                      f"holdout {te:.6f}", flush=True)

        cbam.eval()
        with torch.no_grad():
            te = ((cbam(Xte) - Yte) ** 2).mean().item()
            rel = (cbam(Xte) - Yte).norm() / Yte.norm()
        report[p] = {"init_mse": base, "fit_mse": te, "rel_err": float(rel)}
        print(f"  -> holdout MSE {te:.6f}  relative err {float(rel):.4f}")
        blk.attn = cbam.cpu()
        model.get_submodule(p.rsplit(".", 1)[0]).attn = blk.attn

    model.eval()
    ck["model"] = model
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(args.dst))
    print(f"\nsaved -> {args.dst}  ({args.dst.stat().st_size:,} B)")
    out = args.dst.with_suffix(".fit.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"report -> {out}")
    print("\nNOTE: a low fitting loss does NOT prove the rework is harmless --")
    print("the original attention and CBAM are different function families, so")
    print("the real verdict is downstream mAP.  Run val next.")


if __name__ == "__main__":
    main()
