"""Replace YOLO26's two CBAM modules with plain 1x1 residual convolutions.

PSABlock computes x + attn(x). A zero-initialised Conv2d therefore starts as
an exact identity at each block. Its weights remain trainable. This architecture
uses only the existing PL convolution and Add operators; it needs no CBAM pool,
sigmoid, or broadcast-multiply hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import cbam_pl  # noqa: F401 -- required to unpickle the source checkpoint


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=HERE / "model_cbam_source.pt")
    parser.add_argument("--dst", type=Path, default=HERE / "model_conv1x1.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    model = ckpt["model"].float().eval()
    blocks = [(name, block) for name, block in model.named_modules()
              if type(block).__name__ == "PSABlock" and
              type(block.attn).__name__ == "CBAM"]
    if len(blocks) != 2:
        raise RuntimeError(f"expected two CBAM PSABlocks, found {len(blocks)}")

    for name, block in blocks:
        channels = block.attn.mlp[0].conv.in_channels
        residual = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(residual.weight)
        residual.eval()
        block.attn = residual
        print(f"{name}: CBAM -> zero-initialised Conv1x1 ({channels} channels)")

    grouped = [(name, layer.groups) for name, layer in model.named_modules()
               if isinstance(layer, nn.Conv2d) and layer.groups != 1]
    if grouped:
        raise RuntimeError(f"grouped convs remain: {grouped}")
    if any(type(m).__name__ == "CBAM" for m in model.modules()):
        raise RuntimeError("CBAM remains in the deployment model")

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    if output is None:
        raise RuntimeError("smoke forward returned no output")

    ckpt["model"] = model
    ckpt["ema"] = None
    ckpt["optimizer"] = None
    ckpt["best_fitness"] = None
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"saved {args.dst} ({args.dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
