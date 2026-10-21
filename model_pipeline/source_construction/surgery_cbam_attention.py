"""Phase B: replace the C2PSA attention with a CBAM-style module.

After Phase A the model has no depthwise convolution left in the detection head.
The only two remaining grouped convs are `attn/pe/conv` inside the two PSABlock
attentions.  `Attention.forward` is

    qkv(1x1) -> split q,k,v -> (q^T k)*scale -> softmax -> v @ attn^T
             -> + pe(v)  -> proj(1x1)

No operator in the accelerator's set can express `q^T k`, `softmax` or the
matmul, which is why these two layers are the last PS-side operators and why
they fragment the schedule into 9 range calls.  `cbam_pl.CBAM` replaces all of
that with pooling, convolution and sigmoid.

NOTE this is NOT lossless, unlike Phase A: the trained attention did real work,
so the replacement changes the network and a fine-tune plus a fresh mAP
measurement against the original 0.9942 / 0.8645 is required.  The CBAM is
initialised so its residual contribution is ~-0.018x, i.e. the block output
starts at 0.9817x, so the pretrained features survive initialisation and only
the attention itself has to be relearned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# CBAM must come from an importable module, not be defined here: the checkpoint
# pickles the class by reference, so a __main__-local CBAM makes the saved .pt
# unloadable everywhere else ("Can't get attribute 'CBAM' on __main__").
from cbam_pl import CBAM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def flat(o):
    """Collect tensors from a nested dict/list/tuple model output."""
    out = []
    if isinstance(o, torch.Tensor):
        out.append(o)
    elif isinstance(o, (list, tuple)):
        for v in o:
            out.extend(flat(v))
    elif isinstance(o, dict):
        for v in o.values():
            out.extend(flat(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace attention with CBAM candidate")
    parser.add_argument("--src", type=Path, default=ROOT / "build/model_source/pl_native.pt")
    parser.add_argument("--dst", type=Path, default=ROOT / "build/model_source/pl_only.pt")
    args = parser.parse_args()
    if not args.src.is_file():
        raise FileNotFoundError(f"run surgery_pl_native_head.py first: {args.src}")

    ck = torch.load(str(args.src), map_location="cpu", weights_only=False)
    model = ck["model"].float().eval()

    targets = [(n, m) for n, m in model.named_modules()
               if type(m).__name__ == "PSABlock"]
    print(f"PSABlock modules: {len(targets)}")
    if not targets:
        print("ERROR: no PSABlock found -- wrong checkpoint?")
        return

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref_t = [t.clone() for t in flat(model(x))]

    for name, blk in targets:
        dim = blk.attn.qkv.conv.in_channels
        new = CBAM(dim).float()
        new.train(blk.training)
        blk.attn = new
        print(f"  {name:<28} Attention -> CBAM (dim={dim})")

    model.eval()
    with torch.no_grad():
        out_t = [t.clone() for t in flat(model(x))]

    print(f"\n{'tensor':>6}{'max|old-new|':>15}{'old range':>24}")
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref_t, out_t)):
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"{i:>6}{d:>15.4f}   {a.min().item():>10.3f} .. {a.max().item():>9.3f}")
    print(f"\nworst whole-network difference: {worst:.4f}")
    print("  non-zero as expected -- the trained attention did real work, so a")
    print("  fine-tune is required; this is NOT the lossless Phase A swap.")

    print("\ninit check: block output scale relative to its input")
    for name, blk in targets:
        dim = blk.attn.mlp[0].conv.in_channels
        with torch.no_grad():
            t = torch.randn(1, dim, 8, 8)
            a = blk.attn(t)
        print(f"  {name:<28} |x+attn(x)|/|x| = "
              f"{(t + a).abs().mean().item() / t.abs().mean().item():.4f}"
              f"   residual |attn(x)| = {a.abs().mean().item():.5f}")

    groups = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Conv2d) and m.groups > 1]
    print(f"\ngrouped convs remaining: {len(groups)}   <- must be 0")
    for n, m in groups:
        print(f"  {n}  grp={m.groups}")

    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    ck["model"] = model
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(args.dst))
    print(f"saved -> {args.dst}  ({args.dst.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
