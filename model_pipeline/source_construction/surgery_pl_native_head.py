"""Phase A: make the head depthwise convs PL-native, losing nothing.

WHY THIS IS POSSIBLE WITHOUT RETRAINING
---------------------------------------
The head is built as `cv3.x.y = Sequential(DWConv, Conv)` -- depthwise 3x3 then
pointwise 1x1.  A depthwise conv computes

    out[o] = sum_{3x3} w[o,0,:,:] * in[o]

and a standard conv computes

    out[o] = sum_i sum_{3x3} W[o,i,:,:] * in[i]

Setting W[o,i] = 0 for i != o and W[o,o] = w[o,0] makes the standard conv
EXACTLY the depthwise one.  So the replacement can be initialised to reproduce
the trained model bit-for-bit: no fine-tuning is required for correctness, and
the network keeps the original accuracy while the weights stay free to learn
off-diagonal terms later if we choose to fine-tune.

WHY IT IS WORTH IT
------------------
Those 6 depthwise convs are 6 of the 8 PS-side operators and are what fragments
the PL schedule into 9 range calls (`[101,102)` is a single layer).  Replacing
them costs roughly +8% MACs and removes 6 PS round trips plus 6 schedule
fragments -- and the PL is far more efficient per MAC than the PS.  This needs
NO HLS change: the accelerator already does standard 3x3 convolutions.

The 2 remaining depthwise convs live inside the C2PSA attention's `pe` branch
and are handled by the C2PSA -> CBAM work (Phase B); they vanish with the
attention itself.

The script refuses to claim success without proving it: it runs both models on
the same input and compares outputs, because a diagonal-init bug that silently
changed the network would stay invisible until it reached the board.
"""

from __future__ import annotations

import copy
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]


def dwconv_to_conv(dw) -> nn.Module:
    """Standard Conv whose weight exactly reproduces this DWConv."""
    from ultralytics.nn.modules import Conv as UConv

    c_in = dw.conv.in_channels
    c_out = dw.conv.out_channels
    k = dw.conv.kernel_size[0]
    s = dw.conv.stride[0]
    p = dw.conv.padding[0]
    g = dw.conv.groups
    assert g == c_in == c_out, f"not depthwise: g={g} cin={c_in} cout={c_out}"

    new = UConv(c_in, c_out, k, s, p, g=1).float()
    with torch.no_grad():
        w = new.conv.weight                       # [c_out, c_in, k, k]
        w.zero_()
        dw_w = dw.conv.weight.float()             # [c_out, 1, k, k]
        for o in range(c_out):
            w[o, o] = dw_w[o, 0]                  # diagonal placement
        if getattr(dw, "bn", None) is not None and new.bn is not None:
            new.bn.load_state_dict(
                {k2: v.float() for k2, v in dw.bn.state_dict().items()})
            # eps and momentum are CONSTRUCTOR arguments, not state, so
            # load_state_dict does NOT carry them.  Ultralytics builds
            # BatchNorm2d(eps=0.001, momentum=0.03) while torch defaults to
            # eps=1e-5 -- and because these depthwise outputs have running_var
            # around 0.0016, eps dominates the denominator: sqrt(0.0016+0.001)
            # vs sqrt(0.0016+1e-5) is a 27% scale error, which showed up as a
            # ~19 absolute output difference and looked exactly like a broken
            # weight initialisation.  Copying the state alone is not enough.
            new.bn.eps = dw.bn.eps
            new.bn.momentum = dw.bn.momentum
        if hasattr(new, "act") and hasattr(dw, "act"):
            new.act = copy.deepcopy(dw.act)
    # Match the source's train/eval mode.  A freshly constructed module has
    # training=True, so new.bn would normalise by the BATCH statistics of the
    # comparison input instead of the running stats just copied in -- which
    # showed up as a 14.39 output difference on a batch of 2 and looked exactly
    # like a broken diagonal initialisation.  The weights were correct all along.
    new.train(dw.training)
    return new


def flatten_tensors(obj):
    """Collect every tensor inside a (possibly nested) dict/list/tuple output.

    Ultralytics' DetectionModel returns a dict in some modes and a list/tuple in
    others, so the comparison must not assume a shape.
    """
    out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_tensors(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_tensors(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Map detection-head DWConv to dense 3x3 Conv")
    parser.add_argument("--src", type=Path, default=ROOT / "runs_yolo26/yolo26n_224/weights/best.pt")
    parser.add_argument("--dst", type=Path, default=ROOT / "build/model_source/pl_native.pt")
    args = parser.parse_args()
    if not args.src.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.src}")

    ck = torch.load(str(args.src), map_location="cpu", weights_only=False)
    model = ck["model"]
    # Stored in FP16 (forward failed with "Input type (torch.FloatTensor) and
    # weight type (torch.HalfTensor)").  The host pipeline, the exporter and the
    # INT8 calibration all work in FP32, and float() is exact for FP16 -> FP32.
    model = model.float()
    model.eval()

    # ---- reference output BEFORE surgery ------------------------------
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = model(x)
    ref_t = [t.clone() for t in flatten_tensors(ref)]
    print(f"reference forward ok: {len(ref_t)} tensor(s), shapes "
          f"{[tuple(t.shape) for t in ref_t]}")

    # ---- the surgery ---------------------------------------------------
    # ALL DWConv in the detection head, both branches:
    #   model.23.cv3.*          -- the exported branch
    #   model.23.one2one_cv3.*  -- the NMS-free end-to-end branch, dropped by
    #                              the end2end=False export
    # The project has decided it does not want depthwise at all, so the
    # replacement is applied to both.  It costs nothing to do so (the diagonal
    # initialisation is exact for either branch) and it means the model carries
    # no grouped convolution anywhere -- so a later end2end=True export cannot
    # quietly reintroduce 6 PS-side operators.
    targets = []
    for name, mm in model.named_modules():
        if type(mm).__name__ == "DWConv" and name.startswith("model.23."):
            targets.append((name, mm))
    print(f"\nDWConv modules in the detection head: {len(targets)}")
    for name, mm in targets:
        print(f"  {name:<30} Cin={mm.conv.in_channels} Cout={mm.conv.out_channels} "
              f"k={mm.conv.kernel_size[0]} s={mm.conv.stride[0]}")

    for name, mm in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
        old = getattr(parent, attr)
        new = dwconv_to_conv(old)
        # per-layer equivalence check before committing to the swap
        with torch.no_grad():
            t = torch.randn(2, old.conv.in_channels, 16, 16)
            d = (old.float()(t) - new(t)).abs().max().item()
        setattr(parent, attr, new)
        flag = "ok" if d <= 1e-5 else "**NOT EQUIVALENT**"
        print(f"  {name:<30} per-layer max diff {d:.3e}  {flag}")
        if d > 1e-5:
            sys.exit(1)
    print(f"\nreplaced {len(targets)} DWConv -> standard Conv (diagonal init)")

    # ---- prove the whole network is unchanged --------------------------
    model.eval()
    with torch.no_grad():
        out = model(x)
    out_t = [t.clone() for t in flatten_tensors(out)]
    worst = 0.0
    for a, b in zip(ref_t, out_t):
        worst = max(worst, (a - b).abs().max().item())
    print(f"\nwhole-network max abs output difference: {worst:.3e}")
    if worst < 1e-4:
        print("  -> EXACT (within float rounding): the swap is numerically lossless")
    else:
        print("  -> **DIFFERS** -- diagonal init is wrong, do not use this checkpoint")
        sys.exit(1)

    # ---- what is left? -------------------------------------------------
    left = [(n, mm) for n, mm in model.named_modules()
            if isinstance(mm, nn.Conv2d) and mm.groups > 1]
    print(f"\ngrouped convs remaining: {len(left)}")
    for n, mm in left:
        print(f"  {n:<40} grp={mm.groups} ({mm.in_channels}ch, k={mm.kernel_size[0]})")
    print("  (the two `pe` convs vanish when the attention is replaced in Phase B)")

    print(f"\nparams: {sum(p.numel() for p in model.parameters()):,}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    ck["model"] = model
    torch.save(ck, str(args.dst))
    print(f"saved -> {args.dst}  ({args.dst.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
