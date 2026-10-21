"""Intermediate CBAM class required to load the source checkpoint.

The delivered accelerator uses the fitted 1x1 residual replacement instead of
implementing CBAM. Notes below record the operator candidate that led to the
source checkpoint; they do not describe the final PL datapath.

Lives in its own module (not inside a script's __main__) because the checkpoint
pickles this class by reference: a CBAM defined in `surgery_cbam_attention.py`
unpickles only when that script is __main__, which breaks every later consumer
(export, fine-tune, ONNX census) with "Can't get attribute 'CBAM'".

WHY CBAM HERE
-------------
YOLO26's C2PSA attention is `qkv(1x1) -> q^T k -> softmax -> v @ attn^T + pe(v)
-> proj(1x1)`.  No operator in the accelerator's set can express q^T k, softmax
or the matmul, which makes those two layers the last PS-side operators and the
reason the schedule is split into 9 range calls.  CBAM replaces them with
pooling, convolution and sigmoid -- all things the accelerator can do.

THE RESIDUAL SLOT
-----------------
`PSABlock.forward` is `x = x + self.attn(x)`, so this slot expects a RESIDUAL:
contributing ~0 is identity.  CBAM's natural output is a GATE (~x), which would
make the block compute x + x*gate and roughly double the activations.  So
`forward` returns the residual form explicitly:

    return x * (ca * sa) - x        =>  PSABlock yields  x * ca * sa

OPERATOR REQUIREMENTS (all inside one OP_ATTENTION)
---------------------------------------------------
  global average pool  (x.mean over H,W)
  global max pool      (x.amax over H,W)
  channel mean / max   (over C)
  1x1 conv, 3x3 conv
  sigmoid
  broadcast multiply

Only the last three plus the two pooling forms are new; the plan is to
implement them inside the existing OP_ATTENTION code slot rather than add four
new opcodes.

INITIALISATION
--------------
Both sigmoids are biased to ~1 at step 0 (ca*sa = 0.9817, residual -0.018x), so
the block output starts at 0.9817x -- within 2% of identity -- and the
pretrained features survive initialisation while the gradient stays healthy
(sigmoid at 4, not saturated at 10).  This is NOT lossless like the depthwise
swap: the trained attention did real work, so a fine-tune is required.

The bias goes on bn.bias, NOT conv.bias: ultralytics' Conv is built with
bias=False because the following BatchNorm owns the bias, so writing conv.bias
silently does nothing and leaves both sigmoids at sigmoid(0)=0.5, i.e. a gate of
0.25 and a block output of 0.25x.  That was measured, not theorised.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CBAM(nn.Module):
    """Channel + spatial attention with a residual-form output."""

    def __init__(self, c: int, r: int = 16, k: int = 3, init_bias: float = 4.0):
        super().__init__()
        from ultralytics.nn.modules import Conv as UConv

        hidden = max(c // r, 8)
        self.mlp = nn.Sequential(
            UConv(c, hidden, 1),
            UConv(hidden, c, 1, act=False),
        )
        # k=3 rather than CBAM's usual 7: the accelerator's conv engine is 3x3.
        self.spatial = UConv(2, 1, k, 1, k // 2, act=False)

        with torch.no_grad():
            last = self.mlp[-1]
            nn.init.zeros_(last.conv.weight)
            if last.bn is not None:
                last.bn.weight.fill_(1.0)
                last.bn.bias.fill_(init_bias)
                last.bn.running_mean.fill_(0.0)
                last.bn.running_var.fill_(1.0)
            self.spatial.conv.weight.zero_()
            if self.spatial.bn is not None:
                self.spatial.bn.weight.fill_(1.0)
                self.spatial.bn.bias.fill_(init_bias)
                self.spatial.bn.running_mean.fill_(0.0)
                self.spatial.bn.running_var.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel attention: shared MLP over average- and max-pooled features
        avg = x.mean(dim=(2, 3), keepdim=True)
        mx = x.amax(dim=(2, 3), keepdim=True)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        g = x * ca

        # spatial attention: channel-wise avg/max -> conv -> sigmoid
        am = g.mean(dim=1, keepdim=True)
        mm = g.amax(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([am, mm], dim=1)))

        # residual form, so PSABlock's `x + attn(x)` yields x * ca * sa
        return x * (ca * sa) - x
