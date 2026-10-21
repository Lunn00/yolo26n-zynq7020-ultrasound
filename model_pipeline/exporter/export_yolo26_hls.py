#!/usr/bin/env python3
"""Build the YOLO26n ONNX graph schedule used by the final INT8 exporter.

The standalone FP32 export retains historical intermediate file names for
compatibility. The delivered inference artifacts come from
export_yolo26_hls_int8.py and the checked-in HLS layer table and blobs.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


OP_COPY = 0
OP_CONV = 1
OP_SPLIT = 2
OP_ADD = 3
OP_CONCAT = 4
OP_UP2X = 5
OP_MAXPOOL = 6
# YOLO26 (C2PSA) only.  QK^T / softmax / AV are executed on the PS, not in the
# fabric -- see project handover 922 "YOLO26部署路线.md" §4.  The PL engine must
# never receive this op: the host driver splits the run around it.
OP_ATTENTION = 7

# ---- YOLO26n @224 head geometry ------------------------------------------
# YOLOv8n @640 was 80/40/20 grids (6400/1600/400 anchors, box = 4*16 DFL bins).
# YOLO26n @224 is 28/14/7 (784/196/49 anchors, box = 4 direct xywh values), and
# the two raw head tensors are laid out box-then-class exactly like v8.
INPUT_H = 224
INPUT_W = 224
ANCHORS = 784 + 196 + 49
BOX_CHANNELS = 4
HEAD_ANCHOR_OFF = {0: 0, 1: 784, 2: 784 + 196}

# The Detect head lives in a different Sequential slot per architecture:
#   YOLOv8n : /model.22   (and its last conv is /model.22/dfl/conv/Conv, which is
#                         what the old shape pass used as its stop marker)
#   YOLO26n : /model.23   and there is NO dfl conv at all (reg_max=1), so the old
#                         marker never fires and the shape pass walks into the
#                         make-anchors arithmetic (Shape/Gather/...) and dies.
# Parameterised here instead of hard-coded.
HEAD_MODULE = "/model.23"

# Everything after the head convs is decode arithmetic that either runs on the PS
# or is static for a fixed input size.
HEAD_TAIL_OPS = {
    "Shape", "Gather", "GatherElements", "Unsqueeze", "Flatten", "Cast",
    "ReduceMax", "Mod", "ConstantOfShape", "Expand", "Tile", "TopK",
}


@dataclass
class Layer:
    op: int
    inputs: list[str] = field(default_factory=list)
    out: str = ""
    out2: str = ""
    oc: int = 0
    ic: int = 0
    oh: int = 0
    ow: int = 0
    ih: int = 0
    iw: int = 0
    k: int = 0
    stride: int = 1
    pad: int = 0
    silu: int = 0
    out_target: int = 0
    woff: int = 0
    boff: int = 0
    out_off: int = 0
    out_stride: int = 0
    cin: list[int] = field(default_factory=list)
    # Physical convolution inputs after eliminating channel-concat and
    # nearest-neighbour resize materialisation.  One flag per input says that
    # the source is read as a virtual 2x upsampled view.
    in_up2x: list[int] = field(default_factory=list)
    # A residual Add can be folded into the producer convolution's row flush.
    # quant_out keeps the original Conv output quantisation domain so this
    # transformation stays bit-exact with the unfused INT8 schedule.
    fuse_add: int = 0
    add_input: str = ""
    quant_out: str = ""
    c0: int = 0
    c1: int = 0
    # Convolution groups.  1 for every conv in YOLOv8n; 8 of YOLO26's convs are
    # DEPTHWISE (groups == out_channels, weight shape (C,1,3,3), so ic == 1 while
    # the input tensor has C channels).  The engine's layer model assumes every
    # output channel consumes all `ic` input channels, which is exactly what makes
    # grouped convs unrepresentable there -- see the 922 handover note.
    groups: int = 1
    # OP_ATTENTION only: the depthwise `pe` conv (shape (C,1,3,3)) and its bias
    # are appended to the weight blob and addressed through woff/boff, exactly
    # like a normal conv.  They stay FP32 because the unit ran on the CPU.
    #
    # OP_ATTENTION (CBAM) only: the six FP32 parameter tensors of the CBAM block,
    # in the order the exporter consumes them:
    #   [0] mlp0 weight (hidden, C, 1, 1)   [1] mlp0 bias (hidden)
    #   [2] mlp1 weight (C, hidden, 1, 1)   [3] mlp1 bias (C)
    #   [4] sp   weight (1, 2, k, k)        [5] sp   bias (1)
    # They are offsets into the FP32 weight blob, like woff/boff.  c0 carries the
    # MLP hidden width, which the C2PSA layout used for num_heads.
    cbam_woff: list[int] = field(default_factory=list)
    # OP_ATTENTION (CBAM) only: byte offset of the 275-float parameter block
    # inside the FIXED-PARAM blob (yolo_fixed_params_i8.bin), which is what
    # op_cbam_i8 reads through L.ascale_off.  NOT an act-scale-blob offset.
    ascale_off: int = 0


def resolve_alias(name: str, alias: dict[str, str]) -> str:
    return alias.get(name, name)


def shape_of(name: str, shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    return shapes[name]


def conv_output_shape(in_shape, w_shape, stride, pad):
    _, _, ih, iw = in_shape
    oc, _, kh, kw = w_shape
    oh = (ih + 2 * pad - kh) // stride + 1
    ow = (iw + 2 * pad - kw) // stride + 1
    return (1, oc, oh, ow)


# ---------------------------------------------------------------------------
# CBAM attention (the PL's OP_ATTENTION)
# ---------------------------------------------------------------------------
# `cbam_pl.CBAM` exports as ordinary ONNX ops -- two reduce-pools, a shared
# 1x1 MLP, a sigmoid, a channel mean/max pair, a 3x3 conv, a second sigmoid and
# a residual Sub -- about 20 nodes.  `op_cbam_i8` executes the whole thing as ONE
# layer, so emitting those nodes individually would produce a layer table with 20
# rows where the fabric has a single operator (and the CBAM block count would no
# longer match the two C2PSA slots).  This pass collapses each one.
#
# THE PATTERN, read backwards from the residual (the only unambiguous anchor):
#
#     out = Sub(Mul(x, gate), x)                     gate = Mul(ca, sa)
#     ca  = Sigmoid(Add(mlp1(avg), mlp1(mx)))        avg/mx = ReduceMean/Max(x, H,W)
#     sa  = Sigmoid(sp(Concat(mean(g), max(g),1)))   g = Mul(x, ca)
#
# Nothing is matched by node name: the weights are taken from the discovered
# conv nodes' own initializer inputs, so a renamed export still works.  Every
# step validates what it found and raises, because a half-matched CBAM subgraph
# would otherwise emit a plausible-looking row whose weights belong to something
# else -- and that fails on the board, not in the export.
def find_cbam_blocks(graph, tensors, shapes):
    """Locate every CBAM block and return (plans, interior node names)."""

    producer: dict[str, object] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    def conv_params(node):
        if node is None or node.op_type != "Conv":
            return None
        w = tensors.get(node.input[1])
        if w is None:
            return None
        b = tensors.get(node.input[2]) if len(node.input) > 2 else None
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        groups = int(attrs.get("group", 1))
        ks = attrs.get("kernel_shape", [w.shape[2], w.shape[3]])
        pads = attrs.get("pads", [0, 0, 0, 0])
        strides = attrs.get("strides", [1, 1])
        if groups != 1 or strides[0] != 1 or strides[1] != 1:
            return None
        return dict(w=w, b=b, k=int(ks[0]), pad=int(pads[0]))

    plans = []
    interior: set[str] = set()

    for sub in graph.node:
        if sub.op_type != "Sub" or len(sub.input) != 2:
            continue
        m2 = producer.get(sub.input[0])
        x = sub.input[1]
        if m2 is None or m2.op_type != "Mul" or x not in list(m2.input):
            continue
        gate_name = m2.input[0] if m2.input[1] == x else m2.input[1]
        gate = producer.get(gate_name)
        if gate is None or gate.op_type != "Mul" or len(gate.input) != 2:
            continue

        # gate = Mul(ca, sa); separate the two sigmoids by what they consume:
        # the channel gate is Sigmoid(Add(...)), the spatial gate is
        # Sigmoid(3x3 Conv(Concat(...))).
        ca_sig = sa_sig = None
        for t in gate.input:
            sig = producer.get(t)
            if sig is None or sig.op_type != "Sigmoid":
                break
            inner = producer.get(sig.input[0])
            if inner is None:
                break
            if inner.op_type == "Add":
                ca_sig = sig
            elif inner.op_type == "Conv" and conv_params(inner) is not None:
                sa_sig = sig
        if ca_sig is None or sa_sig is None:
            continue

        # ---- spatial branch: mean/max over C of the gated tensor -----------
        sp_node = producer[sa_sig.input[0]]
        sp = conv_params(sp_node)
        cat = producer.get(sp_node.input[0])
        if sp is None or cat is None or cat.op_type != "Concat" or len(cat.input) != 2:
            continue
        pools = [producer.get(t) for t in cat.input]
        if any(p is None or p.op_type not in ("ReduceMean", "ReduceMax") for p in pools):
            continue
        g_name = pools[0].input[0]
        if any(p.input[0] != g_name for p in pools):
            continue
        g = producer.get(g_name)
        if g is None or g.op_type != "Mul" or x not in list(g.input):
            continue
        if (g.input[0] if g.input[1] == x else g.input[1]) != ca_sig.output[0]:
            continue

        # ---- channel branch: shared MLP applied to avg and max -------------
        add = producer[ca_sig.input[0]]
        m1_nodes = [producer.get(t) for t in add.input]
        m1 = [conv_params(n) for n in m1_nodes]
        if any(c is None for c in m1):
            continue
        m0_nodes, m0, pools0 = [], [], []
        for node in m1_nodes:
            silu = producer.get(node.input[0])
            if silu is None or silu.op_type != "Mul" or len(silu.input) != 2:
                break
            conv0 = producer.get(silu.input[0])
            p0 = conv_params(conv0)
            if p0 is None:
                break
            red = producer.get(conv0.input[0])
            if red is None or red.op_type not in ("ReduceMean", "ReduceMax"):
                break
            if red.input[0] != x:
                break
            m0_nodes.append(conv0)
            m0.append(p0)
            pools0.append(red)
        if len(m0) != 2:
            continue

        xs = shapes.get(x)
        if xs is None or len(xs) != 4:
            raise RuntimeError(f"CBAM input {x} has no 4-D shape")
        _, c, hh, ww = (int(v) for v in xs)
        hidden = int(m0[0]["w"].shape[0])
        k = int(sp["k"])
        if m0[0]["w"].shape != (hidden, c, 1, 1) or m0[1]["w"].shape != (hidden, c, 1, 1):
            raise RuntimeError(f"CBAM {x}: mlp0 weight {m0[0]['w'].shape} != ({hidden},{c},1,1)")
        if m1[0]["w"].shape != (c, hidden, 1, 1) or m1[1]["w"].shape != (c, hidden, 1, 1):
            raise RuntimeError(f"CBAM {x}: mlp1 weight {m1[0]['w'].shape} != ({c},{hidden},1,1)")
        if sp["w"].shape != (1, 2, k, k):
            raise RuntimeError(f"CBAM {x}: spatial weight {sp['w'].shape} != (1,2,{k},{k})")
        # The MLP is SHARED between the average- and max-pooled branches: the
        # ONNX spells the module out twice but both copies must read the same
        # initializers, because op_cbam_i8 has exactly one set of weights.
        if m0_nodes[0].input[1] != m0_nodes[1].input[1]:
            raise RuntimeError(
                f"CBAM {x}: the two mlp0 convolutions do not share weights "
                f"({m0_nodes[0].input[1]} vs {m0_nodes[1].input[1]}); a single "
                "OP_ATTENTION cannot express a non-shared MLP"
            )
        if m1_nodes[0].input[1] != m1_nodes[1].input[1]:
            raise RuntimeError(
                f"CBAM {x}: the two mlp1 convolutions do not share weights"
            )

        # Everything reachable from the residual without crossing `x` is the
        # interior of this block and must not become a layer of its own.
        names: set[str] = set()
        stack = [sub.input[0]]
        seen_t: set[str] = set()
        allowed = {"Conv", "Sigmoid", "Mul", "Add", "ReduceMean", "ReduceMax", "Concat"}
        while stack:
            t = stack.pop()
            if t == x or t in seen_t:
                continue
            seen_t.add(t)
            node = producer.get(t)
            if node is None or node.op_type == "Constant":
                continue
            if node.op_type not in allowed:
                raise RuntimeError(
                    f"CBAM {x}: unexpected {node.op_type} node {node.name} inside the "
                    "block -- the walk escaped the subgraph"
                )
            names.add(node.name)
            stack.extend(node.input)

        plans.append(
            dict(
                x=x,
                out=sub.output[0],
                sub_name=sub.name,
                hidden=hidden,
                c=c,
                ih=hh,
                iw=ww,
                k=k,
                pad=int(sp["pad"]),
                # A missing bias becomes a correctly SIZED zero vector, not a
                # 1-element one: the exporter slices these by length, and a short
                # array would silently read into the next tensor's bytes.
                params=[
                    m0[0]["w"],
                    m0[0]["b"] if m0[0]["b"] is not None else np.zeros(hidden, np.float32),
                    m1[0]["w"],
                    m1[0]["b"] if m1[0]["b"] is not None else np.zeros(c, np.float32),
                    sp["w"],
                    sp["b"] if sp["b"] is not None else np.zeros(1, np.float32),
                ],
                spans=[m0_nodes[0].name, m1_nodes[0].name, sp_node.name],
            )
        )
        interior |= names

    return plans, interior


def build_reduced_graph(model):
    graph = model.graph
    tensors = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    inputs = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in graph.input}
    shapes: dict[str, tuple[int, ...]] = dict(inputs)
    consumers: dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    # First pass: compute every tensor shape we need from constant shapes.
    for node in graph.node:
        # NOTE: no early break here.  v8 could stop the shape pass at its dfl conv
        # because everything after it was head decode.  YOLO26 has no dfl AND its
        # head interleaves decode Reshapes with the cv2/cv3 convs one level at a
        # time (cv2.0, cv3.0, Reshape, cv2.1, cv3.1, Reshape, ...), so any name
        # boundary drops real convs.  Instead: process every node whose inputs are
        # already known, and skip the rest (the decode tensors are never needed --
        # those nodes are skipped in the emit pass too).
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        # ONNX Constant *nodes* (as opposed to graph.initializer) carry the shape
        # constants that the YOLO26 attention Reshapes feed off.  Without these
        # the Reshape branch below silently fails and the shape never registers.
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    v = onnx.helper.get_attribute_value(a)
                    if isinstance(v, onnx.TensorProto):
                        tensors[node.output[0]] = numpy_helper.to_array(v)
            continue
        # PyTorch's ONNX exporter deduplicates identical initializers by making
        # later weight names Identity outputs.  Both zero-initialised residual
        # 1x1 convolutions use the same tensor; resolve the alias before the
        # Conv pass reads its weight, rather than treating it as an activation.
        if node.op_type == "Identity" and node.input[0] in tensors:
            tensors[node.output[0]] = tensors[node.input[0]]
            continue
        if node.op_type == "MatMul":
            # batched [.., M, K] x [.., K, N] -> [.., M, N]
            a0 = shapes.get(node.input[0])
            a1 = shapes.get(node.input[1])
            if a0 and a1:
                shapes[node.output[0]] = tuple(a0[:-1]) + (a1[-1],)
            continue
        # Process any node whose inputs are already known; skip the rest.  The
        # skipped ones are head-decode shape arithmetic whose results are never
        # needed (those nodes are skipped in the emit pass too).
        if not all((not i) or (i in shapes) or (i in tensors) for i in node.input):
            continue
        if node.op_type == "Conv":
            if node.input[1] not in tensors:
                continue
            w = tensors[node.input[1]]
            stride = attrs.get("strides", [1, 1])
            pad = attrs.get("pads", [0, 0, 0, 0])
            shapes[node.output[0]] = conv_output_shape(
                shape_of(node.input[0], shapes), w.shape, stride[0], pad[0]
            )
        elif node.op_type in ("Sigmoid", "MaxPool", "Softmax"):
            shapes[node.output[0]] = shape_of(node.input[0], shapes)
        elif node.op_type in ("ReduceMean", "ReduceMax"):
            # CBAM's global/channel pools.  `axes` is an attribute before opset
            # 18 and an input from 18 on, so both spellings are read.
            in_shape = shape_of(node.input[0], shapes)
            axes = attrs.get("axes")
            if axes is None and len(node.input) > 1 and node.input[1] in tensors:
                axes = [int(x) for x in tensors[node.input[1]].reshape(-1)]
            if axes is None:
                axes = list(range(len(in_shape)))
            axes = [int(a) % len(in_shape) for a in axes]
            keep = int(attrs.get("keepdims", 1))
            if keep:
                shapes[node.output[0]] = tuple(
                    1 if i in axes else int(d) for i, d in enumerate(in_shape)
                )
            else:
                shapes[node.output[0]] = tuple(
                    int(d) for i, d in enumerate(in_shape) if i not in axes
                )
        elif node.op_type == "Sub":
            shapes[node.output[0]] = shape_of(node.input[0], shapes)
        elif node.op_type == "Transpose":
            perm = attrs.get("perm")
            in_shape = shape_of(node.input[0], shapes)
            shapes[node.output[0]] = tuple(in_shape[p] for p in perm)
        elif node.op_type == "Mul":
            # y = a * sigmoid(a) has the same shape as its first operand.
            shapes[node.output[0]] = shape_of(node.input[0], shapes)
        elif node.op_type == "Split":
            in_shape = shape_of(node.input[0], shapes)
            if len(node.input) > 1 and node.input[1] in tensors:
                sizes = tensors[node.input[1]].tolist()
            else:
                c = in_shape[1]
                sizes = [c // 2, c - c // 2]
            for out_name, size in zip(node.output, sizes):
                shapes[out_name] = (1, size, in_shape[2], in_shape[3])
        elif node.op_type == "Add":
            shapes[node.output[0]] = shape_of(node.input[0], shapes)
        elif node.op_type == "Concat":
            in0 = shape_of(node.input[0], shapes)
            axis = attrs.get("axis", 1)
            if axis == 1:
                total = sum(shape_of(n, shapes)[1] for n in node.input)
                shapes[node.output[0]] = (1, total, in0[2], in0[3])
            else:
                total = sum(shape_of(n, shapes)[-1] for n in node.input)
                shapes[node.output[0]] = in0[:-1] + (total,)
        elif node.op_type == "Resize":
            in_shape = shape_of(node.input[0], shapes)
            scales = tensors[node.input[2]].flatten().tolist()
            shapes[node.output[0]] = (
                1,
                in_shape[1],
                int(in_shape[2] * scales[2]),
                int(in_shape[3] * scales[3]),
            )
        elif node.op_type == "Reshape":
            if len(node.input) > 1 and node.input[1] in tensors:
                shape = tensors[node.input[1]].tolist()
                src = shapes.get(node.input[0])
                if src is None:
                    continue
                n = int(np.prod(src))
                shape = [n if d == -1 else d for d in shape]
                shapes[node.output[0]] = tuple(shape)

    # Build aliases for conv -> Sigmoid -> Mul chains.
    alias: dict[str, str] = {}
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        conv_out = node.output[0]
        sig_nodes = [c for c in consumers.get(conv_out, []) if c.op_type == "Sigmoid"]
        if not sig_nodes:
            continue
        sig_out = sig_nodes[0].output[0]
        mul_nodes = [c for c in consumers.get(sig_out, []) if c.op_type == "Mul"]
        mul = mul_nodes[0] if mul_nodes else None
        if mul is None or mul.op_type != "Mul":
            continue
        alias[conv_out] = mul.output[0]
        alias[mul.output[0]] = mul.output[0]

    def storage(name: str) -> str:
        return resolve_alias(name, alias)

    # ------------------------------------------------------------------
    # YOLO26 C2PSA: collapse the whole attention subgraph into ONE layer.
    #
    # The ONNX spells the attention out as 13 nodes, but 6 Reshape + 3 Transpose
    # are pure layout (zero compute) and qkv/pe/proj are ordinary 1x1 convs the
    # engine already runs.  The only real new compute is
    #     A_h = softmax(Q_h^T K_h / sqrt(32))   [49x49]   (2 heads)
    #     O_h = V_h A_h^T                       [64x49]
    # = 921,984 MACs for the whole network (0.036% of it).  That runs on the PS.
    #
    # So we emit one OP_ATTENTION consuming the qkv conv output [256,7,7] and
    # producing TWO tensors: the attention result [128,7,7] and v [128,7,7]
    # (the pe conv consumes v, so it has to be materialised as well).
    # ------------------------------------------------------------------
    # attn_plan[qkv_conv_name] = dict describing the CPU unit that replaces the
    # whole /attn/ subgraph.  The geometry is read out of the graph rather than
    # assumed -- the Reshape target [1, H, D, N] and the Split sizes decide the
    # channel layout the CPU unit has to walk:
    #     qkv [1, 256, 7, 7] -> view [1, H=2, D=128, N=49]
    #     split(D) = [key_dim, key_dim, head_dim] = [32, 32, 64]
    # so per head h:  q = channels [h*128, +32), k = [h*128+32, +32), v = [+64, +64).
    attn_plan: dict[str, dict] = {}
    for node in graph.node:
        if node.op_type == "Conv" and node.name.endswith("/attn/qkv/conv/Conv"):
            prefix = node.name.split("/qkv/")[0]          # .../attn
            r1 = r2 = add_out = ""
            view_shape = None
            split_sizes = None
            for n2 in graph.node:
                if not n2.name.startswith(prefix + "/"):
                    continue
                if n2.name.endswith("/Reshape_1"):
                    r1 = n2.output[0]                     # attention result -> [128,7,7]
                elif n2.name.endswith("/Reshape_2"):
                    r2 = n2.output[0]                     # v reshaped back -> [128,7,7]
                elif n2.name == prefix + "/Add":
                    add_out = n2.output[0]                # attention + pe
                elif n2.name == prefix + "/Reshape" and len(n2.input) > 1:
                    sh = tensors.get(n2.input[1])
                    if sh is not None:
                        view_shape = [int(x) for x in sh.reshape(-1)]
                elif n2.name == prefix + "/Split" and len(n2.input) > 1:
                    sp = tensors.get(n2.input[1])
                    if sp is not None:
                        split_sizes = [int(x) for x in sp.reshape(-1)]
            if view_shape is None or len(view_shape) != 4:
                raise RuntimeError(f"cannot read attention view shape at {node.name}")
            if not split_sizes or len(split_sizes) != 3:
                raise RuntimeError(f"cannot read attention split at {node.name}")
            num_heads = int(view_shape[1])
            key_dim, _, head_dim = (int(x) for x in split_sizes)
            # The layer's output is the Add result, i.e. attention + pe.  pe is a
            # DEPTHWISE 3x3 conv (ultralytics Attention: pe = Conv(dim, dim, 3, g=dim)),
            # and this engine's layer model (ic = w.shape[1], in_scale.size == ic)
            # cannot express grouped convs -- YOLOv8n has none, YOLO26 has 12.  So pe
            # and the residual Add run on the PS together with the attention.  That is
            # cheap: pe is only 128 x 7 x 7 x 9 = 56,448 MACs.
            out_name = add_out or r1
            pe_node = next(
                (n2 for n2 in graph.node if n2.name == prefix + "/pe/conv/Conv"),
                None,
            )
            if out_name:
                attn_plan[node.name] = dict(
                    out_name=out_name,
                    r1=r1,
                    r2=r2,
                    pe_node=pe_node,
                    num_heads=num_heads,
                    key_dim=key_dim,
                    head_dim=head_dim,
                )

    layers: list[Layer] = []
    weight_bytes = bytearray()

    # ---- CBAM attention: one OP_ATTENTION row per block -------------------
    cbam_plans, cbam_interior = find_cbam_blocks(graph, tensors, shapes)
    cbam_layers: dict[str, Layer] = {}
    for plan in cbam_plans:
        offs = []
        for arr in plan["params"]:
            if arr is None:
                arr = np.zeros(1, dtype=np.float32)
            arr = np.asarray(arr).astype("<f4").reshape(-1)
            offs.append(len(weight_bytes) // 4)
            weight_bytes += arr.tobytes()
        layer = Layer(
            op=OP_ATTENTION,
            inputs=[plan["x"]],
            out=plan["out"],
            ic=int(plan["c"]),
            oc=int(plan["c"]),
            ih=int(plan["ih"]),
            iw=int(plan["iw"]),
            oh=int(plan["ih"]),
            ow=int(plan["iw"]),
            k=int(plan["k"]),
            pad=int(plan["pad"]),
            # OP_ATTENTION reuses c0/c1: for CBAM, c0 is the MLP hidden width
            # (the contract's YOLO_CBAM_HIDDEN) and c1 is unused.
            c0=int(plan["hidden"]),
            c1=0,
            cbam_woff=offs,
        )
        cbam_layers[plan["x"]] = layer
        shapes[plan["out"]] = shapes[plan["x"]]
        print(
            f"INFO: cbam attention {plan['out']}: {plan['c']}x{plan['ih']}x{plan['iw']} "
            f"hidden={plan['hidden']} sp={plan['k']}x{plan['k']} (pad {plan['pad']}) "
            f"from {plan['x']}"
        )

    split_nodes = {
        "/model.2/Split": 32,
        "/model.4/Split": 64,
        "/model.6/Split": 128,
        "/model.8/Split": 256,
        "/model.12/Split": 128,
        "/model.15/Split": 64,
        "/model.18/Split": 128,
        "/model.21/Split": 256,
    }

    for node in graph.node:
        op = node.op_type
        if node.name.startswith(HEAD_MODULE + "/dfl/"):
            continue
        # CBAM internals (reduce-pools, both MLPs, the sigmoids, the 3x3 conv and
        # the residual Sub) are all inside the OP_ATTENTION row built above.
        if node.name in cbam_interior:
            continue
        # ---- YOLO26 attention.  Everything in /attn/ EXCEPT the qkv and proj
        # convs is folded into one OP_ATTENTION executed on the PS:
        #   layout ops (Reshape/Transpose/Split), the matmuls + softmax,
        #   the depthwise pe conv (unsupported by this engine), and the residual
        #   Add.  Result: qkv conv (PL) -> OP_ATTENTION (PS) -> proj conv (PL).
        if "/attn/" in node.name and op in (
            "Reshape", "Transpose", "MatMul", "Softmax", "Mul", "Split", "Add",
        ):
            continue
        if "/attn/pe/" in node.name:
            continue
        if op in ("Reshape", "Transpose", "Softmax", "Slice", "Sub", "Div"):
            continue
        if op in HEAD_TAIL_OPS:
            continue
        if op == "Mul" and node.name.startswith(HEAD_MODULE + "/Mul"):
            continue
        if op == "Sigmoid" and node.name == HEAD_MODULE + "/Sigmoid":
            continue
        if op == "Add" and node.name.startswith(HEAD_MODULE + "/Add"):
            continue
        if op == "Concat" and node.name.startswith(HEAD_MODULE + "/Concat"):
            continue
        if op == "Conv":
            w = tensors[node.input[1]]
            bias = tensors[node.input[2]] if len(node.input) > 2 else None
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            k = attrs.get("kernel_shape", [w.shape[2], w.shape[3]])[0]
            stride = attrs.get("strides", [1, 1])[0]
            pad = attrs.get("pads", [0, 0, 0, 0])[0]
            groups = int(attrs.get("group", 1))
            oc, ic, _, _ = w.shape
            in_shape = shape_of(node.input[0], shapes)
            out_shape = conv_output_shape(in_shape, w.shape, stride, pad)
            _, _, oh, ow = out_shape

            # A Conv feeding Sigmoid/Mul has SiLU activation folded into the layer.
            sig = next(
                (c for c in consumers.get(node.output[0], []) if c.op_type == "Sigmoid"),
                None,
            )
            silu = 1 if sig is not None and sig.op_type == "Sigmoid" else 0
            storage_out = storage(node.output[0])

            layer = Layer(
                op=OP_CONV,
                inputs=[storage(node.input[0])],
                out=storage_out,
                oc=oc,
                ic=ic,          # per-group input channels; total = ic * groups
                oh=oh,
                ow=ow,
                ih=in_shape[2],
                iw=in_shape[3],
                k=k,
                stride=stride,
                pad=pad,
                silu=silu,
                groups=groups,
                woff=len(weight_bytes) // 4,
            )
            weight_bytes += w.astype("<f4").tobytes()
            layer.boff = len(weight_bytes) // 4
            if bias is not None:
                weight_bytes += bias.astype("<f4").tobytes()
            layers.append(layer)

            # Immediately after the qkv conv, insert the CPU-executed attention.
            if node.name in attn_plan:
                plan = attn_plan[node.name]
                a_out = plan["out_name"]
                r1, r2 = plan["r1"], plan["r2"]
                pe_node = plan["pe_node"]
                # The qkv conv's OUTPUT channels (w.shape[0] == `oc` above == 256),
                # not its input channels.  Using in_shape[1] (=128, the C2PSA module
                # width) halved both fields -- caught by checking the generated row
                # against the YoloLayer struct order:
                #   op,nin,silu,k,stride,pad,out_target,c0,c1,oc,ic,oh,ow,ih,iw,...
                chin = int(oc)                # 256 = 2 heads x (32 q + 32 k + 64 v)
                half = chin // 2              # 128
                hh, ww = int(oh), int(ow)
                if chin % 2 or half <= 0:
                    raise RuntimeError(f"unexpected qkv width {chin} at {node.name}")
                shapes[a_out] = (1, half, hh, ww)
                if r1:
                    alias[r1] = a_out
                if r2:
                    alias[r2] = a_out     # v is consumed inside the CPU unit now
                layers.append(
                    Layer(
                        op=OP_ATTENTION,
                        inputs=[storage_out],
                        out=a_out,
                        ic=chin,          # qkv output channels, 256
                        oc=half,          # attention output channels, 128
                        ih=hh,
                        iw=ww,
                        oh=hh,
                        ow=ww,
                        # OP_ATTENTION reuses fields that are meaningless for it:
                        #   k      -> pe's depthwise kernel size (3)
                        #   c0, c1 -> num_heads, key_dim (the CPU unit needs both
                        #             to walk the qkv channel layout)
                        #   woff   -> FP32 offset of pe's weight  (C,1,3,3)
                        #   boff   -> FP32 offset of pe's bias    (C,)
                        # pe stays FP32: it is a 56k-MAC residual branch on the
                        # attention path, so quantizing it buys nothing and costs
                        # accuracy exactly where the gradient is largest.
                        k=3,
                        c0=int(plan["num_heads"]),
                        c1=int(plan["key_dim"]),
                    )
                )
                al = layers[-1]
                if pe_node is not None and pe_node.input[1] in tensors:
                    pw = tensors[pe_node.input[1]]
                    al.woff = len(weight_bytes) // 4
                    weight_bytes += pw.astype("<f4").tobytes()
                    al.boff = len(weight_bytes) // 4
                    if len(pe_node.input) > 2 and pe_node.input[2] in tensors:
                        pb = tensors[pe_node.input[2]]
                        weight_bytes += pb.astype("<f4").tobytes()
                    print(
                        f"INFO: attention {a_out}: qkv={chin} out={half} "
                        f"{hh}x{ww} heads={al.c0} key_dim={al.c1} "
                        f"pe={tuple(pw.shape)}@{al.woff}"
                    )

        elif op == "Split":
            in_shape = shape_of(node.input[0], shapes)
            if len(node.input) > 1 and node.input[1] in tensors:
                sizes = tensors[node.input[1]].tolist()
            else:
                sizes = [in_shape[1] // 2, in_shape[1] - in_shape[1] // 2]
            layers.append(
                Layer(
                    op=OP_SPLIT,
                    inputs=[storage(node.input[0])],
                    out=storage(node.output[0]),
                    out2=storage(node.output[1]),
                    oc=in_shape[1],
                    c0=sizes[0],
                    c1=sizes[1],
                    oh=in_shape[2],
                    ow=in_shape[3],
                )
            )
        elif op == "Add":
            in_shape = shape_of(node.input[0], shapes)
            layers.append(
                Layer(
                    op=OP_ADD,
                    inputs=[storage(node.input[0]), storage(node.input[1])],
                    out=storage(node.output[0]),
                    oc=in_shape[1],
                    oh=in_shape[2],
                    ow=in_shape[3],
                )
            )
        elif op == "Concat":
            axis = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}.get("axis", 1)
            if axis != 1:
                # Head concat is handled by writing the final conv directly into pred.
                continue
            in_shapes = [shape_of(n, shapes) for n in node.input]
            cin = [s[1] for s in in_shapes]
            layers.append(
                Layer(
                    op=OP_CONCAT,
                    inputs=[storage(n) for n in node.input],
                    out=storage(node.output[0]),
                    oc=sum(cin),
                    cin=cin,
                    oh=in_shapes[0][2],
                    ow=in_shapes[0][3],
                )
            )
        elif op == "Resize":
            in_shape = shape_of(node.input[0], shapes)
            out_shape = shape_of(node.output[0], shapes)
            layers.append(
                Layer(
                    op=OP_UP2X,
                    inputs=[storage(node.input[0])],
                    out=storage(node.output[0]),
                    oc=in_shape[1],
                    ih=in_shape[2],
                    iw=in_shape[3],
                    oh=out_shape[2],
                    ow=out_shape[3],
                )
            )
        elif op == "MaxPool":
            in_shape = shape_of(node.input[0], shapes)
            layers.append(
                Layer(
                    op=OP_MAXPOOL,
                    inputs=[storage(node.input[0])],
                    out=storage(node.output[0]),
                    oc=in_shape[1],
                    ih=in_shape[2],
                    iw=in_shape[3],
                    oh=in_shape[2],
                    ow=in_shape[3],
                    k=5,
                    pad=2,
                )
            )

    # Place each OP_ATTENTION immediately after the layer that produces its
    # input tensor.  The input is produced by whatever the emitter made of the
    # CBAM input's producer -- an OP_SPLIT view for the first block, a fused
    # Conv+Add (or a plain OP_ADD) for the second -- so the insertion point is
    # found by name rather than assumed.
    for x_name, layer in cbam_layers.items():
        pos = None
        for i, cand in enumerate(layers):
            if cand.out == x_name or (cand.out2 and cand.out2 == x_name):
                pos = i
        if pos is None:
            raise RuntimeError(
                f"no producer layer for CBAM input {x_name}; cannot place "
                f"OP_ATTENTION for {layer.out}"
            )
        layers.insert(pos + 1, layer)

    return layers, weight_bytes, shapes, alias


def allocate_work(layers: list[Layer], shapes: dict, alias: dict) -> tuple[dict[str, int], int]:
    def storage(name: str) -> str:
        return resolve_alias(name, alias)

    # Split is only a pair of channel-contiguous views.  Keep both outputs as
    # views of the input allocation and extend the lifetime of that allocation
    # until every view has been consumed.  This removes all split DDR copies
    # without allowing the allocator to overwrite a still-live view.
    view_parent: dict[str, tuple[str, int]] = {}
    for layer in layers:
        if layer.op == OP_SPLIT:
            plane = int(layer.oh * layer.ow)
            view_parent[layer.out] = (layer.inputs[0], 0)
            view_parent[layer.out2] = (layer.inputs[0], int(layer.c0 * plane))

    def root_and_delta(name: str) -> tuple[str, int]:
        name = storage(name)
        delta = 0
        seen: set[str] = set()
        while name in view_parent:
            if name in seen:
                raise RuntimeError(f"cyclic storage view: {name}")
            seen.add(name)
            parent, part_delta = view_parent[name]
            delta += part_delta
            name = storage(parent)
        return name, delta

    def all_inputs(layer: Layer) -> list[str]:
        inputs = list(layer.inputs)
        add_input = getattr(layer, "add_input", "")
        if add_input:
            inputs.append(add_input)
        return inputs

    # Count future uses of each storage tensor.  A residual consumed by a
    # fused Conv+Add must stay live until that convolution finishes.
    uses: dict[str, int] = {}
    for layer in layers:
        for inp in all_inputs(layer):
            root, _ = root_and_delta(inp)
            uses[root] = uses.get(root, 0) + 1

    alloc_offsets: dict[str, int] = {}
    live: set[str] = set()
    free: list[tuple[int, int]] = []
    high_water = 0

    def alloc(name: str, size: int) -> int:
        size = int(size)
        for i, (off, cap) in enumerate(free):
            if cap >= size:
                free.pop(i)
                alloc_offsets[name] = off
                live.add(name)
                return off
        nonlocal high_water
        off = high_water
        high_water += size
        alloc_offsets[name] = off
        live.add(name)
        return off

    def free_name(name: str) -> None:
        if name in live:
            size = int(np.prod(shapes[name]))
            free.append((alloc_offsets[name], size))
            live.remove(name)

    for layer in layers:
        if layer.op == OP_SPLIT:
            root, base_delta = root_and_delta(layer.inputs[0])
            if root not in alloc_offsets:
                raise RuntimeError(f"split input has no allocation: {root}")
            alloc_offsets[layer.out] = alloc_offsets[root] + base_delta
            alloc_offsets[layer.out2] = (
                alloc_offsets[root] + base_delta + int(layer.c0 * layer.oh * layer.ow)
            )
        else:
            # OP_ATTENTION produces exactly one visible tensor: q, k and v are all
            # slices of the qkv conv output and pe runs inside the unit, so v is
            # never materialised (out2 stays empty).
            outs = [layer.out]
            for out_name in outs:
                if out_name.startswith("head_") or out_name == "images":
                    continue
                root, _ = root_and_delta(out_name)
                out_size = int(np.prod(shapes[root])) if root in shapes else 0
                if out_size and root not in alloc_offsets:
                    alloc(root, out_size)
                if root in alloc_offsets:
                    alloc_offsets[out_name] = alloc_offsets[root]
        # Consume one use of each input.
        for inp in all_inputs(layer):
            root, _ = root_and_delta(inp)
            uses[root] -= 1
            if uses[root] <= 0 and root in live:
                free_name(root)

    return alloc_offsets, high_water


def finalize_layers(layers: list[Layer], offsets: dict[str, int]) -> list[Layer]:
    out_layers = []
    for layer in layers:
        l = Layer(**{k: v for k, v in vars(layer).items()})
        if l.op == OP_CONV:
            if l.out.startswith("head_box"):
                scale = int(l.out.split("_")[-1])
                anchor_off = HEAD_ANCHOR_OFF[scale]
                l.out_off = anchor_off
                l.out_stride = ANCHORS
            elif l.out.startswith("head_conf"):
                scale = int(l.out.split("_")[-1])
                anchor_off = HEAD_ANCHOR_OFF[scale]
                l.out_off = BOX_CHANNELS * ANCHORS + anchor_off
                l.out_stride = 1
            else:
                l.out_off = offsets[l.out]
                l.out_stride = l.oh * l.ow
        elif l.op == OP_COPY:
            l.out_off = offsets[l.out]
            l.out_stride = 1
        elif l.op == OP_ATTENTION:
            # Two outputs: the attention result and v (feeds the pe conv).
            l.out_off = offsets[l.out]
            l.out2_off = offsets.get(l.out2, 0)
            l.out_stride = l.oh * l.ow
            l.in_offs = [offsets.get(i, 0) for i in l.inputs]
        elif l.op == OP_SPLIT:
            l.out_off = offsets[l.out]
            l.out_stride = 1
            l.out2_off = offsets.get(l.out2, 0)
        else:
            l.out_off = offsets[l.out]
            l.out_stride = 1
            l.in_offs = [offsets.get(i, 0) for i in l.inputs]
        # Keep a resolved input offset list for the C++ table.
        l.in_offs = [offsets.get(i, 0) for i in l.inputs]
        out_layers.append(l)
    return out_layers


def write_header(path: Path, layers: list[Layer], work_floats: int, weight_floats: int) -> None:
    lines = [
        "#ifndef YOLO_LAYERS_H",
        "#define YOLO_LAYERS_H",
        "",
        "// Generated by scripts/export_yolo_hls.py - do not edit manually.",
        "",
        "#include <stdint.h>",
        "",
        "#define YOLO_INPUT_C 3",
        f"#define YOLO_INPUT_H {INPUT_H}",
        f"#define YOLO_INPUT_W {INPUT_W}",
        "#define YOLO_INPUT_FLOATS (YOLO_INPUT_C*YOLO_INPUT_H*YOLO_INPUT_W)",
        f"#define YOLO_WORK_FLOATS {work_floats}u",
        f"#define YOLO_WEIGHT_FLOATS {weight_floats}u",
        "#define YOLO_NUM_LAYERS " + str(len(layers)) + "u",
        f"#define YOLO_ANCHORS {ANCHORS}",
        f"#define YOLO_BOX_CHANNELS {BOX_CHANNELS}",
        "#define YOLO_PRED_FLOATS ((YOLO_BOX_CHANNELS+1)*YOLO_ANCHORS)",
        "",
        "#define OP_COPY 0",
        "#define OP_CONV 1",
        "#define OP_SPLIT 2",
        "#define OP_ADD 3",
        "#define OP_CONCAT 4",
        "#define OP_UP2X 5",
        "#define OP_MAXPOOL 6",
        "#define OP_ATTENTION 7",
        "",
        "// OP_ATTENTION (YOLO26 C2PSA) is the one op the PL engine does not",
        "// implement; it executes on the PS.  Its row reuses otherwise unused",
        "// members:",
        "//   k        pe (depthwise 3x3) kernel size",
        "//   c0, c1   num_heads, key_dim  (head_dim = oc / num_heads)",
        "//   ic       qkv output channels = num_heads*(2*key_dim + head_dim)",
        "//   oc       output channels     = num_heads*head_dim",
        "//   woff     FP32 offset of pe weight (oc,1,k,k)",
        "//   boff     FP32 offset of pe bias   (oc,)",
        "",
        "typedef struct {",
        "    uint8_t op;",
        "    uint8_t nin;",
        "    uint8_t silu;",
        "    uint8_t k;",
        "    uint8_t stride;",
        "    uint8_t pad;",
        "    uint8_t out_target;",
        "    uint16_t c0;",
        "    uint16_t c1;",
        "    uint32_t oc;",
        "    uint32_t ic;",
        "    uint32_t oh;",
        "    uint32_t ow;",
        "    uint32_t ih;",
        "    uint32_t iw;",
        "    uint32_t woff;",
        "    uint32_t boff;",
        "    uint32_t out_off;",
        "    uint32_t out_stride;",
        "    uint32_t out2_off;",
        "    uint32_t in_off[4];",
        "    uint16_t cin[4];",
        "} YoloLayer;",
        "",
        "static const YoloLayer yolo_layers[YOLO_NUM_LAYERS] = {",
    ]
    for l in layers:
        while len(l.in_offs) < 4:
            l.in_offs.append(0)
        while len(l.cin) < 4:
            l.cin.append(0)
        ins = ", ".join(str(x) for x in l.in_offs)
        cins = ", ".join(str(x) for x in l.cin)
        lines.append(
            "    {%d, %d, %d, %d, %d, %d, %d, %d, %d, %du, %du, %du, %du, %du, %du, %du, %du, %du, %du, %du, {%s}, {%s}},"
            % (
                l.op,
                len(l.inputs),
                l.silu,
                l.k,
                l.stride,
                l.pad,
                l.out_target,
                l.c0,
                l.c1,
                l.oc,
                l.ic,
                l.oh,
                l.ow,
                l.ih,
                l.iw,
                l.woff,
                l.boff,
                l.out_off,
                l.out_stride,
                getattr(l, "out2_off", 0),
                ins,
                cins,
            )
        )
    lines.extend(["};", "", "#endif", ""])
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=Path(__file__).resolve().parents[1] / "model_conv1x1_fit.onnx")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    layers, weight_bytes, shapes, alias = build_reduced_graph(model)

    # Insert a copy op for the external NCHW input into the scratch buffer.
    copy = Layer(
        op=OP_COPY,
        inputs=["images"],
        out="images_work",
        oc=3,
        ih=INPUT_H,
        iw=INPUT_W,
        oh=INPUT_H,
        ow=INPUT_W,
        out_stride=1,
    )
    layers.insert(0, copy)
    shapes["images_work"] = (1, 3, INPUT_H, INPUT_W)

    # Mark final head convs before allocation.
    for l in layers:
        if l.op == OP_CONV and "/cv2." in l.out and l.out.endswith("/Conv_output_0"):
            scale = int(l.out.split("/cv2.")[1][0])
            l.out = f"head_box_{scale}"
            l.out_target = 1
        elif l.op == OP_CONV and "/cv3." in l.out and l.out.endswith("/Conv_output_0"):
            scale = int(l.out.split("/cv3.")[1][0])
            l.out = f"head_conf_{scale}"
            l.out_target = 1

    offsets, work_floats = allocate_work(layers, shapes, alias)
    layers = finalize_layers(layers, offsets)

    out_dir = args.out_dir
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "src").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "yolov8n_weights.bin").write_bytes(weight_bytes)
    write_header(out_dir / "src" / "yolo_layers.h", layers, work_floats, len(weight_bytes) // 4)

    print(f"layers={len(layers)} work_floats={work_floats} weight_bytes={len(weight_bytes)}")
    print(f"wrote {out_dir / 'data' / 'yolov8n_weights.bin'}")
    print(f"wrote {out_dir / 'src' / 'yolo_layers.h'}")


if __name__ == "__main__":
    main()
