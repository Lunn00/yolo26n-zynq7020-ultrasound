#!/usr/bin/env python3
"""Post-training INT8 export and validation for the delivered YOLO26n HLS engine.

The script mirrors the FP32 schedule in export_yolo_hls.py, but quantizes:
  - conv weights to INT8 with a per-output-channel scale
  - per-input-channel activation scales folded into the INT8 weights
  - intermediate activations to INT8 with a per-channel scale / zero point
  - the detection head to signed INT16 Q10 in the pred buffer

The v2 weight blob remains compatible, while a generated integer parameter
blob replaces every run-time floating-point scale, bias and requantization.

Generated files:
  data/yolo26n_weights_i8.bin : INT8 weights + legacy FP32 metadata
  data/yolo_fixed_params_i8.bin: integer-only run-time parameters
  src/yolo_layers_i8.h        : extended layer table with scale fields
  data/yolo_quant_i8.json     : calibration / validation summary
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from export_yolo26_hls import (
    OP_ADD,
    OP_ATTENTION,
    OP_CONCAT,
    OP_CONV,
    OP_COPY,
    OP_MAXPOOL,
    OP_SPLIT,
    OP_UP2X,
    Layer,
    allocate_work,
    build_reduced_graph,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "model_conv1x1_fit.onnx"
DEFAULT_IMAGES = Path(__file__).resolve().parents[2] / "datasets" / "carotid" / "images" / "val"
DEFAULT_TRAIN_IMAGES = Path(__file__).resolve().parents[2] / "datasets" / "carotid" / "images" / "train"
TEST_IMAGES = ROOT / "test_img"

INPUT_H = 224
INPUT_W = 224
ANCHORS = 1029
# YOLO26 has reg_max=1 and emits 4 direct box values per anchor. Overridden by main()
# from --box-channels; every use below reads this module global at call time.
BOX_CHANNELS = 4
# Direct box regression has one value per side.
# Derived from --box-channels in main(): box_channels == 4 * reg_max.
REG_MAX = 1
PRED_FLOATS = (BOX_CHANNELS + 1) * ANCHORS
EPS = 1e-8


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def q_round(x: np.ndarray) -> np.ndarray:
    return np.rint(x)


def qint8(x: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
    q = q_round(np.asarray(x, dtype=np.float64) / scale)
    return np.clip(q, -128, 127).astype(np.int8)


def qact(x: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = q_round(np.asarray(x, dtype=np.float64) / scale) + zp
    return np.clip(q, 0, 255).astype(np.uint8)


def deq(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (q.astype(np.float64) - zp) * scale


def letterbox_rgb_nchw(img: Image.Image) -> np.ndarray:
    src_w, src_h = img.size
    scale = min(INPUT_W / src_w, INPUT_H / src_h)
    new_w = int(src_w * scale + 0.5)
    new_h = int(src_h * scale + 0.5)
    pad_x = (INPUT_W - new_w) // 2
    pad_y = (INPUT_H - new_h) // 2
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    out = np.full((3, INPUT_H, INPUT_W), 114.0 / 255.0, dtype=np.float32)
    for c in range(3):
        for y in range(INPUT_H):
            iy = y - pad_y
            if iy < 0 or iy >= new_h:
                continue
            for x in range(INPUT_W):
                ix = x - pad_x
                if ix < 0 or ix >= new_w:
                    continue
                sx = int(ix / scale)
                sy = int(iy / scale)
                out[c, y, x] = rgb[sy, sx, c]
    return out


def finalize_layers_preserve(layers: list[Layer], offsets: dict[str, int]) -> list[Layer]:
    """Same as export_yolo_hls.finalize_layers, but preserves extra attributes."""

    out_layers = []
    for layer in layers:
        l = copy.copy(layer)
        if l.op == OP_CONV:
            if l.out.startswith("head_box"):
                scale = int(l.out.split("_")[-1])
                anchor_off = {0: 0, 1: 784, 2: 980}[scale]
                l.out_off = anchor_off
                l.out_stride = ANCHORS
            elif l.out.startswith("head_conf"):
                scale = int(l.out.split("_")[-1])
                anchor_off = {0: 0, 1: 784, 2: 980}[scale]
                l.out_off = BOX_CHANNELS * ANCHORS + anchor_off
                l.out_stride = 1
            else:
                l.out_off = offsets[l.out]
                l.out_stride = l.oh * l.ow
        elif l.op == OP_COPY:
            l.out_off = offsets[l.out]
            l.out_stride = 1
        elif l.op == OP_ATTENTION:
            # CBAM: ONE output tensor.  The old PS-side unit also materialised
            # `v` for the depthwise pe conv; op_cbam_i8 needs no such thing, so
            # `out2`/`out2_off` stay empty.
            l.out_off = offsets[l.out]
            l.out_stride = l.oh * l.ow
        elif l.op == OP_SPLIT:
            l.out_off = offsets[l.out]
            l.out_stride = 1
            l.out2_off = offsets.get(l.out2, 0)
        else:
            l.out_off = offsets[l.out]
            l.out_stride = 1
        l.in_offs = [offsets.get(i, 0) for i in l.inputs]
        l.add_off = offsets.get(getattr(l, "add_input", ""), 0)
        out_layers.append(l)
    return out_layers


def fuse_layer_memory_ops(layers: list[Layer]) -> tuple[list[Layer], dict[str, int]]:
    """Eliminate DDR materialisation without reducing Conv parallelism.

    - Conv+Add keeps the Conv row on chip and combines the residual while the
      row is flushed.
    - Concat becomes a list of physical channel-plane sources on its consumer
      Conv.
    - A nearest-neighbour Resize feeding such a Concat becomes a virtual 2x
      view populated directly into the Conv line buffer.
    """

    stats = {"add": 0, "concat": 0, "up2x": 0}

    producers: dict[str, tuple[int, Layer]] = {}
    consumers: dict[str, list[int]] = {}
    for idx, layer in enumerate(layers):
        producers[layer.out] = (idx, layer)
        if layer.op == OP_SPLIT:
            producers[layer.out2] = (idx, layer)
        for name in layer.inputs:
            consumers.setdefault(name, []).append(idx)

    removed: set[int] = set()
    for idx, layer in enumerate(layers):
        if layer.op != OP_ADD or len(layer.inputs) != 2:
            continue
        conv_input = ""
        residual = ""
        conv_layer = None
        for pos, name in enumerate(layer.inputs):
            producer = producers.get(name)
            if producer and producer[1].op == OP_CONV and consumers.get(name) == [idx]:
                conv_input = name
                residual = layer.inputs[1 - pos]
                conv_layer = producer[1]
                break
        if conv_layer is None:
            continue
        conv_layer.fuse_add = 1
        conv_layer.add_input = residual
        conv_layer.quant_out = conv_input
        conv_layer.out = layer.out
        removed.add(idx)
        stats["add"] += 1

    remaining = [layer for idx, layer in enumerate(layers) if idx not in removed]
    up_views = {
        layer.out: (idx, layer.inputs[0])
        for idx, layer in enumerate(remaining)
        if layer.op == OP_UP2X and len(layer.inputs) == 1
    }
    consumers = {}
    for idx, layer in enumerate(remaining):
        for name in layer.inputs:
            consumers.setdefault(name, []).append(idx)

    removed = set()
    for idx, layer in enumerate(remaining):
        if layer.op != OP_CONCAT:
            continue
        users = consumers.get(layer.out, [])
        if len(users) != 1:
            continue
        consumer = remaining[users[0]]
        if consumer.op != OP_CONV or consumer.inputs != [layer.out]:
            continue

        physical_inputs: list[str] = []
        in_up2x: list[int] = []
        for name in layer.inputs:
            if name in up_views:
                up_idx, source = up_views[name]
                physical_inputs.append(source)
                in_up2x.append(1)
                removed.add(up_idx)
                stats["up2x"] += 1
            else:
                physical_inputs.append(name)
                in_up2x.append(0)
        if len(physical_inputs) > 4 or sum(layer.cin) != consumer.ic * consumer.groups:
            continue
        consumer.inputs = physical_inputs
        consumer.cin = list(layer.cin)
        consumer.in_up2x = in_up2x
        removed.add(idx)
        stats["concat"] += 1

    fused = [layer for idx, layer in enumerate(remaining) if idx not in removed]
    for layer in fused:
        if layer.op == OP_CONV:
            # `cin` is the PHYSICAL channel count of each source tensor, so for a
            # grouped conv it is ic*groups, not ic.  YOLO26's depthwise convs have
            # ic == 1 with groups == oc, and defaulting to [ic] made the schedule
            # read a single input channel for all 64/128/256 output channels.
            # numpy's einsum happily broadcasts that size-1 axis, so nothing
            # raised -- the head silently produced garbage and every detection
            # disappeared (box_max_abs_err froze at ~27 on every image).
            if not layer.cin:
                layer.cin = [layer.ic * layer.groups]
            if not layer.in_up2x:
                layer.in_up2x = [0] * len(layer.inputs)
            if sum(layer.cin) != layer.ic * layer.groups or len(layer.cin) != len(layer.inputs):
                raise RuntimeError(
                    f"invalid fused Conv input mapping for {layer.out}: "
                    f"cin={layer.cin} ic={layer.ic} groups={layer.groups} "
                    f"inputs={len(layer.inputs)}"
                )
    return fused, stats


def build_schedule(onnx_path: Path):
    model = onnx.load(onnx_path)
    layers, weight_bytes, shapes, alias = build_reduced_graph(model)

    calib_names = ["images"]
    for l in layers:
        calib_names.extend(l.inputs)
        calib_names.append(l.out)
        if l.op == OP_SPLIT:
            calib_names.append(l.out2)

    copy_layer = Layer(
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
    layers.insert(0, copy_layer)
    shapes["images_work"] = (1, 3, INPUT_H, INPUT_W)
    calib_names.append("images_work")

    layers, fusion_stats = fuse_layer_memory_ops(layers)
    print(
        "memory fusion: "
        f"Conv+Add={fusion_stats['add']}  "
        f"virtual Concat={fusion_stats['concat']}  "
        f"virtual Up2x={fusion_stats['up2x']}"
    )

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
    layers = finalize_layers_preserve(layers, offsets)

    calib_names = sorted(set(calib_names) - {"images"})
    return layers, weight_bytes, shapes, work_floats, calib_names


def make_session(onnx_path: Path, extra_outputs: list[str]):
    model = onnx.load(onnx_path)
    existing = {o.name for o in model.graph.output}
    for n in extra_outputs:
        if n not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(n))
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def calibrate(onnx_path: Path, image_paths: list[Path], names: list[str], quantile: float):
    model = onnx.load(onnx_path)
    real_names = {i.name for i in model.graph.input} | {
        o for n in model.graph.node for o in n.output
    }
    names = [n for n in names if n in real_names]
    session = make_session(onnx_path, names)
    bufs: dict[str, list[np.ndarray]] = {n: [] for n in names}
    maxabs: dict[str, float] = {n: 0.0 for n in names}

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inp = letterbox_rgb_nchw(img)[None].astype(np.float32)
        outs = session.run(names, {"images": inp})
        for n, a in zip(names, outs):
            a = np.asarray(a, dtype=np.float64)
            bufs[n].append(a)
            m = float(np.abs(a).max()) if a.size else 0.0
            if m > maxabs[n]:
                maxabs[n] = m

    metas: dict[str, tuple[float, int]] = {}
    for n in names:
        arrs = bufs[n]
        if not arrs:
            metas[n] = (np.array([1.0]), np.array([0], dtype=np.int16))
            continue
        c = arrs[0].shape[1]
        cat = np.concatenate([a.reshape(c, -1) for a in arrs], axis=1)
        lo = cat.min(axis=1)
        hi = cat.max(axis=1)
        if quantile < 1.0:
            hi = np.quantile(cat, quantile, axis=1)
        # The activation format is UNSIGNED with a zero point, so the quantized
        # interval MUST contain zero: zero_point = clip(round(-lo/scale), 0, 255)
        # silently collapses to 0 for any channel whose values are all positive,
        # and the representable interval then becomes [0, hi-lo] instead of
        # [lo, hi] -- every value above hi-lo is clipped flat.  On YOLO26's stem
        # conv that alone cost a 16% relative error on the very first layer (and
        # the network never recovered).  Forcing lo<=0<=hi fixes it.  The cost is
        # resolution on positive-only channels, which is strictly better than
        # losing the top of the range.
        lo = np.minimum(lo, 0.0)
        hi = np.maximum(hi, 0.0)
        scale = np.maximum(hi - lo, EPS) / 255.0
        zp = np.clip(np.rint(-lo / scale), 0, 255).astype(np.int16)
        metas[n] = (scale, zp)
    n_positive = sum(
        1
        for n in names
        if len(bufs[n]) and float(np.min(np.asarray(bufs[n][0]).min(axis=(0, 2, 3)))) > 0.0
    )
    print(
        f"calibration: {len(names)} tensors, {n_positive} with a strictly positive "
        f"first image (zero point forced to 0 -> upper range saved by the clamp)"
    )
    return metas, maxabs, session


def quantize_weights(
    layers: list[Layer],
    weight_bytes: bytes,
    metas: dict[str, tuple[float, int]],
    direct_fold: bool = False,
) -> tuple[bytearray, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    fp32 = np.frombuffer(weight_bytes, dtype="<f4")
    blob = bytearray()
    blob += struct.pack("<4sHH", b"Y8I8", 2, 0)
    quant: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    nconv = 0

    for idx, l in enumerate(layers):
        if l.op == OP_ATTENTION:
            # ------------------------------------------------------------------
            # CBAM, executed in the fabric by `op_cbam_i8`.  The weight blob
            # holds three INT8 tensors back to back at `wq_off` (which the HLS
            # reads as `weights + YOLO_I8_WEIGHT_BASE + wq_off`):
            #
            #   +0           mlp0  8 x 128 x 1   [oc][ic]     = 1024 B
            #   +1024        mlp1  128 x 8 x 1   [oc][ic]     = 1024 B
            #   +2048        sp    1 x 2 x 3 x 3  [oc][ic][ky][kx] = 18 B
            #                                                  ---------
            #                                                  2066 B
            #
            # NO padding between them: the HLS indexes them contiguously.  Each
            # tensor is quantized to int8 exactly like a conv, with a
            # per-OUTPUT-channel scale, and those scales plus the folded biases
            # go into the 275-float block that `export_fixed_params` writes at
            # the offset this layer's `ascale_off` carries.
            #
            # The values are the BN-FOLDED conv weights: ultralytics' Conv has
            # bias=False and the ONNX exporter has already fused each BatchNorm
            # into its conv's weight and bias, which is what the initializers
            # here are.  `verify_cbam_pl.py` re-derives the fold independently
            # from the PyTorch checkpoint and diffs it against these bytes.
            # ------------------------------------------------------------------
            offs = list(l.cbam_woff)
            if len(offs) != 6:
                raise RuntimeError(
                    f"OP_ATTENTION {l.out} has no CBAM weight offsets "
                    f"(cbam_woff={offs}).  This exporter now only supports the "
                    "CBAM operator (yolo26n_224_pl_fit.onnx); the old PS-side "
                    "C2PSA attention of yolo26n_224_rawhead.onnx -- qkv/pe/proj "
                    "convs around a CPU softmax -- has been retired, and its "
                    "layer rows cannot be quantized into the CBAM blob.  "
                    "Export from the pl_fit ONNX."
                )
            hidden = int(l.c0)
            c = int(l.ic)
            k = int(l.k)
            n0 = hidden * c
            n1 = c * hidden
            nsp = 2 * k * k

            def take(off: int, count: int) -> np.ndarray:
                return fp32[off : off + count]

            w0 = take(offs[0], n0).reshape(hidden, c).astype(np.float64)
            b0 = take(offs[1], hidden).astype(np.float64)
            w1 = take(offs[2], n1).reshape(c, hidden).astype(np.float64)
            b1 = take(offs[3], c).astype(np.float64)
            wsp = take(offs[4], nsp).reshape(2, k, k).astype(np.float64)
            bsp = take(offs[5], 1).astype(np.float64)

            s0 = np.maximum(np.max(np.abs(w0), axis=1) / 127.0, EPS)
            s1 = np.maximum(np.max(np.abs(w1), axis=1) / 127.0, EPS)
            ssp = np.maximum(np.abs(wsp).max() / 127.0, EPS)
            q0 = qint8(w0, s0[:, None])
            q1 = qint8(w1, s1[:, None])
            qsp = qint8(wsp, ssp)

            w_bytes = q0.tobytes() + q1.tobytes() + qsp.tobytes()
            if len(w_bytes) != n0 + n1 + nsp:
                raise RuntimeError(f"CBAM weight block is {len(w_bytes)} bytes")
            l.wq_off = len(blob) - 8
            blob += w_bytes
            while len(blob) % 4:
                blob += b"\x00"

            in_s = np.asarray(metas[l.inputs[0]][0], dtype=np.float64).reshape(-1)
            out_s = np.asarray(metas[l.out][0], dtype=np.float64).reshape(-1)
            # OFF_XSCALE is ONE float for both the input dequant and the output
            # requant of the frozen contract, while both tensors really carry
            # per-channel (scale, zero point) records -- the input's 128 scales
            # span 8-44x on this model and it is negative on 118-126 of its
            # channels.  No single scalar can therefore be exact; the mean is the
            # least-squares compromise and the standard per-channel records are
            # written to yolo_act_scales_i8.bin alongside it, which is what a
            # corrected operator should read.  See verify_cbam_pl.py for the
            # measured cost.
            xscale = float(in_s.mean()) if in_s.size else 1.0

            quant[idx] = {
                "wq": [q0, q1, qsp],
                "scale": [s0.astype(np.float32), s1.astype(np.float32),
                          np.asarray([ssp], dtype=np.float32)],
                "bias": [b0.astype(np.float32), b1.astype(np.float32),
                         bsp.astype(np.float32)],
                "xscale": xscale,
                "in_scale_mean": xscale,
                "out_scale_mean": float(out_s.mean()) if out_s.size else 1.0,
                "hidden": hidden,
                "k": k,
                "pad": l.pad if l.pad else k // 2,
                "mlp0_silu": CBAM_MLP0_SILU,
                "quant_max": [float(np.abs(q0).max()), float(np.abs(q1).max()),
                              float(np.abs(qsp).max())],
            }
            print(
                f"  cbam {l.out}: wq_off={l.wq_off} ({len(w_bytes)}B) "
                f"mlp0 scale[{s0.min():.3e},{s0.max():.3e}] "
                f"mlp1 scale[{s1.min():.3e},{s1.max():.3e}] sp={ssp:.3e} "
                f"xscale={xscale:.6g} (input has {in_s.size} per-channel scales, "
                f"spread {in_s.max()/in_s.min():.1f}x)"
            )
            continue
        if l.op != OP_CONV:
            continue
        w = fp32[l.woff : l.woff + l.oc * l.ic * l.k * l.k].reshape(l.oc, l.ic, l.k, l.k)
        b = fp32[l.boff : l.boff + l.oc]
        w_scale = np.max(np.abs(w), axis=(1, 2, 3)) / 127.0
        w_scale = np.maximum(w_scale, EPS)
        wq = qint8(w, w_scale[:, None, None, None])

        # Fold per-input-channel activation scale into the weights, then
        # requantize to int8 so the HLS kernel can use an int32 accumulator.
        in_scale = np.concatenate(
            [np.asarray(metas[name][0], dtype=np.float64).reshape(-1) for name in l.inputs]
        )
        groups = int(getattr(l, "groups", 1))
        if groups > 1:
            # DEPTHWISE (groups == oc, weight shape (C,1,3,3)).  The activation
            # scale array is per-CHANNEL (C entries) while l.ic == 1 per group,
            # so the plain per-input-channel broadcast is wrong.  Output channel
            # c consumes input channel c, so fold in_scale[c] into w[c] -- a
            # per-OUTPUT-channel broadcast instead.
            if in_scale.size != l.oc or l.ic != 1:
                raise RuntimeError(
                    f"unsupported grouped conv: groups={groups} ic={l.ic} "
                    f"oc={l.oc} in_scale={in_scale.size} for {l.out}"
                )
            fold = in_scale.reshape(-1, 1, 1, 1)
        else:
            if in_scale.size != l.ic:
                raise RuntimeError(
                    f"input scale count {in_scale.size} != ic {l.ic} for {l.out}"
                )
            fold = in_scale.reshape(1, -1, 1, 1)
        # The 910 exporter quantizes once before folding and then quantizes the
        # folded tensor again. Keep that legacy path for an exact A/B, while
        # direct_fold removes the first rounding step.
        if direct_fold:
            wf = w.astype(np.float64) * fold
        else:
            wf = wq.astype(np.float64) * fold
        w_fold_scale = np.max(np.abs(wf), axis=(1, 2, 3)) / 127.0
        w_fold_scale = np.maximum(w_fold_scale, EPS)
        wq2 = qint8(wf, w_fold_scale[:, None, None, None])
        w_comb = (w_fold_scale if direct_fold else w_fold_scale * w_scale).astype(np.float32)

        w_bytes = wq2.astype(np.int8).tobytes()
        pad = (-len(w_bytes)) % 4
        wq_off = len(blob) - 8
        blob += w_bytes + b"\x00" * pad
        bias_off = len(blob) - 8
        blob += b.astype("<f4").tobytes()
        ws_off = len(blob) - 8
        blob += w_comb.tobytes()

        l.wq_off = wq_off
        l.bias_off = bias_off
        l.ws_off = ws_off
        quant[idx] = (wq2, b.copy(), w_comb)
        nconv += 1

    blob[4:6] = struct.pack("<H", 2)
    blob[6:8] = struct.pack("<H", nconv)
    return blob, quant


def build_scale_bin(
    layers: list[Layer],
    metas: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[bytes, dict[str, int]]:
    blob = bytearray()
    records: dict[str, int] = {}

    def add(name: str) -> int:
        if name in records or name == "" or name.startswith("head_"):
            return records.get(name, 0)
        if name not in metas:
            return 0
        scale, zp = metas[name]
        off = len(blob)
        blob.extend(np.asarray(scale, dtype="<f4").tobytes())
        blob.extend(np.asarray(zp, dtype="<i2").tobytes())
        while len(blob) % 4:
            blob.extend(b"\x00")
        records[name] = off
        return off

    for l in layers:
        for i in l.inputs:
            add(i)
        add(l.out)
        if l.op == OP_SPLIT:
            add(l.out2)
    return bytes(blob), records


def write_i8_header(
    path: Path,
    layers: list[Layer],
    work_floats: int,
    weight_bytes: int,
    scale_offs: dict[str, int],
    scale_bytes: int = 0,
) -> None:
    # ---- geometry the PL kernel's scratch arrays must satisfy ---------------
    # Only the layers the PL actually executes matter (grouped convs and
    # OP_ATTENTION run on the PS).
    dense = [
        l
        for l in layers
        if l.op == OP_CONV and not (l.ic == 1 and l.oc > 1 and len(l.inputs) == 1)
    ]

    def ic_total(l: Layer) -> int:
        return sum(l.cin) if l.cin else l.ic

    row_need = max(
        (ic_total(l) * ((l.iw + 7) >> 3)) + 8 for l in dense
    ) if dense else 0
    tile_need = max((ic_total(l) * l.k * l.k for l in dense), default=0)
    maxic_need = max((ic_total(l) for l in dense), default=0)
    maxoc_need = max((l.oc for l in dense), default=0)
    maxow_need = max((l.ow for l in dense), default=0)

    lines = [
        "#ifndef YOLO_LAYERS_I8_H",
        "#define YOLO_LAYERS_I8_H",
        "",
        "// Generated by scripts/export_yolo_hls_int8.py - do not edit manually.",
        "",
        "#include <stdint.h>",
        "",
        "#define YOLO_I8_INPUT_C 3",
        f"#define YOLO_I8_INPUT_H {INPUT_H}",
        f"#define YOLO_I8_INPUT_W {INPUT_W}",
        "#define YOLO_I8_INPUT_BYTES (YOLO_I8_INPUT_C*YOLO_I8_INPUT_H*YOLO_I8_INPUT_W)",
        f"#define YOLO_I8_WORK_BYTES {work_floats}u",
        f"#define YOLO_I8_WEIGHT_BYTES {weight_bytes}u",
        f"#define YOLO_I8_ACT_SCALE_BYTES {scale_bytes}u",
        f"#define YOLO_I8_NUM_LAYERS {len(layers)}u",
        f"#define YOLO_I8_ANCHORS {ANCHORS}",
        f"#define YOLO_I8_BOX_CHANNELS {BOX_CHANNELS}",
        "#define YOLO_I8_PRED_ELEMS ((YOLO_I8_BOX_CHANNELS+1)*YOLO_I8_ANCHORS)",
        "#define YOLO_I8_PRED_BYTES (YOLO_I8_PRED_ELEMS*2)",
        "",
        "// ---- scratch-array geometry REQUIRED by this layer table ---------------",
        "// The HLS kernel sizes its row cache, weight tile and per-channel buffers",
        "// with #defines that were originally derived from the yolov8n table.  Those",
        "// are MODEL-DEPENDENT, and getting one too small writes past the end of a",
        "// partitioned BRAM array -- which csim survives (plain C arrays, benign",
        "// neighbour) and the board does not.  That is exactly how YOLO26 first",
        "// failed on hardware: the row cache needed 1032 bytes per bank and had 800,",
        "// so /model.16/cv1 wrote 232 bytes past the end of every row bank and every",
        "// detection vanished.  These values are emitted from the table so the kernel",
        "// can fail the BUILD instead of the board.",
        f"#define YOLO_I8_ROW_NEED {row_need}u",
        f"#define YOLO_I8_TILE_NEED {tile_need}u",
        f"#define YOLO_I8_MAXIC_NEED {maxic_need}u",
        f"#define YOLO_I8_MAXOC_NEED {maxoc_need}u",
        f"#define YOLO_I8_MAXOW_NEED {maxow_need}u",
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
        "// OP_ATTENTION executes in the PL: src/yolo_cbam_op.h, op_cbam_i8.",
        "//",
        "// The row is the CBAM block on a 128 x 7 x 7 tensor:",
        "//   ic, oc       128 (channel count; input == output)",
        "//   ih, iw       7, 7     oh, ow  7, 7",
        "//   k            spatial conv kernel (3), zero padded by k/2 = 1",
        "//   c0           MLP hidden width (8)   c1 = unused",
        "//   wq_off       byte offset, from the end of the 8-byte \"Y8I8\" header,",
        "//                of the CBAM INT8 weight block:",
        "//                  +0      mlp0  8 x 128 x 1  [oc][ic]    (1024 B)",
        "//                  +1024   mlp1  128 x 8 x 1  [oc][ic]    (1024 B)",
        "//                  +2048   sp    1 x 2 x 3 x 3 [oc][ic][ky][kx] (18 B)",
        "//                total 2066 B, no padding between the three tensors.",
        "//   bias_off     unused by this op (0)",
        "//   ascale_off   BYTE OFFSET INTO yolo_fixed_params_i8.bin of a 275-float",
        "//                block -- NOT an act-scale-blob offset, which is what the",
        "//                name means for every other op.  The block is:",
        "//                  [  0]   1   input activation dequant scale",
        "//                  [  1]   8   mlp0 per-output-channel weight scales",
        "//                  [  9]   8   mlp0 biases (BN folded)",
        "//                  [ 17] 128   mlp1 per-output-channel weight scales",
        "//                  [145] 128   mlp1 biases (BN folded)",
        "//                  [273]   1   spatial conv weight scale",
        "//                  [274]   1   spatial conv bias (BN folded)",
        "//                (YOLO_CBAM_OFF_* in src/yolo_cbam_op.h)",
        "//",
        "// IMPORTANT -- THE EXPORTED off_xscale FIELD CANNOT BE EXACT.  The contract",
        "// has ONE scalar for the layer's input dequant and output requant, while",
        "// this operator's input really carries a per-channel (scale, zero point)",
        "// record like every other tensor in the engine: its 128 scales span 8-44x",
        "// and 118-126 of its channels go negative.  This exporter writes the mean",
        "// of those scales and also emits the normal per-channel record for both",
        "// the input and the output tensor in yolo_act_scales_i8.bin.  See the",
        "// report / verify_cbam_pl.py for the measured cost of the approximation.",
        "",
        "typedef struct {",
        "    uint8_t op;",
        "    uint8_t nin;",
        "    uint8_t silu;",
        "    uint8_t k;",
        "    uint8_t stride;",
        "    uint8_t pad;",
        "    uint8_t out_target;",
        "    uint8_t fuse_add;",
        "    uint16_t c0;",
        "    uint16_t c1;",
        "    uint32_t oc;",
        "    uint32_t ic;",
        "    uint32_t oh;",
        "    uint32_t ow;",
        "    uint32_t ih;",
        "    uint32_t iw;",
        "    uint32_t wq_off;",
        "    uint32_t bias_off;",
        "    uint32_t ws_off;",
        "    uint32_t ascale_off;",
        "    uint32_t out_off;",
        "    uint32_t out_stride;",
        "    uint32_t out2_off;",
        "    uint32_t add_off;",
        "    uint32_t in_off[4];",
        "    uint16_t cin[4];",
        "    uint8_t in_up2x[4];",
        "    uint32_t in_ascale_off[4];",
        "} YoloLayerI8;",
        "",
        "static const YoloLayerI8 yolo_layers_i8[YOLO_I8_NUM_LAYERS] = {",
    ]

    for l in layers:
        in_offs = list(l.in_offs)
        cin = list(l.cin)
        in_up2x = list(getattr(l, "in_up2x", []))
        while len(in_offs) < 4:
            in_offs.append(0)
        while len(cin) < 4:
            cin.append(0)
        while len(in_up2x) < 4:
            in_up2x.append(0)
        ins = ", ".join(str(x) for x in in_offs)
        cins = ", ".join(str(x) for x in cin)
        up2x = ", ".join(str(x) for x in in_up2x)
        in_asc = []
        for i in l.inputs:
            in_asc.append(scale_offs.get(i, 0))
        if l.op == OP_ATTENTION:
            # Slot 1 is not an input for this op: op_cbam_i8 documents
            # `in_ascale_off[1]` as the OUTPUT tensor's per-channel record, used
            # when the caller passes the act-scale blob (source (a)).  The
            # dispatcher calls it without that pointer, so the operator uses its
            # own fixed-params record instead -- but emitting the offset keeps the
            # row honest and makes the alternative path work if it is ever wired
            # up.  0 means "not set", so only write a real offset.
            out_rec = scale_offs.get(l.out, 0)
            while len(in_asc) < 2:
                in_asc.append(0)
            in_asc[1] = out_rec
        while len(in_asc) < 4:
            in_asc.append(0)
        in_asc_str = ", ".join(str(x) for x in in_asc)

        wq = getattr(l, "wq_off", 0)
        bias = getattr(l, "bias_off", 0)
        ws = getattr(l, "ws_off", 0)
        if l.op == OP_ATTENTION:
            # op_cbam_i8 reads its 275-float parameter block out of the FIXED
            # PARAM blob, not out of the act-scale blob, so this one field is an
            # offset into yolo_fixed_params_i8.bin.  build_fixed_params() sets it
            # (it writes that blob) and must therefore run BEFORE this header is
            # generated -- see main().
            ascale = getattr(l, "ascale_off", 0)
            if not ascale:
                raise RuntimeError(
                    f"OP_ATTENTION {l.out} has no fixed-param offset: "
                    "build_fixed_params() must run before write_i8_header()"
                )
        else:
            ascale = scale_offs.get(l.out, 0)
        lines.append(
            "    {%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %du, %du, %du, %du, %du, %du, %du, %du, %du, %du, %du, %du, %du, %du, {%s}, {%s}, {%s}, {%s}},"
            % (
                l.op,
                len(l.inputs),
                l.silu,
                l.k,
                l.stride,
                l.pad,
                l.out_target,
                getattr(l, "fuse_add", 0),
                l.c0,
                l.c1,
                l.oc,
                l.ic,
                l.oh,
                l.ow,
                l.ih,
                l.iw,
                wq,
                bias,
                ws,
                ascale,
                l.out_off,
                l.out_stride,
                getattr(l, "out2_off", 0),
                getattr(l, "add_off", 0),
                ins,
                cins,
                up2x,
                in_asc_str,
            )
        )

    lines.extend(["};", "", "#endif", ""])
    path.write_text("\n".join(lines), encoding="ascii")


def attention_unit(
    qkv: np.ndarray,
    pe_w: np.ndarray,
    pe_b: np.ndarray,
    num_heads: int,
    key_dim: int,
    oh: int,
    ow: int,
) -> np.ndarray:
    """OLD PS-side C2PSA attention (rawhead ONNX).  Kept for reference only.

    NOTHING CALLS THIS ANY MORE.  Its only caller was the `OP_ATTENTION` branch
    of `run_int8_schedule`, which now runs `cbam_unit` -- the PL operator.  It is
    retained because `verify_attention_unit.py` still diffs it against the
    `yolo26n_224_rawhead.onnx` Attention subgraph, which is how the old 9-call
    schedule was checked.

    The whole YOLO26 C2PSA attention block, in FP32, on the CPU.

    This is the algorithm the PS has to run on the board, kept here as one named
    function so `verify_attention_unit.py` can diff it against ONNX Runtime
    bit-for-bit-ish instead of trusting the numbers that come out the far end of
    the quantization pipeline.

    Args:
        qkv:     [num_heads*(2*key_dim + head_dim), oh*ow] FP32 qkv conv output.
        pe_w:    [out_channels, 1, 3, 3] FP32 depthwise pe weight.
        pe_b:    [out_channels] FP32 pe bias.
        oh, ow:  spatial size; N = oh*ow tokens.

    Returns:
        [out_channels, oh*ow] FP32 = attention(v) + pe(v), i.e. the ONNX
        `/attn/Add` output, ready for the proj conv.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    ic, n = qkv.shape
    head_dim = ic // num_heads - 2 * key_dim
    out_ch = num_heads * head_dim
    if num_heads * (2 * key_dim + head_dim) != ic:
        raise RuntimeError(f"qkv width {ic} does not fit {num_heads} heads")
    if pe_w.shape[0] != out_ch or pe_w.shape[2:] != (3, 3):
        raise RuntimeError(f"pe weight {pe_w.shape} does not fit {out_ch} channels")
    if oh * ow != n:
        raise RuntimeError(f"token count {n} != {oh}x{ow}")

    attn_out = np.empty((out_ch, n), dtype=np.float64)
    v_all = np.empty((out_ch, n), dtype=np.float64)
    softmax_scale = 1.0 / np.sqrt(float(key_dim))
    per_head = 2 * key_dim + head_dim
    for h in range(num_heads):
        base = h * per_head
        qh = qkv[base : base + key_dim]                     # [key_dim, n]
        kh = qkv[base + key_dim : base + 2 * key_dim]       # [key_dim, n]
        vh = qkv[base + 2 * key_dim : base + per_head]      # [head_dim, n]
        logits = (qh.T @ kh) * softmax_scale                # [n, n]
        logits -= logits.max(axis=-1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=-1, keepdims=True)
        # (v @ attn^T): out[c, i] = sum_j v[c, j] * attn[i, j]
        attn_out[h * head_dim : (h + 1) * head_dim] = vh @ logits.T
        v_all[h * head_dim : (h + 1) * head_dim] = vh

    # ultralytics Attention.forward:
    #     x = (v @ attn^T).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    # pe takes v, NOT the attention output -- feeding the attention result here
    # is a silent, plausible-looking error (the shapes all fit).
    attn_img = attn_out.reshape(out_ch, oh, ow)
    vimg = v_all.reshape(out_ch, oh, ow)
    pe_pad = np.zeros((out_ch, oh + 2, ow + 2), dtype=np.float64)
    pe_pad[:, 1 : 1 + oh, 1 : 1 + ow] = vimg
    win = sliding_window_view(pe_pad, (3, 3), axis=(1, 2))
    pe = (
        np.einsum("oxy,ohwxy->ohw", pe_w[:, 0], win, dtype=np.float64)
        + pe_b[:, None, None]
    )
    return (attn_img + pe).reshape(out_ch, n)


# ---------------------------------------------------------------------------
# CBAM -- the PL implementation of OP_ATTENTION
# ---------------------------------------------------------------------------
# `cbam_pl.CBAM` is the authority on the maths, `yolo_cbam_op.h` on the blob
# layout.  This is the Python mirror of `op_cbam_i8`'s FLOAT maths, reading the
# quantized tensors the exporter actually writes, so a passing diff here means
# the exported weights reproduce the FP32 module and not merely that the export
# runs.
#
# TWO THINGS IN `op_cbam_i8` DO NOT MATCH THE MODULE IT CLAIMS TO IMPLEMENT --
# both are measured in verify_cbam_pl.py, both need an HLS change, and the
# exporter cannot compensate for either:
#
#  1. `mlp[0]` is `ultralytics.Conv(c, hidden, 1)`, whose DEFAULT activation is
#     SiLU.  The ONNX spells that out as Sigmoid->Mul after
#     `.../attn/mlp/mlp.0/conv/Conv`, and cbam_pl.CBAM therefore computes
#     `mlp0(v) = silu(W0 v + b0)`.  `op_cbam_i8` does not apply it.  On real
#     activations that is a 0.39-0.98 RELATIVE error on the block output.
#  2. the operator both reads its input as `byte * xscale` (no zero point, one
#     scalar for all 128 channels) and writes its output as a SIGNED byte,
#     while the rest of the engine stores activations as UNSIGNED with a
#     per-channel (scale, zero point) -- see `op_conv_i8`'s `rowA[..] - in_zp`.
#     The CBAM input is negative on 118-126 of its 128 channels and its
#     per-channel scales span 8-44x, so no single scalar can be right, and the
#     engine cannot read a signed byte back.
#
# CBAM_MLP0_SILU selects which of the two the Python schedule models: True (the
# default) matches cbam_pl.CBAM and the FP32 reference, False matches the HLS op
# as it stands today and is used to demonstrate the cost.
CBAM_MLP0_SILU = True

# When True, `run_int8_schedule` models `op_cbam_i8` EXACTLY AS WRITTEN -- single
# scalar input scale with no zero point, signed-byte output, no mlp0 SiLU --
# instead of the correct per-channel CBAM.  This is a diagnostic for attributing
# a csim/board number, not an export mode: it changes nothing the exporter
# writes, only what the Python schedule predicts.  See `--hls-cbam-io`.
HLS_CBAM_IO = False


def cbam_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe logistic; mirrors yolo_cbam_sigmoid's saturated ends."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def cbam_unit(x: np.ndarray, cbam: dict, mlp0_silu: bool | None = None) -> np.ndarray:
    """Float mirror of `op_cbam_i8`, in the RESIDUAL form the PL emits.

    Args:
        x:     [C, H, W] float64, the dequantized activation feeding the layer.
        cbam:  the payload `quantize_weights` produced for this layer, i.e. the
               int8 tensors and the per-output-channel scales that go into the
               weight blob and the 275-float parameter block.
        mlp0_silu: override for CBAM_MLP0_SILU.

    Returns:
        [C, H, W] float64 = x * (ca * sa) - x.  The residual is NOT optional:
        PSABlock.forward is `x + self.attn(x)`, so emitting the gate instead
        makes the block compute x + x*ca*sa and roughly doubles every
        activation.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if mlp0_silu is None:
        mlp0_silu = CBAM_MLP0_SILU

    def deq(idx: int) -> np.ndarray:
        w = np.asarray(cbam["wq"][idx], dtype=np.float64)
        s = np.asarray(cbam["scale"][idx], dtype=np.float64).reshape(-1)
        return w * s.reshape(-1, *([1] * (w.ndim - 1)))

    w0 = deq(0)                       # [hidden, C]
    b0 = np.asarray(cbam["bias"][0], dtype=np.float64)
    w1 = deq(1)                       # [C, hidden]
    b1 = np.asarray(cbam["bias"][1], dtype=np.float64)
    wsp = deq(2)                      # [2, k, k]
    bsp = np.asarray(cbam["bias"][2], dtype=np.float64)
    k = int(cbam["k"])
    pad = int(cbam["pad"])

    x = np.asarray(x, dtype=np.float64)
    c, hh, ww = x.shape
    if w0.shape != (cbam["hidden"], c) or w1.shape != (c, cbam["hidden"]):
        raise RuntimeError(
            f"cbam_unit: weight shapes {w0.shape}/{w1.shape} do not fit "
            f"C={c} hidden={cbam['hidden']}"
        )
    if wsp.shape != (2, k, k):
        raise RuntimeError(f"cbam_unit: spatial weight {wsp.shape} != (2,{k},{k})")

    # channel attention: ONE shared MLP over the average- and max-pooled vectors
    avg = x.mean(axis=(1, 2))
    mx = x.max(axis=(1, 2))

    def mlp0(v: np.ndarray) -> np.ndarray:
        y = w0 @ v + b0
        return y / (1.0 + np.exp(-y)) if mlp0_silu else y

    ca = cbam_sigmoid(w1 @ mlp0(avg) + b1 + w1 @ mlp0(mx) + b1)
    g = x * ca[:, None, None]

    # spatial attention: channel mean/max of the gated tensor -> 3x3 conv
    sp_in = np.stack([g.mean(axis=0), g.max(axis=0)], axis=0)      # [2, H, W]
    xp = np.zeros((2, hh + 2 * pad, ww + 2 * pad), dtype=np.float64)
    xp[:, pad : pad + hh, pad : pad + ww] = sp_in
    win = sliding_window_view(xp, (k, k), axis=(1, 2))
    sa = cbam_sigmoid(np.einsum("ixy,ihwxy->hw", wsp, win, dtype=np.float64) + bsp[0])

    # residual form, so PSABlock's `x + attn(x)` yields x * ca * sa
    return x * (ca[:, None, None] * sa[None]) - x


def run_int8_schedule(
    layers: list[Layer],
    quant: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    metas: dict[str, tuple[np.ndarray, np.ndarray]],
    inp_fp32: np.ndarray,
    work_floats: int,
    fp32_weights: dict[int, np.ndarray] | None = None,
    dump_dir: Path | None = None,
    snapshot: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view

    work = np.zeros(work_floats, dtype=np.uint8)
    pred = np.zeros(PRED_FLOATS, dtype=np.float32)

    # Input is [0,1], asymmetric uint8 with scale 1/255 and zero point 0.
    work[: inp_fp32.size] = qact(inp_fp32.reshape(-1), 1.0 / 255.0, 0)

    for idx, l in enumerate(layers):
        if l.op == OP_COPY:
            continue
        elif l.op == OP_CONV:
            wq, bias, w_scale = quant[idx]
            ic, ih, iw = l.ic, l.ih, l.iw
            k, pad, stride = l.k, l.pad, l.stride
            oh, ow = l.oh, l.ow
            oc = l.oc
            groups = int(getattr(l, "groups", 1))
            dw = groups > 1
            if dw and (l.ic != 1 or groups != l.oc):
                raise RuntimeError(
                    f"only depthwise (groups==oc, ic==1) is supported, got "
                    f"groups={groups} ic={l.ic} oc={l.oc} for {l.out}"
                )

            def _window(x):
                xp = np.zeros(
                    (x.shape[0], ih + 2 * pad, iw + 2 * pad), dtype=x.dtype
                )
                xp[:, pad : pad + ih, pad : pad + iw] = x
                return sliding_window_view(xp, (k, k), axis=(1, 2))[
                    :, ::stride, ::stride
                ]

            input_parts = []
            for name, off, channels, up2x in zip(
                l.inputs, l.in_offs, l.cin, l.in_up2x
            ):
                src_h = ih // 2 if up2x else ih
                src_w = iw // 2 if up2x else iw
                part = work[off : off + channels * src_h * src_w].reshape(
                    channels, src_h, src_w
                )
                if up2x:
                    part = np.repeat(np.repeat(part, 2, axis=1), 2, axis=2)
                input_parts.append(part)
            xq = np.concatenate(input_parts, axis=0)
            # ------------------------------------------------------------------
            # Anti-regression: numpy's einsum BROADCASTS size-1 axes, so a wrong
            # input-channel count does not raise -- it silently reads one channel
            # for every output channel.  That is exactly how YOLO26's depthwise
            # convs produced garbage (`cin` defaulted to [ic]=[1] instead of
            # [ic*groups]=[64]) with no error anywhere.  Check the accumulated
            # channel count and both einsum operands explicitly.
            # ------------------------------------------------------------------
            if xq.shape[0] != l.ic * groups:
                raise RuntimeError(
                    f"conv {l.out}: gathered {xq.shape[0]} input channels but the "
                    f"layer needs ic*groups={l.ic * groups} (cin={l.cin}, "
                    f"groups={groups})"
                )
            if dw:
                wmat = fp32_weights[idx][:, 0] if fp32_weights is not None else wq[:, 0]
                if wmat.shape[0] != l.oc or xq.shape[0] != l.oc:
                    raise RuntimeError(
                        f"depthwise conv {l.out}: weight channels {wmat.shape[0]} "
                        f"vs gathered input channels {xq.shape[0]} (both must be "
                        f"oc={l.oc}); broadcasting would hide this"
                    )
            if fp32_weights is not None:
                in_scale = np.concatenate(
                    [np.asarray(metas[name][0]).reshape(-1) for name in l.inputs]
                )
                in_zp = np.concatenate(
                    [np.asarray(metas[name][1]).reshape(-1) for name in l.inputs]
                )
                # depthwise: scales are per CHANNEL and pair with the output of
                # the same index; dense: per input channel, broadcast over oc
                in_s = in_scale.reshape(-1, 1, 1)
                in_z = in_zp.reshape(-1, 1, 1)
                x = (xq.astype(np.float64) - in_z) * in_s
                win = _window(x)
                wf32 = fp32_weights[idx]
                acc = (
                    np.einsum("oxy,ohwxy->ohw", wf32[:, 0], win, dtype=np.float64)
                    if dw
                    else np.einsum("oixy,ihwxy->ohw", wf32, win, dtype=np.float64)
                )
                real = acc + bias[:, None, None]
            else:
                in_zp = np.concatenate(
                    [np.asarray(metas[name][1]).reshape(-1) for name in l.inputs]
                )
                in_z = in_zp.reshape(-1, 1, 1)
                xc = xq.astype(np.int32) - in_z.astype(np.int32)
                win = _window(xc)
                wq32 = wq.astype(np.int32)
                acc = (
                    np.einsum("oxy,ohwxy->ohw", wq32[:, 0], win, dtype=np.int64)
                    if dw
                    else np.einsum("oixy,ihwxy->ohw", wq32, win, dtype=np.int64)
                )
                real = acc.astype(np.float64) * w_scale[:, None, None] + bias[:, None, None]
            if l.silu:
                real = silu(real)
            if l.out_target:
                for o in range(oc):
                    start = l.out_off + o * l.out_stride
                    pred[start : start + oh * ow] = real[o].reshape(-1).astype(np.float32)
            else:
                conv_out = l.quant_out if l.fuse_add else l.out
                out_s, out_zp = metas[conv_out]
                out_q = np.clip(
                    q_round(real / out_s.reshape(-1, 1, 1)) + out_zp.reshape(-1, 1, 1),
                    0,
                    255,
                ).astype(np.uint8)
                if l.fuse_add:
                    residual = work[
                        l.add_off : l.add_off + oc * oh * ow
                    ].reshape(oc, oh, ow)
                    res_s, res_zp = metas[l.add_input]
                    add_s, add_zp = metas[l.out]
                    conv_real = (
                        out_q.astype(np.float64) - out_zp.reshape(-1, 1, 1)
                    ) * out_s.reshape(-1, 1, 1)
                    residual_real = (
                        residual.astype(np.float64) - res_zp.reshape(-1, 1, 1)
                    ) * res_s.reshape(-1, 1, 1)
                    out_q = np.clip(
                        q_round(
                            (conv_real + residual_real)
                            / add_s.reshape(-1, 1, 1)
                        )
                        + add_zp.reshape(-1, 1, 1),
                        0,
                        255,
                    ).astype(np.uint8)
                for o in range(oc):
                    start = l.out_off + o * l.out_stride
                    work[start : start + oh * ow] = out_q[o].reshape(-1)
        elif l.op == OP_ATTENTION:
            # ------------------------------------------------------------------
            # YOLO26's C2PSA attention, now CBAM, executed by the PL's
            # `op_cbam_i8`.  One row replaces the whole
            # ReduceMean/ReduceMax/Conv/Sigmoid/Mul/Concat/Sub subgraph:
            #   ... (PL) -> [this] -> residual Add (PL)
            # Geometry comes off the row: ic = oc = 128 channels of 7x7, c0 = the
            # MLP hidden width (8), k = the spatial conv kernel (3, pad 1).
            # The weights are the INT8 tensors the exporter wrote into the weight
            # blob at wq_off, dequantized with the per-output-channel scales in
            # the 275-float block -- i.e. the same numbers the fabric sees, so
            # the diff here includes the quantization error and not just the
            # maths.
            # ------------------------------------------------------------------
            payload = quant[idx]
            n = l.oh * l.ow
            in_s, in_zp = metas[l.inputs[0]]
            if HLS_CBAM_IO:
                # ----------------------------------------------------------
                # MODEL `yolo_cbam_op.h` AS WRITTEN, for csim triage only.
                #
                # This is NOT a valid export setting: it reproduces the
                # operator's three known defects so that a csim/board number can
                # be ATTRIBUTED rather than guessed at.  Namely
                #   1. the input is dequantized as `byte * OFF_XSCALE` -- one
                #      scalar, no zero point -- instead of the per-channel
                #      (scale, zp) record every other layer uses;
                #   2. the output is stored as a SIGNED byte
                #      (`(uint8_t)(int8_t)q`) while the consumer reads
                #      `(byte - zp) * scale`, so every negative value comes back
                #      off by 256 codes;
                #   3. mlp[0]'s SiLU is never applied.
                # If a csim run lands on these numbers, the gap to the reference
                # is the operator, not the harness or the table.
                # ----------------------------------------------------------
                codes = work[
                    l.in_offs[0] : l.in_offs[0] + l.ic * n
                ].reshape(l.ic, l.oh, l.ow)
                x = codes.astype(np.float64) * payload["xscale"]
                real = cbam_unit(x, payload, mlp0_silu=False).reshape(l.oc, n)
                q_signed = np.clip(
                    q_round(real / payload["xscale"]), -128.0, 127.0
                )
                byte = np.where(q_signed < 0.0, q_signed + 256.0, q_signed)
                work[l.out_off : l.out_off + l.oc * n] = (
                    byte.astype(np.uint8).reshape(-1)
                )
            else:
                xq = work[l.in_offs[0] : l.in_offs[0] + l.ic * n].reshape(
                    l.ic, l.oh, l.ow
                )
                x = (
                    xq.astype(np.float64)
                    - np.asarray(in_zp, dtype=np.float64).reshape(-1, 1, 1)
                ) * np.asarray(in_s, dtype=np.float64).reshape(-1, 1, 1)

                real = cbam_unit(x, payload).reshape(l.oc, n)

                out_s, out_zp = metas[l.out]
                q = np.clip(
                    q_round(real / np.asarray(out_s).reshape(-1, 1))
                    + np.asarray(out_zp).reshape(-1, 1),
                    0,
                    255,
                ).astype(np.uint8)
                work[l.out_off : l.out_off + l.oc * n] = q.reshape(-1)
        elif l.op == OP_SPLIT:
            plane = l.oh * l.ow
            src = work[l.in_offs[0] : l.in_offs[0] + l.oc * plane].reshape(l.oc, plane)
            work[l.out_off : l.out_off + l.c0 * plane] = src[: l.c0].reshape(-1)
            work[l.out2_off : l.out2_off + l.c1 * plane] = src[l.c0 :].reshape(-1)
        elif l.op == OP_ADD:
            n = l.oc * l.oh * l.ow
            sa, za = metas[l.inputs[0]]
            sb, zb = metas[l.inputs[1]]
            a = (
                work[l.in_offs[0] : l.in_offs[0] + n].reshape(l.oc, -1).astype(np.float64)
                - za.reshape(-1, 1)
            ) * sa.reshape(-1, 1)
            b = (
                work[l.in_offs[1] : l.in_offs[1] + n].reshape(l.oc, -1).astype(np.float64)
                - zb.reshape(-1, 1)
            ) * sb.reshape(-1, 1)
            out_s, out_zp = metas[l.out]
            q = np.clip(
                q_round((a + b) / out_s.reshape(-1, 1)) + out_zp.reshape(-1, 1),
                0,
                255,
            ).astype(np.uint8)
            work[l.out_off : l.out_off + n] = q.reshape(-1)
        elif l.op == OP_CONCAT:
            plane = l.oh * l.ow
            dst = work[l.out_off : l.out_off + l.oc * plane].reshape(l.oc, plane)
            cum = 0
            for j, c in enumerate(l.cin):
                src = work[l.in_offs[j] : l.in_offs[j] + c * plane].reshape(c, plane)
                # The output metadata is the channel-wise concatenation of the
                # input metadata, so concatenation is an exact byte copy.
                dst[cum : cum + c] = src
                cum += c
        elif l.op == OP_UP2X:
            for c in range(l.oc):
                for h in range(l.oh):
                    for w in range(l.ow):
                        work[l.out_off + (c * l.oh + h) * l.ow + w] = work[
                            l.in_offs[0] + (c * l.ih + h // 2) * l.iw + w // 2
                        ]
        elif l.op == OP_MAXPOOL:
            ic, ih, iw = l.oc, l.oh, l.ow
            x = work[l.in_offs[0] : l.in_offs[0] + ic * ih * iw].reshape(ic, ih, iw)
            xp = np.zeros((ic, ih + 4, iw + 4), dtype=np.uint8)
            xp[:, 2 : 2 + ih, 2 : 2 + iw] = x
            win = sliding_window_view(xp, (5, 5), axis=(1, 2))
            out = win.max(axis=(-2, -1)).astype(np.uint8)
            work[l.out_off : l.out_off + ic * ih * iw] = out.reshape(-1)
        if snapshot is not None:
            # Snapshot the layer's own output region NOW.  Reading `work` after
            # the whole schedule finishes is worthless for diagnostics: the
            # allocator frees early buffers and later layers reuse those exact
            # offsets, so you end up diffing some unrelated activation that
            # happens to have a similar range (which looks like a permutation
            # bug rather than a stale-read bug).
            plane = l.oh * l.ow
            if l.op == OP_SPLIT:
                snapshot[(idx, "out")] = (
                    l.out_off,
                    work[l.out_off : l.out_off + l.c0 * plane].copy(),
                )
                snapshot[(idx, "out2")] = (
                    l.out2_off,
                    work[l.out2_off : l.out2_off + l.c1 * plane].copy(),
                )
            else:
                snapshot[(idx, "out")] = (
                    l.out_off,
                    work[l.out_off : l.out_off + l.oc * plane].copy(),
                )
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            (dump_dir / f"layer_{idx:03d}.bin").write_bytes(work.tobytes())
    return pred, work


def anchor_table(anchors: int, input_w: int):
    """(grid, stride) per scale, derived from the anchor count.

    YOLOv8n @640  -> 6400 + 1600 + 400  = 8400   (80/40/20, strides 8/16/32)
    YOLO26n @224  ->  784 +  196 +  49  = 1029   (28/14/7,  strides 8/16/32)
    """
    if anchors == 8400 and input_w == 640:
        return [(80, 8), (40, 16), (20, 32)]
    if anchors == 1029 and input_w == 224:
        return [(28, 8), (14, 16), (7, 32)]
    # Generic fallback: three scales of stride 8/16/32 whose grids sum to
    # `anchors`.  Solves a^2 + (a/2)^2 + (a/4)^2 = anchors for a.
    a = int(round((anchors / (1 + 0.25 + 0.0625)) ** 0.5))
    return [(a, 8), (a // 2, 16), (a // 4, 32)]


def anchor_info(idx: int):
    """(stride, cx, cy) for one anchor, grid centres at half-integers."""
    cum = 0
    for grid, stride in anchor_table(ANCHORS, INPUT_W):
        n = grid * grid
        if idx < cum + n:
            rem = idx - cum
            return stride, (rem % grid) + 0.5, (rem // grid) + 0.5
        cum += n
    raise IndexError(idx)


def decode_pred(pred: np.ndarray, img_w: int, img_h: int, conf_th: float = 0.25, iou_th: float = 0.45):
    box = pred[: BOX_CHANNELS * ANCHORS].reshape(BOX_CHANNELS, ANCHORS)
    conf = pred[BOX_CHANNELS * ANCHORS :]
    scale = min(INPUT_W / img_w, INPUT_H / img_h)
    new_w = int(img_w * scale + 0.5)
    new_h = int(img_h * scale + 0.5)
    pad_x = (INPUT_W - new_w) * 0.5
    pad_y = (INPUT_H - new_h) * 0.5

    cand = []
    for a in range(ANCHORS):
        score = 1.0 / (1.0 + np.exp(-conf[a]))
        if score < conf_th:
            continue
        stride, cx, cy = anchor_info(a)
        if REG_MAX == 1:
            # YOLO26: no DFL.  The head emits the four distances directly, in
            # GRID units -- the ONNX graph's Sub/Add/Div/Concat decode (which
            # turns them into xywh and multiplies by the stride) is NOT part of
            # the engine, so the PS has to do dist2bbox itself:
            #     x1 = (cx - l) * stride   ... exactly like the DFL branch below.
            dl, dt, dr, db = (float(box[i, a]) for i in range(4))
        else:
            dists = []
            for side in range(4):
                v = box[side * 16 : side * 16 + 16, a].astype(np.float64)
                v -= v.max()
                e = np.exp(v)
                p = e / e.sum()
                dists.append(float(np.dot(p, np.arange(16))))
            dl, dt, dr, db = dists
        x1 = (cx - dl) * stride
        y1 = (cy - dt) * stride
        x2 = (cx + dr) * stride
        y2 = (cy + db) * stride
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        if x2 <= x1 or y2 <= y1:
            continue
        cand.append([max(x1, 0.0), max(y1, 0.0), min(x2, img_w), min(y2, img_h), score])
    if not cand:
        return np.zeros((0, 5), dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    order = cand[:, 4].argsort()[::-1]
    cand = cand[order]

    def iou(a, b):
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    out = []
    for d in cand:
        if any(iou(d, f) > iou_th for f in out):
            continue
        out.append(d)
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 5), dtype=np.float32)


def box_match_iou(a: np.ndarray, b: np.ndarray, iou_th: float = 0.5) -> tuple[int, float]:
    used = np.zeros(len(b), dtype=bool)
    total = 0.0
    n = 0
    for da in a:
        best_i, best_v = -1, 0.0
        for j, db in enumerate(b):
            if used[j]:
                continue
            ix1 = max(da[0], db[0])
            iy1 = max(da[1], db[1])
            ix2 = min(da[2], db[2])
            iy2 = min(da[3], db[3])
            iw = ix2 - ix1
            ih = iy2 - iy1
            if iw <= 0 or ih <= 0:
                v = 0.0
            else:
                inter = iw * ih
                v = inter / ((da[2] - da[0]) * (da[3] - da[1]) + (db[2] - db[0]) * (db[3] - db[1]) - inter)
            if v > best_v:
                best_i, best_v = j, v
        if best_i >= 0 and best_v >= iou_th:
            used[best_i] = True
            total += best_v
            n += 1
    return n, total


def validate(
    onnx_path: Path,
    layers: list[Layer],
    quant: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    metas: dict[str, tuple[np.ndarray, np.ndarray]],
    work_floats: int,
    image_paths: list[Path],
    fp32_weights: dict[int, np.ndarray] | None = None,
    out_dir: Path | None = None,
    label_dir: Path | None = None,
    conf_th: float = 0.25,
    iou_th: float = 0.45,
    head_box_tensor: str = "/model.22/Concat_output_0",
    head_conf_tensor: str = "/model.22/Concat_1_output_0",
) -> dict:
    session = make_session(onnx_path, [head_box_tensor, head_conf_tensor])
    rows = []
    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        inp = letterbox_rgb_nchw(img)[None].astype(np.float32)
        raw_box, raw_conf = session.run(
            [head_box_tensor, head_conf_tensor],
            {"images": inp},
        )
        pred_q, _ = run_int8_schedule(
            layers, quant, metas, inp, work_floats, fp32_weights
        )
        pred_fp = np.zeros(PRED_FLOATS, dtype=np.float32)
        pred_fp[: BOX_CHANNELS * ANCHORS] = np.asarray(raw_box, dtype=np.float32).reshape(-1)
        pred_fp[BOX_CHANNELS * ANCHORS :] = np.asarray(raw_conf, dtype=np.float32).reshape(-1)
        box_err = float(np.max(np.abs(pred_q[: BOX_CHANNELS * ANCHORS] - pred_fp[: BOX_CHANNELS * ANCHORS])))
        conf_err = float(np.max(np.abs(pred_q[BOX_CHANNELS * ANCHORS :] - pred_fp[BOX_CHANNELS * ANCHORS :])))
        fp_dets = decode_pred(pred_fp, *img.size, conf_th=conf_th, iou_th=iou_th)
        q_dets = decode_pred(pred_q, *img.size, conf_th=conf_th, iou_th=iou_th)
        n_match, iou_sum = box_match_iou(fp_dets, q_dets)
        gt_iou = None
        tp = fp = fn = None
        if label_dir is not None:
            values = np.fromstring(
                (label_dir / f"{path.stem}.txt").read_text(encoding="utf-8"),
                sep=" ",
            )
            xc, yc, bw, bh = values[1:5]
            iw, ih = img.size
            gt = np.array(
                [
                    (xc - bw / 2) * iw,
                    (yc - bh / 2) * ih,
                    (xc + bw / 2) * iw,
                    (yc + bh / 2) * ih,
                ],
                dtype=np.float32,
            )
            gt_iou = 0.0
            for det in q_dets:
                ix1, iy1 = max(det[0], gt[0]), max(det[1], gt[1])
                ix2, iy2 = min(det[2], gt[2]), min(det[3], gt[3])
                inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
                union = (
                    (det[2] - det[0]) * (det[3] - det[1])
                    + (gt[2] - gt[0]) * (gt[3] - gt[1])
                    - inter
                )
                gt_iou = max(gt_iou, float(inter / union) if union > 0 else 0.0)
            tp = int(gt_iou >= 0.5)
            fn = 1 - tp
            fp = len(q_dets) - tp
        if out_dir is not None and idx == 0:
            (out_dir / "data").mkdir(parents=True, exist_ok=True)
            (out_dir / "data" / "verify_input_i8.bin").write_bytes(
                qact(inp.reshape(-1), 1.0 / 255.0, 0).astype(np.uint8).tobytes()
            )
            (out_dir / "data" / "verify_pred_i8.bin").write_bytes(pred_q.tobytes())
        rows.append(
            {
                "image": path.name,
                "box_max_abs_err": box_err,
                "conf_max_abs_err": conf_err,
                "det_fp": len(fp_dets),
                "det_i8": len(q_dets),
                "match": n_match,
                "mean_iou": float(iou_sum / n_match) if n_match else 0.0,
                "gt_best_iou": gt_iou,
                "tp50": tp,
                "fp50": fp,
                "fn50": fn,
            }
        )
    return rows


def pick_images(directory: Path, count: int) -> list[Path]:
    """Evenly spaced sample of a whole image directory, in NUMERIC name order.

    ``sorted(dir.glob("*.png"))[:N]`` is a trap: the names are lexicographic, so
    the "first 32" of data1000 are 1, 10, 100..129 -- a narrow slice of the
    acquisition space.  The static per-channel scales then badly under-cover the
    rest of the dataset (the head activations were clipped to a fifth of their
    real range and every detection vanished).  Sampling across the whole
    directory is what makes the calibration representative.
    """
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    ]

    def key(p: Path):
        s = p.stem
        return (0, int(s), s) if s.isdigit() else (1, 0, s)

    files.sort(key=key)
    if count <= 0 or count >= len(files):
        return files
    idx = sorted({int(round(x)) for x in np.linspace(0, len(files) - 1, count)})
    return [files[i] for i in idx]


def apply_derived_metas(layers: list[Layer], metas: dict) -> None:
    """Fill in the quantization metadata of ops that have no calibration entry.

    Split outputs slice the input meta; upsample / maxpool preserve it exactly;
    concatenation is a byte copy, so its meta is the channel-wise concatenation of
    its inputs.  Shared by main() and the diagnostics so the two cannot drift.
    """
    metas["images"] = (
        np.full(3, 1.0 / 255.0),
        np.zeros(3, dtype=np.int16),
    )
    metas["images_work"] = metas["images"]
    for l in layers:
        if l.op == OP_SPLIT:
            s, z = metas[l.inputs[0]]
            metas[l.out] = (s[: l.c0], z[: l.c0])
            metas[l.out2] = (s[l.c0 :], z[l.c0 :])
        elif l.op in (OP_UP2X, OP_MAXPOOL):
            metas[l.out] = metas[l.inputs[0]]
        elif l.op == OP_CONCAT:
            # Concatenation is a byte copy, not a numerical operation.  Keep
            # each input channel in its original quantization domain.  The
            # following convolution already folds per-input-channel scales
            # into its INT8 weights, so no element-wise requantization is
            # needed here.
            metas[l.out] = (
                np.concatenate([np.asarray(metas[n][0]) for n in l.inputs]),
                np.concatenate([np.asarray(metas[n][1]) for n in l.inputs]).astype(np.int16),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--val-images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument("--max-calib", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument("--validate", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--fp32-weights", action="store_true")
    parser.add_argument(
        "--box-channels",
        type=int,
        default=4,
        help="box values per anchor in the prediction buffer (YOLO26n: 4)",
    )
    parser.add_argument(
        "--direct-fold",
        action="store_true",
        help="fold activation scales into FP32 weights before one INT8 quantization",
    )
    parser.add_argument(
        "--cbam-no-silu",
        action="store_true",
        help="model OP_ATTENTION with the HLS op as it stands TODAY, which omits "
             "the SiLU after the 128->8 MLP that cbam_pl.CBAM has (ultralytics "
             "Conv's default activation). Default models the correct maths; this "
             "flag exists to measure what the omission costs. It does NOT change "
             "a single exported byte -- the weights are identical either way; "
             "only the Python reference differs.",
    )
    parser.add_argument(
        "--hls-cbam-io",
        action="store_true",
        help="diagnostic: predict what yolo_cbam_op.h AS WRITTEN would score, by "
             "modelling its scalar/zero-point-free input read, its signed-byte "
             "output write and its missing mlp0 SiLU. Changes no exported byte; "
             "it makes the Python schedule a predictor for a csim/board number so "
             "a mismatch can be attributed to the operator instead of guessed at.",
    )
    parser.add_argument(
        "--head-box-tensor",
        default="/model.23/Concat_output_0",
        help="RAW box head tensor for the FP32 reference: pre-decode l,t,r,b "
             "distances.  Do NOT use /model.23/Concat_2_output_0: that is the "
             "already-decoded xywh, which the engine never produces.",
    )
    parser.add_argument(
        "--head-conf-tensor",
        default="/model.23/Concat_1_output_0",
        help="raw class head tensor for the FP32 reference (the 1-channel class "
             "branch; note YOLO26's Concat_1 is classes, not boxes).",
    )
    args = parser.parse_args()

    global BOX_CHANNELS
    global CBAM_MLP0_SILU
    if args.cbam_no_silu:
        CBAM_MLP0_SILU = False
        globals()["CBAM_MLP0_SILU"] = False
        print(
            "WARN: --cbam-no-silu: modelling the HLS operator AS WRITTEN, without "
            "the SiLU that cbam_pl.CBAM's mlp[0] has. This is expected to break "
            "the FP32 agreement; it is a measurement, not a valid export."
        )
    if args.hls_cbam_io:
        globals()["HLS_CBAM_IO"] = True
        print(
            "WARN: --hls-cbam-io: the Python schedule now models yolo_cbam_op.h "
            "AS WRITTEN (scalar zero-point-free input read, signed-byte output, "
            "no mlp0 SiLU). The exported bytes are unchanged; only the predicted "
            "schedule output is. Use it to attribute a csim/board number."
        )
    if args.box_channels != BOX_CHANNELS:
        print(f"INFO: BOX_CHANNELS {BOX_CHANNELS} -> {args.box_channels} "
              f"(pred elems {(args.box_channels + 1) * ANCHORS})")
        BOX_CHANNELS = args.box_channels
        globals()["BOX_CHANNELS"] = args.box_channels
        # PRED_FLOATS is derived from BOX_CHANNELS at import time, so it has to be
        # refreshed too -- otherwise the pred buffer keeps the 65*1029 layout and
        # the head writes land in the wrong place.
        globals()["PRED_FLOATS"] = (args.box_channels + 1) * ANCHORS
    if args.box_channels % 4 == 0:
        globals()["REG_MAX"] = max(1, args.box_channels // 4)
        print(f"INFO: REG_MAX {REG_MAX} (box channels {args.box_channels} = 4 * reg_max)")

    layers, weight_bytes, shapes, work_floats, calib_names = build_schedule(args.onnx)

    image_paths = pick_images(args.images, args.max_calib)
    if not image_paths:
        raise SystemExit(f"no calibration images found under {args.images}")

    print(
        f"calibration images: {len(image_paths)}  storage tensors: {len(calib_names)}  "
        f"spread: {image_paths[0].name} .. {image_paths[-1].name}"
    )
    metas, maxabs, _ = calibrate(args.onnx, image_paths, calib_names, args.quantile)
    apply_derived_metas(layers, metas)

    blob, quant = quantize_weights(layers, weight_bytes, metas, args.direct_fold)
    fp32_weights = None
    if args.fp32_weights:
        fp32_blob = np.frombuffer(weight_bytes, dtype="<f4")
        fp32_weights = {}
        for idx, l in enumerate(layers):
            if l.op == OP_CONV:
                fp32_weights[idx] = fp32_blob[
                    l.woff : l.woff + l.oc * l.ic * l.k * l.k
                ].reshape(l.oc, l.ic, l.k, l.k)
    scale_blob, scale_offs = build_scale_bin(layers, metas)

    out_dir = args.out_dir
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "src").mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "data" / "yolo26n_weights_i8.bin"
    scale_path = out_dir / "data" / "yolo_act_scales_i8.bin"
    header_path = out_dir / "src" / "yolo_layers_i8.h"
    weights_path.write_bytes(blob)
    scale_path.write_bytes(scale_blob)

    # Integer-only bias/requant/Add parameters are derived from the freshly
    # generated v2 blobs and layer table.  Keeping this as a second, explicit
    # step makes the legacy blob layout readable while removing all FP math
    # from the FPGA kernel.
    #
    # It MUST run before write_i8_header: for OP_ATTENTION the header's
    # ascale_off field is a byte offset into THIS blob (op_cbam_i8 reads its
    # 275-float parameter block from `fixed_params`), and only
    # build_fixed_params knows where that block lands.  It sets
    # `layer.ascale_off`; write_i8_header raises if it is missing.
    from export_fixed_params import build_fixed_params

    fixed_path, fixed_header = build_fixed_params(out_dir, layers, quant, metas)
    print(f"fixed params : {fixed_path}")
    print(f"fixed header : {fixed_header}")

    write_i8_header(header_path, layers, work_floats, len(blob), scale_offs,
                    len(scale_blob))

    print(f"int8 weights: {len(blob)} bytes -> {weights_path}")
    print(f"act scales  : {len(scale_blob)} bytes -> {scale_path}")
    print(f"header       : {header_path}")
    print(f"work bytes   : {work_floats}")

    all_scales = np.concatenate([np.asarray(s).ravel() for s, _ in metas.values()])
    summary = {
        "calibration": {
            "images": len(image_paths),
            "quantile": args.quantile,
            "scale_min": float(all_scales.min()),
            "scale_max": float(all_scales.max()),
        },
        "weight_bytes": len(blob),
        "act_scale_bytes": len(scale_blob),
        "work_bytes": work_floats,
        "validation": [],
    }

    if not args.no_validate:
        # Spread the validation images too: taking the first N of the val split
        # measures one acquisition, not the model.
        val_paths = pick_images(args.val_images, args.validate)
        if not val_paths:
            raise SystemExit(f"no validation images found under {args.val_images}")
        print(
            f"validation images: {len(val_paths)}  "
            f"spread: {val_paths[0].name} .. {val_paths[-1].name}"
        )
        rows = validate(
            args.onnx,
            layers,
            quant,
            metas,
            work_floats,
            val_paths,
            fp32_weights,
            out_dir,
            args.val_labels,
            args.conf,
            args.iou,
            args.head_box_tensor,
            args.head_conf_tensor,
        )
        for r in rows:
            print(
                f"  {r['image']:12s} box_err={r['box_max_abs_err']:.3f} "
                f"conf_err={r['conf_max_abs_err']:.3f} "
                f"dets fp={r['det_fp']} i8={r['det_i8']} match={r['match']} "
                f"mean_iou={r['mean_iou']:.2f}"
            )
        summary["validation"] = rows
        if args.val_labels is not None:
            tp = sum(r["tp50"] for r in rows)
            fp = sum(r["fp50"] for r in rows)
            fn = sum(r["fn50"] for r in rows)
            gt_ious = np.asarray([r["gt_best_iou"] for r in rows])
            summary["detection_metrics_at_iou50"] = {
                "confidence_threshold": args.conf,
                "nms_iou_threshold": args.iou,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "mean_best_iou": float(gt_ious.mean()),
                "minimum_best_iou": float(gt_ious.min()),
            }

    meta_path = out_dir / "data" / "yolo_quant_i8.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"meta         : {meta_path}")

    # Integer-only bias/requant/Add parameters are derived from the freshly
    # generated v2 blobs and layer table.  Keeping this as a second, explicit
    # step makes the legacy blob layout readable while removing all FP math
    # from the FPGA kernel.
    from export_fixed_params import build_fixed_params

    fixed_path, fixed_header = build_fixed_params(out_dir, layers, quant, metas)
    print(f"fixed params : {fixed_path}")
    print(f"fixed header : {fixed_header}")


if __name__ == "__main__":
    main()

