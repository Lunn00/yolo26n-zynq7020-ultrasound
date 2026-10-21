#!/usr/bin/env python3
"""Build integer-only run-time parameters for the optimized HLS kernel.

This module is called by export_yolo_hls_int8.py after calibration and INT8
weight folding.  Floating-point work is deliberately confined to this offline
export step; the generated FPGA kernel consumes only integers.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np


HEAD_FRAC_BITS = 10
Q31_BITS = 31


def _quantize_multiplier(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return signed Q31 multipliers and right shifts for positive ratios."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mul = np.zeros(values.size, dtype=np.int32)
    rshift = np.zeros(values.size, dtype=np.int32)
    for i, value in enumerate(values):
        if not np.isfinite(value) or value <= 0.0:
            continue
        significand, exponent = np.frexp(float(value))
        q = int(np.rint(significand * (1 << Q31_BITS)))
        if q == (1 << Q31_BITS):
            q //= 2
            exponent += 1
        mul[i] = np.int32(q)
        rshift[i] = np.int32(Q31_BITS - exponent)
    return mul, rshift


def _clip_i32(values: np.ndarray) -> np.ndarray:
    info = np.iinfo(np.int32)
    return np.clip(np.rint(values), info.min, info.max).astype(np.int32)


def build_fixed_params(
    out_dir: Path,
    layers: list[Any],
    quant: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    metas: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[Path, Path]:
    """Write the fixed parameter blob and its compile-time descriptor table."""

    blob = bytearray(struct.pack("<4sHH", b"Y8QP", 1, len(layers)))
    descs: list[list[int]] = []

    def append(values: np.ndarray, dtype: str) -> int:
        while len(blob) & 3:
            blob.append(0)
        off = len(blob)
        blob.extend(np.asarray(values, dtype=dtype).tobytes())
        return off

    for idx, layer in enumerate(layers):
        # Conv input/quant fields followed by optional fused-Add fields:
        # in0_zp, in1_zp, conv_out_zp, bias, mul0, shift0, mul1, shift1,
        # add_out_zp, add_mul0, add_shift0, add_mul1, add_shift1.
        desc = [0] * 13
        if layer.op == 1:  # OP_CONV
            _, bias, combined_scale = quant[idx]
            combined_scale = np.asarray(combined_scale, dtype=np.float64)
            bias_acc = _clip_i32(np.asarray(bias, dtype=np.float64) / combined_scale)
            in_zp = np.concatenate(
                [np.asarray(metas[name][1], dtype=np.int16).reshape(-1)
                 for name in layer.inputs]
            )
            desc[0] = append(in_zp, "<i2")
            desc[3] = append(bias_acc, "<i4")

            if layer.out_target:
                # Final box/conf logits are emitted as signed INT16 Q10.
                mul0, shift0 = _quantize_multiplier(
                    combined_scale * float(1 << HEAD_FRAC_BITS)
                )
            elif layer.silu:
                # Stage 1 maps the accumulator into signed Q10 for integer
                # SiLU; stage 2 maps Q10 SiLU output into the UINT8 tensor.
                conv_out = layer.quant_out if layer.fuse_add else layer.out
                out_scale, out_zp = metas[conv_out]
                out_scale = np.asarray(out_scale, dtype=np.float64)
                desc[2] = append(np.asarray(out_zp, dtype=np.int16), "<i2")
                mul0, shift0 = _quantize_multiplier(
                    combined_scale * float(1 << HEAD_FRAC_BITS)
                )
                mul1, shift1 = _quantize_multiplier(
                    1.0 / (float(1 << HEAD_FRAC_BITS) * out_scale)
                )
                desc[6] = append(mul1, "<i4")
                desc[7] = append(shift1, "<i4")
            else:
                conv_out = layer.quant_out if layer.fuse_add else layer.out
                out_scale, out_zp = metas[conv_out]
                out_scale = np.asarray(out_scale, dtype=np.float64)
                desc[2] = append(np.asarray(out_zp, dtype=np.int16), "<i2")
                mul0, shift0 = _quantize_multiplier(combined_scale / out_scale)

            desc[4] = append(mul0, "<i4")
            desc[5] = append(shift0, "<i4")

            if layer.fuse_add:
                conv_scale, _ = metas[layer.quant_out]
                residual_scale, residual_zp = metas[layer.add_input]
                add_scale, add_zp = metas[layer.out]
                desc[1] = append(np.asarray(residual_zp, dtype=np.int16), "<i2")
                desc[8] = append(np.asarray(add_zp, dtype=np.int16), "<i2")
                add_mul0, add_shift0 = _quantize_multiplier(
                    np.asarray(conv_scale, dtype=np.float64)
                    / np.asarray(add_scale, dtype=np.float64)
                )
                add_mul1, add_shift1 = _quantize_multiplier(
                    np.asarray(residual_scale, dtype=np.float64)
                    / np.asarray(add_scale, dtype=np.float64)
                )
                desc[9] = append(add_mul0, "<i4")
                desc[10] = append(add_shift0, "<i4")
                desc[11] = append(add_mul1, "<i4")
                desc[12] = append(add_shift1, "<i4")

        elif layer.op == 7:  # OP_ATTENTION (CBAM, src/yolo_cbam_op.h)
            # op_cbam_i8 reads its parameter block at
            # `fixed_params + L.ascale_off`, so THIS blob -- not
            # yolo_act_scales_i8.bin, which is what the field means for every
            # other op -- owns it.  The block is:
            #
            #   [   0, 1100)   275 floats, YOLO_CBAM_OFF_* (float units):
            #                    0  reserved (OFF_XSCALE; a single scalar cannot
            #                       describe a per-channel tensor, nothing reads it)
            #                    1..9   mlp0 per-output-channel weight scales
            #                    9..17  mlp0 biases (BN folded)
            #                   17..145 mlp1 per-output-channel weight scales
            #                  145..273 mlp1 biases (BN folded)
            #                  273,274  spatial conv weight scale / bias
            #   [1100, 3736)   2636 bytes of PER-CHANNEL TENSOR RECORDS, the
            #                  same layout yolo_attn_ps.h reads out of the
            #                  act-scale blob: float scale[n] then int16 zero[n].
            #                  YOLO_CBAM_REC_* gives the four offsets, relative to
            #                  the block base:
            #                    1100  in_scale[128]   1612  in_zp[128]
            #                    1868  out_scale[128]  2380  out_zp[128]
            #
            # The records are what the operator actually dequantises and
            # requantises with; a zeroed record is refused (`yolo_cbam_rec_ok`),
            # so an exporter that forgets this half fails loudly instead of
            # producing garbage.  Keeping them in the layer's own block (rather
            # than reusing the act-scale offsets) means the kernel does not need
            # the act-scale blob at all -- it never receives that pointer.
            p = quant[idx]
            block = np.zeros(275, dtype=np.float32)
            block[0] = np.float32(p["xscale"])
            block[1:9] = np.asarray(p["scale"][0], dtype=np.float32).reshape(-1)
            block[9:17] = np.asarray(p["bias"][0], dtype=np.float32).reshape(-1)
            block[17:145] = np.asarray(p["scale"][1], dtype=np.float32).reshape(-1)
            block[145:273] = np.asarray(p["bias"][1], dtype=np.float32).reshape(-1)
            block[273] = np.float32(np.asarray(p["scale"][2]).reshape(-1)[0])
            block[274] = np.float32(np.asarray(p["bias"][2]).reshape(-1)[0])

            c = int(layer.ic)
            if c != 128 or int(layer.oc) != c:
                raise RuntimeError(
                    f"OP_ATTENTION {layer.out}: the CBAM parameter block is fixed "
                    f"to 128 channels, got ic={c} oc={layer.oc}"
                )
            in_s, in_z = metas[layer.inputs[0]]
            out_s, out_z = metas[layer.out]

            def zeroed(zero) -> bytes:
                z = np.clip(np.asarray(zero, dtype=np.int64).reshape(-1), 0, 255)
                if z.size != c:
                    raise RuntimeError(
                        f"OP_ATTENTION {layer.out}: zero-point record has "
                        f"{z.size} entries, expected {c}"
                    )
                return z.astype("<i2").tobytes()

            def scaled(scale) -> bytes:
                s = np.asarray(scale, dtype="<f4").reshape(-1)
                if s.size != c:
                    raise RuntimeError(
                        f"OP_ATTENTION {layer.out}: scale record has {s.size} "
                        f"entries, expected {c}"
                    )
                return s.tobytes()

            raw = bytearray(block.astype("<f4").tobytes())
            raw += scaled(in_s)
            raw += zeroed(in_z)
            raw += scaled(out_s)
            raw += zeroed(out_z)
            expected = 275 * 4 + (4 + 2 + 4 + 2) * c
            if len(raw) != expected:
                raise RuntimeError(
                    f"OP_ATTENTION {layer.out}: parameter block is {len(raw)} "
                    f"bytes, the operator's layout needs {expected}"
                )
            # write_i8_header() reads back layer.ascale_off for the row's
            # ascale_off field.
            layer.ascale_off = append(np.frombuffer(bytes(raw), dtype=np.uint8),
                                      "<u1")

        elif layer.op == 3:  # OP_ADD
            sa, za = metas[layer.inputs[0]]
            sb, zb = metas[layer.inputs[1]]
            so, zo = metas[layer.out]
            desc[0] = append(np.asarray(za, dtype=np.int16), "<i2")
            desc[1] = append(np.asarray(zb, dtype=np.int16), "<i2")
            desc[2] = append(np.asarray(zo, dtype=np.int16), "<i2")
            mul0, shift0 = _quantize_multiplier(
                np.asarray(sa, dtype=np.float64) / np.asarray(so, dtype=np.float64)
            )
            mul1, shift1 = _quantize_multiplier(
                np.asarray(sb, dtype=np.float64) / np.asarray(so, dtype=np.float64)
            )
            desc[4] = append(mul0, "<i4")
            desc[5] = append(shift0, "<i4")
            desc[6] = append(mul1, "<i4")
            desc[7] = append(shift1, "<i4")

        descs.append(desc)

    data_dir = out_dir / "data"
    src_dir = out_dir / "src"
    data_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "yolo_fixed_params_i8.bin"
    data_path.write_bytes(blob)

    lines = [
        "#ifndef YOLO_FIXED_PARAMS_I8_H",
        "#define YOLO_FIXED_PARAMS_I8_H",
        "",
        "// Generated by scripts/export_fixed_params.py - do not edit manually.",
        "#include <stdint.h>",
        "",
        f"#define YOLO_I8_FIXED_PARAM_BYTES {len(blob)}u",
        f"#define YOLO_I8_HEAD_FRAC_BITS {HEAD_FRAC_BITS}",
        "",
        "typedef struct {",
        "    uint32_t in0_zp_off;",
        "    uint32_t in1_zp_off;",
        "    uint32_t out_zp_off;",
        "    uint32_t bias_off;",
        "    uint32_t mul0_off;",
        "    uint32_t shift0_off;",
        "    uint32_t mul1_off;",
        "    uint32_t shift1_off;",
        "    uint32_t add_out_zp_off;",
        "    uint32_t add_mul0_off;",
        "    uint32_t add_shift0_off;",
        "    uint32_t add_mul1_off;",
        "    uint32_t add_shift1_off;",
        "} YoloFixedParamDesc;",
        "",
        "static const YoloFixedParamDesc yolo_fixed_desc[YOLO_I8_NUM_LAYERS] = {",
    ]
    lines.extend("    {%s}," % ", ".join(f"{v}u" for v in desc) for desc in descs)
    lines.extend(["};", "", "#endif", ""])
    header_path = src_dir / "yolo_fixed_params_i8.h"
    header_path.write_text("\n".join(lines), encoding="ascii")
    return data_path, header_path


if __name__ == "__main__":
    raise SystemExit("Run export_yolo_hls_int8.py; it generates fixed parameters automatically.")
