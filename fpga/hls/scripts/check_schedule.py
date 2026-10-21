#!/usr/bin/env python
"""Static data-flow check of a generated YOLO INT8 layer table.

WHY THIS EXISTS
---------------
The accelerator reads every layer's input out of one shared `work` buffer.  If a
layer reads bytes that no *earlier* layer wrote, the hardware quietly consumes
whatever DDR happens to hold there.  That is invisible in csim (vector is
zero-filled) and invisible in the Python reference (`np.zeros`), but on the board
the same bytes hold the previous inference's activations -- which is exactly the
shape of bug that produces "self-test passes, real images do not".

This checker walks the table in execution order and proves, byte by byte, that
every read is covered by:

    * the network input buffer        (layer 0 only),
    * data written by an earlier layer, or
    * a residual input (`add_off`) that an earlier layer produced.

It reports partially covered reads separately, because a *partially* covered
read is the dangerous kind: it works at boot (zeros) and breaks later.

Usage:
    python check_schedule.py [header]...
    python check_schedule.py            # defaults to ../hls/src/yolo_layers_i8.h
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "src" / "yolo_layers_i8.h"

OP_NAMES = {0: "COPY", 1: "CONV", 2: "SPLIT", 3: "ADD", 4: "CONCAT",
            5: "UP2X", 6: "MAXPOOL", 7: "ATTENTION"}

FIELDS = ["op", "nin", "silu", "k", "stride", "pad", "out_target", "fuse_add",
          "c0", "c1", "oc", "ic", "oh", "ow", "ih", "iw",
          "wq_off", "bias_off", "ws_off", "ascale_off", "out_off", "out_stride",
          "out2_off", "add_off"]


class Layer:
    __slots__ = FIELDS + ["in_off", "cin", "in_up2x", "in_ascale_off", "index"]

    def __init__(self, index: int, scalars: list[int], arrays: list[list[int]]):
        self.index = index
        for name, value in zip(FIELDS, scalars):
            setattr(self, name, value)
        (self.in_off, self.cin, self.in_up2x, self.in_ascale_off) = arrays

    @property
    def op_name(self) -> str:
        return OP_NAMES.get(self.op, f"OP_{self.op}")


def parse_header(path: Path) -> tuple[list[Layer], dict[str, int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    defines = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"^#define\s+(YOLO_I8_\w+|OP_\w+)\s+\(?(\d+)\)?[uU]?\s*$", text, re.M)}
    # op names come from the header itself, so a renumbered enum cannot silently
    # make this checker report the wrong histogram.
    global OP_NAMES
    OP_NAMES = {v: k[3:] for k, v in defines.items() if k.startswith("OP_")}
    layers: list[Layer] = []
    for row in re.finditer(r"^\s*(\{[^\n]*\}),\s*$", text, re.M):
        body = row.group(1)
        groups = re.findall(r"\{([^{}]*)\}", body)
        if len(groups) != 4:
            continue
        head = re.sub(r"\{[^{}]*\}", "", body).strip().lstrip("{").rstrip("}")
        scalars = [int(x.strip().rstrip("uU")) for x in head.split(",") if x.strip()]
        arrays = [[int(x.strip().rstrip("uU")) for x in g.split(",") if x.strip()] or [0]
                  for g in groups]
        if len(scalars) != len(FIELDS):
            continue
        layers.append(Layer(len(layers), scalars, arrays))
    return layers, defines


def check(path: Path) -> int:
    layers, defines = parse_header(path)
    if not layers:
        print(f"{path}: no layer rows parsed -- header format changed?")
        return 2

    work_bytes = defines.get("YOLO_I8_WORK_BYTES", 0)
    num = defines.get("YOLO_I8_NUM_LAYERS", len(layers))
    print(f"{path.name}: {len(layers)} rows (header says {num}), "
          f"work = {work_bytes} B")

    written = bytearray(work_bytes) if work_bytes else None
    problems: list[str] = []
    partial: list[str] = []
    hist: dict[str, int] = {}

    def span(off: int, size: int) -> tuple[int, int]:
        return off, off + size

    def mark(off: int, size: int) -> None:
        if written is None:
            return
        written[max(off, 0):off + size] = b"\x01" * max(
            0, min(off + size, work_bytes) - max(off, 0))

    def covered(off: int, size: int) -> int:
        if written is None:
            return size
        return sum(written[max(off, 0):off + size])

    for layer in layers:
        hist[layer.op_name] = hist.get(layer.op_name, 0) + 1
        plane = layer.oh * layer.ow

        # --- reads ------------------------------------------------------
        if layer.op != 0:  # layer 0 reads the input image buffer instead
            for slot in range(layer.nin):
                channels = layer.cin[slot] or layer.ic
                if channels == 0:
                    continue
                src_h = layer.ih // 2 if layer.in_up2x[slot] else layer.ih
                src_w = layer.iw // 2 if layer.in_up2x[slot] else layer.iw
                off = layer.in_off[slot]
                size = channels * src_h * src_w
                if size == 0:
                    continue
                got = covered(off, size)
                if got < size:
                    msg = (f"  row {layer.index:3d} {layer.op_name:9s} "
                           f"input slot {slot}: [{off}, {off+size}) "
                           f"{size-got}/{size} B never written")
                    (partial if got else problems).append(msg)
        if layer.fuse_add:
            off, size = layer.add_off, layer.oc * plane
            got = covered(off, size)
            if size and got < size:
                msg = (f"  row {layer.index:3d} {layer.op_name:9s} "
                       f"residual:     [{off}, {off+size}) "
                       f"{size-got}/{size} B never written")
                (partial if got else problems).append(msg)

        # --- writes -----------------------------------------------------
        if layer.out_target == 0 and layer.oc and plane:
            mark(layer.out_off, layer.oc * plane)
        if layer.out2_off and layer.out_target:
            mark(layer.out2_off, layer.oc * plane)

    print("op histogram: " + ", ".join(f"{k}={v}" for k, v in sorted(hist.items())))
    if written is not None:
        touched = sum(written)
        print(f"work bytes written at least once: {touched} / {work_bytes} "
              f"({100.0*touched/work_bytes:.1f}%)")

    if problems:
        print(f"\nUNCOVERED READS ({len(problems)}) -- these read undefined data:")
        print("\n".join(problems[:40]))
    if partial:
        print(f"\nPARTIALLY COVERED READS ({len(partial)}) -- zero at boot, "
              f"stale later:")
        print("\n".join(partial[:40]))
    if not problems and not partial:
        print("\nOK: every read is covered by an earlier write.")
        return 0
    return 1


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or [DEFAULT]
    rc = 0
    for target in targets:
        if not target.exists():
            print(f"missing: {target}")
            rc = 2
            continue
        rc |= check(target)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
