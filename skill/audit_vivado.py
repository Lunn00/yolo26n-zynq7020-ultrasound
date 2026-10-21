"""Extract routed timing and utilization from Vivado text reports.

This checks report completeness and negative slack, not functional correctness.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def timing(text: str) -> dict[str, float]:
    match = re.search(r"WNS\(ns\).*?WHS\(ns\).*?\n\s*-+.*?\n\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+\d+\s+\d+\s+([-+]?\d+\.\d+)", text, re.S)
    if not match:
        raise ValueError("Vivado Design Timing Summary not found")
    return {"wns_ns": float(match[1]), "tns_ns": float(match[2]), "whs_ns": float(match[3])}


def resource(text: str, label: str) -> dict[str, float]:
    pattern = rf"^\|\s*{re.escape(label)}\s*\|([^\n]+)$"
    match = re.search(pattern, text, re.M)
    if not match:
        raise ValueError(f"Vivado utilization row not found: {label}")
    cells = [cell.strip() for cell in match[1].split("|")]
    if len(cells) < 5:
        raise ValueError(f"Incomplete utilization row: {label}")
    return {"used": float(cells[0]), "available": float(cells[3])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timing", required=True, type=Path)
    ap.add_argument("--util", required=True, type=Path)
    args = ap.parse_args()
    t = timing(args.timing.read_text(encoding="utf-8", errors="replace"))
    u = args.util.read_text(encoding="utf-8", errors="replace")
    report = {"timing": t, "resources": {
        "lut": resource(u, "Slice LUTs"),
        "bram_tile": resource(u, "Block RAM Tile"),
        "dsp": resource(u, "DSPs"),
    }}
    report["gate_pass"] = t["wns_ns"] >= 0 and t["whs_ns"] >= 0 and all(
        x["used"] <= x["available"] for x in report["resources"].values()
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
