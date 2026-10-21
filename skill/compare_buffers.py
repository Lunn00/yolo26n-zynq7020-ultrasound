"""Compare two captured binary buffers and report the first differing byte.

Usage: python skill/compare_buffers.py reference.bin board.bin
This tool compares bytes; the caller must establish that both files represent
the same tensor boundary, dtype, shape, layout, and input/model version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(path: Path) -> tuple[bytes, dict]:
    data = path.read_bytes()
    return data, {
        "name": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    ref, ref_summary = summarize(args.reference)
    got, got_summary = summarize(args.candidate)
    first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    if first is None and len(ref) != len(got):
        first = min(len(ref), len(got))
    result = {
        "reference": ref_summary,
        "candidate": got_summary,
        "equal": ref == got,
        "first_different_offset": first,
        "reference_byte": ref[first] if first is not None and first < len(ref) else None,
        "candidate_byte": got[first] if first is not None and first < len(got) else None,
    }
    print(json.dumps(result, indent=2))
    if not result["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
