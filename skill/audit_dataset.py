"""Audit a YOLO dataset's image/label pairing and optional group split.

Example:
    python skill/audit_dataset.py --root datasets/carotid \
        --manifest datasets/carotid/split_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def inspect_split(root: Path, split: str) -> dict:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"missing image or label directory for {split}")
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}
    missing = sorted(images.keys() - labels.keys())
    orphan = sorted(labels.keys() - images.keys())
    empty, bad, boxes = [], [], 0
    for stem, path in labels.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        if not any(line.strip() for line in lines):
            empty.append(stem)
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            fields = line.split()
            try:
                cls = int(fields[0])
                coords = [float(x) for x in fields[1:]]
                valid = (
                    len(fields) == 5
                    and cls >= 0
                    and len(coords) == 4
                    and all(0 <= x <= 1 for x in coords)
                    and coords[2] > 0
                    and coords[3] > 0
                )
            except (ValueError, IndexError):
                valid = False
            if valid:
                boxes += 1
            else:
                bad.append(f"{stem}.txt:{n}")
    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": boxes,
        "missing_labels": missing,
        "orphan_labels": orphan,
        "empty_labels": sorted(empty),
        "invalid_label_lines": bad,
    }


def inspect_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {"train": set(), "val": set()}
    ids = {"train": set(), "val": set()}
    for row in rows:
        split = row.get("split")
        if split not in groups:
            raise ValueError(f"unknown split: {split!r}")
        groups[split].add(row["group"])
        image_id = row.get("id") or Path(row.get("renamed_image", "")).stem
        if image_id in ids[split]:
            raise ValueError(f"duplicate image ID in {split}: {image_id}")
        ids[split].add(image_id)
    return {
        "rows": len(rows),
        "train_groups": len(groups["train"]),
        "val_groups": len(groups["val"]),
        "group_overlap": sorted(groups["train"] & groups["val"]),
        "id_overlap": sorted(ids["train"] & ids["val"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = {
        "train": inspect_split(args.root, "train"),
        "val": inspect_split(args.root, "val"),
    }
    if args.manifest:
        result["manifest"] = inspect_manifest(args.manifest)
    result["gate_pass"] = all(
        not result[split][key]
        for split in ("train", "val")
        for key in ("missing_labels", "orphan_labels", "invalid_label_lines")
    )
    if args.manifest:
        result["gate_pass"] &= not result["manifest"]["group_overlap"]
        result["gate_pass"] &= not result["manifest"]["id_overlap"]
        result["gate_pass"] &= result["manifest"]["rows"] == (
            result["train"]["images"] + result["val"]["images"]
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
