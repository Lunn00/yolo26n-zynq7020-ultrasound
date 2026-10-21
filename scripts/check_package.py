"""Check project files and release artifacts without requiring FPGA tools."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
errors = []

required = (
    "LICENSE",
    "requirements.txt",
    "third_party_notices.md",
    "docs/reproduction.md",
    "fpga/hls/src/yolo_hls_i8_26.cpp",
    "fpga/hls/src/yolo_layers_i8.h",
    "fpga/hls/data/yolo26n_weights_i8.bin",
    "fpga/vivado/project_sources/system.bd",
    "prebuilt/yolo26n_vivado_ip_repository.zip",
    "software/ps/src/main.c",
    "sd_card/boot.bin",
    "sd_card/yolo_i8.bin",
    "prebuilt/system_wrapper.bit",
    "model_pipeline/model_conv1x1_fit.pt",
    "model_pipeline/model_conv1x1_fit.onnx",
    "model_pipeline/model_cbam_source.pt",
    "model_pipeline/source_construction/surgery_pl_native_head.py",
    "model_pipeline/source_construction/surgery_cbam_attention.py",
    "model_pipeline/source_construction/fit_cbam_to_attention.py",
    "skill/README.md",
    "skill/evidence_index.md",
    "skill/validation_matrix.md",
    "skill/experiment_template.md",
    "skill/workflows/data_model_contract.md",
    "skill/workflows/fpga_design_review.md",
    "skill/workflows/verification_release.md",
    "skill/audit_vivado.py",
    "skill/audit_dataset.py",
    "skill/compare_buffers.py",
    "skill/model_surgery.md",
    "skill/activation_snapshot.md",
    "skill/end_to_end_profile.md",
    "skill/pl_dataflow.md",
    "skill/ps_pl_protocol.md",
    "skill/domain_training.md",
    "verification/serial_run_128_raw_three_times.jpg",
    "verification/serial_run_128_raw_three_times.md",
    "skill/fpga-model-deployment/SKILL.md",
    "verification/dataset_audit_summary.json",
    "verification/release_identity.json",
)
for rel in required:
    if not (root / rel).is_file():
        errors.append(f"required file missing: {rel}")

tracked = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
)
publishable = {p.decode("utf-8") for p in tracked.split(b"\0") if p}
for rel in publishable:
    for part in Path(rel).parts:
        if part in {"README.md", "SKILL.md", "LICENSE", ".gitignore"}:
            continue
        # Vivado-generated IP instance names are referenced by BD/XCI and must stay intact.
        if part == "system_rst_ps7_0_100M_0" or part == "system_rst_ps7_0_100M_0.xci":
            continue
        if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-.'" for c in part):
            errors.append(f"nonportable file name: {rel}")
            break
identity = json.loads((root / "verification/release_identity.json").read_text(encoding="utf-8"))
for rel, expected in identity["artifacts_sha256"].items():
    path = root / rel
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(f"release identity mismatch: {rel}")

text_suffixes = {".md", ".py", ".ps1", ".tcl", ".c", ".h", ".cpp", ".json",
                 ".yaml", ".rpt", ".log", ".html", ".xml", ".bd", ".bda",
                 ".xci", ".v", ".xdc", ".csv", ".svg", ".txt"}
local_path = re.compile(rb"(?i)(?:[A-Z]:[/\\](?:Users|YoloProject|XilinxProject|Xilinx)|[/\\]home[/\\][A-Za-z0-9._-]+[/\\])")
email = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
for rel in sorted(publishable):
    path = root / rel
    if not path.is_file():
        continue
    data = path.read_bytes()
    if local_path.search(data) or email.search(data):
        errors.append(f"local path or email in: {rel}")
    if path.suffix.lower() in text_suffixes and re.search(rb"(?<![A-Za-z])[A-Z]:[/\\]", data):
        errors.append(f"absolute drive path in text: {rel}")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                errors.append(f"invalid archive member in: {rel}")
            for member in archive.infolist():
                if member.is_dir() or member.file_size > 100_000_000:
                    continue
                content = archive.read(member)
                if local_path.search(content) or email.search(content):
                    errors.append(f"local path or email in archive: {rel}!{member.filename}")

if (root / "software/ps/src/yolo_selftest_vectors.c").stat().st_size > 4096:
    errors.append("image-derived PS self-test vector is present in public source")

if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("Project files and release artifacts OK.")
