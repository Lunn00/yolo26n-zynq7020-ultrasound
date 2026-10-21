"""Record hashes linking the delivered model, FPGA, software and boot image."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
artifacts = (
    "model_pipeline/model_conv1x1_fit.pt",
    "model_pipeline/model_conv1x1_fit.onnx",
    "fpga/hls/src/yolo_hls_i8_26.cpp",
    "fpga/hls/src/yolo_layers_i8.h",
    "fpga/hls/data/yolo26n_weights_i8.bin",
    "fpga/hls/data/yolo_fixed_params_i8.bin",
    "fpga/vivado/project_sources/validated_bd/system/system.bd",
    "prebuilt/hls_ip_export.zip",
    "prebuilt/system_wrapper.bit",
    "prebuilt/fsbl.elf",
    "prebuilt/yolo26_lcd_ui.elf",
    "sd_card/boot.bin",
)
record = {
    "model": "YOLO26n INT8",
    "input_shape": [224, 224, 3],
    "hls_layers": 111,
    "device": "xc7z020clg400-2",
    "toolchain": "Vivado/Vitis/Vitis HLS 2021.2",
    "clock_mhz": 150,
    "artifacts_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in artifacts
    },
    "evidence": {
        "implementation": "fpga/vivado/reports/",
        "csim": "verification/csim_console.log",
        "board": "verification/serial_run_128_raw_three_times.md",
    },
}
target = root / "verification/release_identity.json"
target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Updated {target.relative_to(root)}: {len(artifacts)} artifacts")
