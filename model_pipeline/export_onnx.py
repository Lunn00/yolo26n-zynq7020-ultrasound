"""Export the CBAM-free candidate and fail if unsupported attention ops remain."""

from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

import onnx
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=HERE / "model_conv1x1_fit.pt")
    args = parser.parse_args()
    dest = args.weights.with_suffix(".onnx")
    model = YOLO(str(args.weights))
    path = Path(model.export(format="onnx", imgsz=224, opset=12,
                             simplify=False, dynamic=False, end2end=False))
    if path.resolve() != dest.resolve():
        dest.write_bytes(path.read_bytes())
    graph = onnx.load(str(dest)).graph
    counts = Counter(node.op_type for node in graph.node)
    grouped = [node.name for node in graph.node if node.op_type == "Conv" and
               any(a.name == "group" and a.i > 1 for a in node.attribute)]
    forbidden = {"ReduceMean", "ReduceMax", "Softmax", "MatMul", "HardSigmoid"}
    present = sorted(forbidden & counts.keys())
    if grouped or present:
        raise RuntimeError(f"unsupported architecture: grouped={grouped}, ops={present}")
    record = {"nodes": len(graph.node), "ops": dict(counts),
              "grouped_conv_count": len(grouped), "forbidden_ops": present}
    (HERE / f"onnx_census_{args.weights.stem}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
