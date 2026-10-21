# 交付模型与导出链

| 文件 | 作用 |
| --- | --- |
| `model_cbam_source.pt` | 224 输入的已训练模型，作为注意力结构替换的输入 |
| `source_construction/` | 从已训练 YOLO26n 检查点构造并拟合上述 CBAM 检查点的三步源码；新输出写入 `build/model_source/` |
| `replace_cbam.py` | 将两个 CBAM 模块改为可在现有卷积数据通路上执行的 1×1 残差候选，生成临时 `model_conv1x1.pt` |
| `fit_conv1x1.py` | 在训练图像上拟合残差权重，生成最终 `model_conv1x1_fit.pt` |
| `validate_model.py`、`validation_model_conv1x1_fit.json` | 同一验证划分上的结构与检测指标核对 |
| `export_onnx.py`、`model_conv1x1_fit.onnx` | 导出并检查交付浮点计算图 |
| `exporter/export_yolo26_hls_int8.py` | 标定、INT8 权重量化、整数参数及 HLS 层表导出 |

`exporter/export_yolo26_hls.py` 提供计算图调度与内存分配逻辑，供最终 INT8 导出器调用。其独立 FP32 导出入口和部分内部标识沿用原型阶段命名，不定义交付模型身份。交付模型以 `model_conv1x1_fit.pt`、对应 ONNX、`fpga/hls/src/yolo_layers_i8.h` 和参数 blob 的一致性为准。

完整命令、数据准备和版本对应关系见 [`../docs/reproduction.md`](../docs/reproduction.md)。仓库提供交付检查点；从初始权重重训及拟合会生成另一套需要重新验证的构件。原始训练图像不在仓库中。
