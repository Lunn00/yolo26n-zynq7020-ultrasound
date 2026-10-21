# 技能提炼的工程证据

`skill/` 描述可迁移的方法，本页只记录它们在本作品中由什么观察触发、可用仓库哪些文件核对。方法的适用范围、操作步骤和失效条件在各技能文件中；本工程数值不作为其他项目的默认阈值。

| 观察与决策 | 仓库核对入口 | 提炼的方法 |
| --- | --- | --- |
| 训练数据必须先确认图像、标签及分组关系，才能讨论缩小输入的效果 | `verification/dataset_audit_summary.json`、`training/training_summary.json`、`training/dataset.example.yaml` | [数据集契约](dataset_contract.md)、[领域训练](domain_training.md) |
| 特殊算子改变通道依赖和访存模式；最终采用经同划分验证的 1×1 残差拟合与统一层表 | `model_pipeline/replace_cbam.py`、`fit_conv1x1.py`、`validation_model_conv1x1_fit.json`、`fpga/hls/src/yolo_layers_i8.h` | [算子映射](operator_mapping.md)、[结构替换](model_surgery.md) |
| 浮点、INT8 与检测头缓冲区需要对齐张量语义；复用工作区要在层完成时取快照 | `model_pipeline/exporter/`、`fpga/hls/src/`、`verification/csim_console.log` | [量化契约](quantization_contract.md)、[输出语义](tensor_contract.md)、[激活快照](activation_snapshot.md) |
| 片上复用、DDR 访问、循环 II 和布线资源须联合判断，局部排程指标不能代替整机结果 | `fpga/hls/src/yolo_hls_i8_26.cpp`、`fpga/vivado/reports/`、`verification/serial_run_128_raw_three_times.md` | [PL 数据流](pl_dataflow.md)、[优化判读](optimization.md)、[整机剖析](end_to_end_profile.md) |
| PS 与 PL 的缓存、地址和启动完成协议共同决定板端结果；应用界面要避免显示陈旧检测框 | `software/ps/src/`、`docs/diagrams/02_ps_software_flow.vsdx` | [PS—PL 协议](ps_pl_protocol.md)、[板级失配](board_mismatch.md)、[存储边界](memory_boundary.md) |
| 最终模型、参数、IP、bit、ELF 和 BOOT 必须能被一组哈希关联 | `verification/release_identity.json`、`scripts/check_package.py` | [版本身份](release_identity.md) |

17 项方法的逐项效果和迁移边界见 [有效性矩阵](validation_matrix.md)。该矩阵区分交付验证和早期候选试验，不把不同网络或测量窗口的数字合并。
