# 复建与验证

## 1. 固定输入与环境

目标器件是 `xc7z020clg400-2`，PL 目标频率 150 MHz。交付构件使用 Vivado、Vitis、Vitis HLS 2021.2。模型流程使用 Python 3.10；已核对的包版本在仓库根目录 `requirements.txt`。在仓库根目录创建 Python 环境后安装依赖：

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

根据本机 CPU/GPU 选择适用的 PyTorch 安装方式。训练需另行取得允许使用的 YOLO26n 初始权重，并放在仓库根目录为 `yolo26n.pt`。将获授权的数据按 `training/dataset.example.yaml` 放置或修改其中的相对路径。交付实验使用 880 张训练图与 220 张验证图，图像和标签一一对应；图像与患者关联信息不在公开包内。数据审计脚本见 `skill/audit_dataset.py`。

## 2. 模型与参数

若仅复建 FPGA 与板端程序，可直接使用已提供的浮点模型、HLS 参数 blob 与 `sd_card/` 文件。交付模型的结构路线为：224 输入训练检查点 → 检测头 depthwise 的对角稠密化 → 注意力候选拟合 → 1×1 残差替换与拟合 → ONNX、INT8 和 HLS 层表。第一处替换保存原算子数值；后两处通过训练图像上的拟合及同划分检测指标验收。仓库提供各步代码与已经验收的 `model_cbam_source.pt`、`model_conv1x1_fit.pt`。

若从已提供的 CBAM 检查点继续，可执行：

```powershell
python model_pipeline/replace_cbam.py
python model_pipeline/fit_conv1x1.py --images 64
python model_pipeline/validate_model.py --weights model_pipeline/model_conv1x1_fit.pt
python model_pipeline/export_onnx.py --weights model_pipeline/model_conv1x1_fit.pt
python model_pipeline/exporter/export_yolo26_hls_int8.py --onnx model_pipeline/model_conv1x1_fit.onnx --images datasets/carotid/images/train --val-images datasets/carotid/images/val --out-dir build/model_export
```

若从初始权重和获授权数据重新训练，再按顺序运行 `training/train_yolo26n_224.py`、`model_pipeline/source_construction/surgery_pl_native_head.py`、`surgery_cbam_attention.py`、`fit_cbam_to_attention.py`。这些脚本将中间检查点写入被忽略的 `build/model_source/`。随后把 `replace_cbam.py --src build/model_source/model_cbam_source.pt --dst build/model_source/model_conv1x1.pt` 和 `fit_conv1x1.py --src build/model_source/model_cbam_source.pt --conv-src build/model_source/model_conv1x1.pt --train-images datasets/carotid/images/train --dst build/model_source/model_conv1x1_fit.pt` 串起来；使用新检查点验证、导出和重新构建。重训及随机拟合产生的是**新构件**，不能直接与仓库中的层表、IP 或 bitstream 混用。先核对层数、张量布局、量化参数和数值结果，再重跑 HLS、布线与板测。`training/training_summary.json`、`model_pipeline/validation_model_conv1x1_fit.json` 和 `verification/dataset_audit_summary.json` 保存交付流程的指标口径。

## 3. HLS 与 Vivado

在 PowerShell 中从仓库根目录执行。脚本参数接受相应工具启动程序的位置；可改用环境变量 `VITIS_HLS_BAT`、`VIVADO_BAT`、`XSCT_BAT`、`ARM_GCC`、`BOOTGEN_BAT`。不提供训练图像时，先跑综合与 IP 导出；C 仿真需加入一对获授权的 INT8 输入和 FP32 黄金预测：

```powershell
./scripts/build_hls.ps1 -VitisHlsBat '<vitis_hls.bat>' -SkipCsim
./scripts/build_vivado.ps1 -VivadoBat '<vivado.bat>'
./scripts/create_vivado_gui.ps1 -VivadoBat '<vivado.bat>' -NoOpen
```

HLS C 仿真输入应放在 `fpga/hls/data/verify_input_i8.bin` 和 `verify_pred_i8.bin`，随后去掉 `-SkipCsim`。测试台核对 111 层表、固定参数长度及活动候选位置的预测误差。可用 `scripts/build_vivado.ps1 -IpArchive '<new-ip.zip>'` 检验本次 HLS 导出的 IP；默认脚本使用交付 IP 包。Vivado 脚本输出位流、检查点和资源时序报告到 `fpga/vivado/output/`。GUI 工程由脚本生成在被 Git 忽略的 `fpga/vivado/gui_project_alias/`，源 BD、XCI 与约束已纳入仓库。

## 4. PS、BOOT 与上板

`prebuilt/ps_platform_0928.xsa` 提供 PS 外设配置以生成 standalone BSP；最终 PL 实现由本仓库的 bitstream 和 IP 定义。生成 BSP、编译 PS 应用并打包 BOOT：

```powershell
./scripts/create_ps_platform.ps1 -XsctBat '<xsct.bat>'
./scripts/build_ps.ps1 -BspDir './build/ps_platform/yolo26n_platform/export/yolo26n_platform/sw/yolo26n_platform/standalone_ps7_cortexa9_0' -Gcc '<arm-none-eabi-gcc.exe>'
./scripts/package_boot.ps1 -BootgenBat '<bootgen.bat>'
```

`package_boot.ps1` 默认用交付 bit/ELF，生成的 BOOT.BIN 与 `sd_card/boot.bin` 哈希相同。若编译了新的应用或位流，应通过 `-ApplicationElf`、`-Bitstream` 指定并重新上板验证。程序以 FAT32 SD 卡启动；将 `scripts/prepare_sd_card.ps1 -Destination '<sd-card-root>'` 输出的四个大写运行文件和获授权的十进制编号 BMP 放到卡根目录。PS 应用默认 RAW，可在触摸屏选择图片、运行检测和切换滤波模式。可编辑 Visio 图在 `docs/diagrams/`。

## 5. 验收口径

按“数据与模型 → INT8 张量 → HLS C 仿真 → Vivado 时序与资源 → 板端显示”逐级核对。`verification/` 保存公开的摘要、日志和报告；板端 PL 时间从 `ap_start` 到 `ap_done`，INFER 为完整单次模型调用，SD 读图和 PS 滤波单列。交付阶段观察为 PL 201 ms、INFER 203 ms。若更换输入、软件或硬件构件，保留图片编号、运行模式、串口记录、检测结果与对应构件哈希，避免跨版本比较。

运行 `python scripts/check_package.py` 核对工程结构、关键构件哈希及隐私扫描。Skill 的使用边界和已验证效果见 `skill/README.md` 与 `skill/validation_matrix.md`。
