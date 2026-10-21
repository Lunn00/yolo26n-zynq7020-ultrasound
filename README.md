# 颈动脉斑块超声图像识别的 YOLO26n FPGA 系统

本工程在 Zynq-7020（`xc7z020clg400-2`）上实现颈动脉斑块超声图像的 YOLO26n 检测。输入尺寸为 224×224，PL 端采用 INT8 层表驱动的 HLS 内核完成 111 层计算，PS 端完成 SD 读图、可选滤波、缓存维护、检测头解码、NMS 和 LCD 触摸交互。PL 目标时钟为 150 MHz。

YOLO26n 中的特殊算子和距离检测头需要适配可综合的数据通路；Zynq-7020 的 220 个 DSP 与 140 个 BRAM tile 则要求同时控制片上缓存、DDR 访问和布线开销。工程采用结构拟合、统一层表与分层量化处理网络计算，在 PS 端完成检测头解码。针对超声图像，在固定数据划分上训练 224 输入模型，并对 RAW、GAUSS 和 RGBG5 预处理进行配对验证。

## 交付内容

| 目录 | 内容 |
| --- | --- |
| `training/`、`model_pipeline/` | 训练配置、结构拟合、模型验证、ONNX 与 INT8 导出；含交付浮点模型 |
| `fpga/hls/` | 手写 HLS 推理内核、层表、量化参数、模型 blob、测试台与 Tcl |
| `fpga/vivado/` | Block Design、XCI、顶层 RTL、约束、构建 Tcl、资源及时序报告 |
| `software/ps/` | PS 裸机应用、滤波、检测后处理、LCD 与触摸源码 |
| `prebuilt/`、`sd_card/` | IP、bit、XSA、去调试信息的 ELF，以及交付 BOOT 与运行模型参数 |
| `verification/` | C 仿真、数据审计、实现与板端观察记录 |
| `docs/` | 复建说明、可编辑 Visio 图及工程图像 |
| `skill/` | 可加载技能、通用工作流、方法、验证证据与校验脚本 |
| `scripts/` | HLS、Vivado、Vitis、BOOT 与 SD 卡构建脚本 |

数据集和由医学图像派生的验证向量未公开；复建训练和 C 仿真时需提供有权使用的图像、标签和黄金参考。交付 IP 内部的 `yolo8n_hls_i8` 是生成时的模块标识，当前模型与层表为 YOLO26n；更改该标识会影响 BD 引用。

## 复建入口

硬件与软件构件使用 Vivado、Vitis、Vitis HLS **2021.2** 构建；迁移工具版本后需重新核对时序、数值和板端行为。模型训练与导出使用 Python 3.10，依赖版本见 [`requirements.txt`](requirements.txt)。输入、命令、输出和验证顺序见 [`docs/reproduction.md`](docs/reproduction.md)。

```powershell
python scripts/check_package.py
./scripts/build_hls.ps1 -VitisHlsBat '<vitis_hls.bat>' -SkipCsim
./scripts/build_vivado.ps1 -VivadoBat '<vivado.bat>'
./scripts/create_ps_platform.ps1 -XsctBat '<xsct.bat>'
./scripts/build_ps.ps1 -BspDir './build/ps_platform/yolo26n_platform/export/yolo26n_platform/sw/yolo26n_platform/standalone_ps7_cortexa9_0' -Gcc '<arm-none-eabi-gcc.exe>'
./scripts/package_boot.ps1 -BootgenBat '<bootgen.bat>'
```

HLS C 仿真需在 `fpga/hls/data/` 放入获授权的 `verify_input_i8.bin` 与 `verify_pred_i8.bin`，然后去掉 `-SkipCsim`。`scripts/create_vivado_gui.ps1` 可从仓库中的 BD/XCI 生成可编辑 GUI 工程。SD 卡部署和模型导出步骤见复建文档。

## 已记录的结果

- 同一 880/220 划分上，224 输入的结构拟合模型验证集 mAP50 为 0.9946，mAP50–95 为 0.8461；训练与验证配置见 `training/` 和 `model_pipeline/`。
- 最终 Vivado 实现占用 DSP 182/220、BRAM tile 108/140、LUT 39,278/53,200；WNS +0.012 ns、WHS +0.046 ns。
- 同一输入连续三次板端运行均显示 PL 201 ms、完整推理调用 203 ms；串口照片与逐项转录见 [`verification/serial_run_128_raw_three_times.md`](verification/serial_run_128_raw_three_times.md)。SD 读图与 PS 滤波另计。

复现时应使模型、层表、量化参数、IP、bit、ELF 与 BOOT 属于同一构建；步骤见 [`docs/reproduction.md`](docs/reproduction.md)。

## 许可与数据

本仓库采用 [AGPL-3.0-only](LICENSE)，覆盖贡献者有权许可的工程代码和文档；YOLO26 训练产物遵循 [Ultralytics 的 AGPL-3.0 说明](https://www.ultralytics.com/license)。AMD 工具生成物及第三方组件仍受其各自条款约束，详见 [`third_party_notices.md`](third_party_notices.md)。公开包不含训练图像和可还原的医学验证输入。此作品展示工程研究结果，不作临床诊断用途声明。
