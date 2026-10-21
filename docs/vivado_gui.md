# Vivado Block Design

Vivado 2021.2 工程由仓库中的 BD、XCI、顶层 RTL 和约束生成：

```powershell
./scripts/create_vivado_gui.ps1 -VivadoBat '<vivado.bat>'
```

生成后打开 `fpga/vivado/gui_project_alias/yolo26n_gui_alias.xpr`，在 **Design Sources → system_wrapper → system.bd** 查看系统连接。PS7 通过 AXI-Lite 配置 HLS 推理 IP；IP 的两个 AXI 主口经互连访问 PS 的 HP 通道。AXI VDMA、Video Timing Controller 和 AXI4-Stream to Video Out 组成 LCD 显示链路。全局连接图见 [`figures/fig04_vivado_bd_full.png`](figures/fig04_vivado_bd_full.png)。

GUI 工程使用的 `yolo26n_hls_i8` 是交付 IP 的类型别名，底层 RTL 与原 IP 相同。生成 GUI 工程不会重新综合位流；若修改 BD、时钟或 IP，需重新实现并验证板端结果。
