# PS 裸机应用

`src/` 是交付版的 Cortex-A9 standalone 应用源代码。`main.c` 负责图像、UI 状态机、IP 控制与计时；`filter_sw.*` 处理三档输入，`yolo_detect.*` 完成检测头解码和 NMS，`touch_hw.*`、`ui_widgets.*` 实现触摸与 LCD 交互。

复建需要 Vitis 2021.2 的 XSCT 和 ARM 编译器。先从仓库根目录运行 `scripts/create_ps_platform.ps1 -XsctBat <xsct.bat>`；脚本从 `prebuilt/ps_platform_0928.xsa` 生成包含 xilffs、VDMA 等 Xilinx 驱动的 BSP。再运行 `scripts/build_ps.ps1 -BspDir ./build/ps_platform/yolo26n_platform/export/yolo26n_platform/sw/yolo26n_platform/standalone_ps7_cortexa9_0 -Gcc <arm-none-eabi-gcc.exe>`。该脚本输出 `build/ps/yolo26_lcd_ui.elf`。使用最终 PL 位流打包时，请将新 ELF 传给 `scripts/package_boot.ps1 -ApplicationElf <ELF路径>` 并重新上板验证。

默认构建关闭 `YOLO_BOOT_SELFTEST`。公开的 `yolo_selftest_vectors.c` 不含医学图像派生向量；若启用板上自检，须先加入自己有权使用的输入及 Q10 黄金预测，再验证该模式。
