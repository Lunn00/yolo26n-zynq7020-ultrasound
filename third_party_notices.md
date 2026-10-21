# 许可范围与第三方组件

仓库根目录的 `LICENSE` 为 AGPL-3.0-only。它适用于贡献者有权许可的 HLS、PS、训练与导出代码、脚本、Skill 和文字说明。任何贡献者应仅提交自己有权再分发的内容。

| 内容 | 位置 | 许可或使用边界 |
| --- | --- | --- |
| Ultralytics YOLO26 架构、训练框架与由其训练得到的模型 | `training/`、`model_pipeline/*.pt`、`model_pipeline/*.onnx`、运行参数 blob | Ultralytics 说明其代码及训练模型默认适用 AGPL-3.0；如需其他许可，应向权利人取得授权。[官方许可说明](https://www.ultralytics.com/license) |
| AMD/Xilinx IP 配置、生成的工程文件与预构建构件 | `fpga/vivado/`、`prebuilt/`、`sd_card/boot.bin` | 保留 AMD/Xilinx 原有权利及工具/IP 条款；本仓库的 AGPL 文件不对第三方组件重新授权。构建者需自行取得相应 2021.2 工具及 IP 使用权限。[AMD IP 文档](https://docs.amd.com/r/2021.2-English/ug896-vivado-ip/Introduction) |
| 医学图像、标签与图像派生的黄金向量 | 不在公开包内 | 按数据来源、去标识及授权条件自行取得。仓库中的板端照片仅用于展示原型运行状态。 |
