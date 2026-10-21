# 交付检查点的结构构造

此目录保留从 224 输入 YOLO26n 训练检查点到 `model_cbam_source.pt` 的三步源码。`surgery_pl_native_head.py` 用对角权重把检测头 depthwise 卷积映射到稠密卷积，并做同输入数值核对；`surgery_cbam_attention.py` 构造注意力候选；`fit_cbam_to_attention.py` 从训练图像采集原注意力输入与输出，拟合两个 CBAM 模块。之后由上级目录的 `replace_cbam.py` 与 `fit_conv1x1.py` 得到最终 1×1 残差结构。

默认输入是 `runs_yolo26/yolo26n_224/weights/best.pt` 和 `datasets/carotid/images/train/`，输出写入被 Git 忽略的 `build/model_source/`。各脚本的 `--help` 可指定其他路径。已有交付检查点位于上级目录，可直接使用；重新训练和拟合的检查点必须独立验证，不能与仓库已交付的 HLS 参数或位流混用。完整流程见 [`../../docs/reproduction.md`](../../docs/reproduction.md)。
