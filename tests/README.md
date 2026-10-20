# 自动化验证

本目录只存放由 pytest 执行的自动化验证代码，不放训练入口、实验脚本、数据处理或结果分析。

首轮仅验证四类高风险不变量：相机对齐、上游默认接口回归、Gaussian feature 与 topology 对齐、端到端梯度链。短程 overfit、`N=2` smoke、显存和视觉质量检查属于实验验证，入口放在 `scripts/experiments/`，不放在本目录。
