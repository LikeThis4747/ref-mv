# 当前状态

更新时间：2026-10-20

## 阶段

阶段 1 已结束：首轮连接接口、训练边界、两种启动方式、保存语义和验收顺序已经冻结，可以进入实现。初版沿用 Ref 原训练流程，不额外冻结 Ref，也不关闭其 densification/pruning；新增 feature 只接入已有 topology 更新。多视角采用 anchor 加预计算 top-K 相机邻居。

两个上游开发副本已经复制到 `third_party/`，`ref_mv/`、`configs/`、`scripts/` 和 `tests/` 已建立。`ref-mv` 与开发副本 MVInverse 已重新 editable install；从项目根目录运行环境检查时，Ref renderer、MVInverse、rasterizer、raytracing 和 cubemapencoder 均确认从开发副本加载，CUDA 设备为 RTX A6000。当前没有联合模型代码，也没有启动训练。

## 当前方案

首轮使用 `C=1` 的零初始化显式 Gaussian feature，经 Ref 原 rasterizer 得到 alpha-premultiplied 二维 map，再用 `14×14` convolution 转为 1024 维 patch tokens，在 DINO 最终 `hidden_patches` 后直接相加。MVInverse 保持冻结，Ref 按原流程继续训练和 densify/prune；MV A/M/R 替换 surfel 的 `albedo/refl_strength/roughness`，并保留原 surfel shading 公式。

## 下一步

1. 按 `docs/IMPLEMENTATION_PLAN.md` 的顺序实现相机、Gaussian topology、renderer 和 MV residual 的最小接口及回归测试。
2. 实现联合模型、采样、完整 saved iteration 加载、输入/输出目录分离和 joint checkpoint。
3. 先通过 `C=1,N=1` 两步梯度与短程 overfit，再验证 `C=1,N=2`；通过前不启动长训练。

## 当前禁止事项

- 不修改 `/home/disk9/wsq/ENVS/mvinverse` 或 `/home/disk9/wsq/ENVS/ref-gaussian` 原始基线；修改只进入 `third_party/` 开发副本。
- 不把旧 FIPT/V6 方案直接当作 Ref+MV 方案。
- 不在相机、renderer 回归和梯度 smoke test 通过前启动长训练、densification 或大规模数据迁移。
- 不在未核对基线已有实现、且未明确标注为新增候选的情况下引入新的对齐、融合或训练方法。
