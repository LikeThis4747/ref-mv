# 当前状态

更新时间：2026-10-27

## 阶段

首轮 Ref-MV 联合闭环已经实现并完成首次实际运行。首次运行确认代码能够进入
joint 分支，同时暴露出明确的性能问题：`N=2` 时每个视角执行两次 Gaussian
rasterization，MVInverse 还计算了 joint 未使用的 normal/shading head；观察到约
35.4 GiB 显存和约 2.28 秒/joint step。

默认工作流已经调整为：先按原 Ref 流程训练到 iteration 20000，再从该完整 saved
iteration 启动 joint。开发副本的 `densify_until_iter` 默认值已改为 20000，joint
阶段默认固定 Gaussian topology；原始上游目录未修改。

## 当前有效方案

- `C=1` Gaussian feature，经 Ref rasterization 和 `14x14` patch convolution 注入冻结 MVInverse。
- `N=2`，随机 anchor 加相机中心距离 top-K 邻居；所有视角使用 Ref 原 RGB/regularization loss。
- MV A/M/R 对应 Ref 的 `albedo/refl_strength/roughness`，Ref surfel shading 公式不变。
- Ref PLY 与 joint 状态分开保存；输入 Ref 目录与 joint 输出目录分离。
- 已确认但尚未编码的优化：拆出 `rasterize_surfel`/`shade_surfel` 以复用 rasterization；joint 只执行 MVInverse 的 A/M/R heads。

方案索引见 `docs/IMPLEMENTATION_PLAN.md`，当前正文见
`docs/plans/2026-10-27-renderer-mvinverse-performance-optimization.md`；首次性能证据见
`docs/discussions/2026-10-23-joint-training-performance-analysis.md`。

## 下一步

1. 按 `docs/plans/2026-10-27-renderer-mvinverse-performance-optimization.md` 第 4、5 节实施两项最小性能优化。
2. 只交付生产代码修改和代码审查结果，不由 AI 启动训练、自动化测试或实验。
3. 用户用 Ref 20000 saved iteration 运行 joint，并比较实际单轮时间和峰值显存。
4. 只有上述优化仍不足时，才讨论 MV 独立分辨率或冻结 encoder feature cache。

## 当前边界

- 不修改 `/home/disk9/wsq/ENVS/mvinverse` 或 `/home/disk9/wsq/ENVS/ref-gaussian` 原始基线。
- 不改变 Ref loss、PBR 公式、相机采样、checkpoint 格式或 CUDA rasterizer。
- 不提前实现独立 MV 分辨率、encoder cache、低频 joint step 或新的训练框架。
- 任何新增假设先汇报并确认，不基于未确认假设继续编码。
