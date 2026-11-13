# 当前状态

更新时间：2026-10-27

## 阶段

首轮 Ref-MV 联合闭环已经实现并完成首次实际运行。针对首次运行暴露的重复计算，
2026-10-27 性能优化方案已经完成代码实现，尚未由用户重新运行。

默认工作流已经调整为：先按原 Ref 流程训练到 iteration 20000，再从该完整 saved
iteration 启动 joint。开发副本的 `densify_until_iter` 默认值已改为 20000，joint
阶段默认固定 Gaussian topology；原始上游目录未修改。

## 当前有效方案

- `C=1` Gaussian feature，经 Ref rasterization 和 `14x14` patch convolution 注入冻结 MVInverse。
- `N=2`，随机 anchor 加相机中心距离 top-K 邻居；所有视角使用 Ref 原 RGB/regularization loss。
- MV A/M/R 对应 Ref 的 `albedo/refl_strength/roughness`，Ref surfel shading 公式不变。
- Ref PLY 与 joint 状态分开保存；输入 Ref 目录与 joint 输出目录分离。
- renderer 已按原边界拆为 `rasterize_surfel`、`shade_surfel` 和兼容 wrapper `render_surfel`。
- iteration 达到 `densify_until_iter` 后，joint 每个视角只执行一次 rasterization；更早的从头 joint 仍保留第二次完整 render 供原 densification 统计使用。
- joint 只请求 MVInverse 的 A/M/R heads；不传 `output_names` 的原 MVInverse 路径仍返回五项。

方案索引见 `docs/IMPLEMENTATION_PLAN.md`，当前正文见
`docs/plans/2026-10-27-renderer-mvinverse-performance-optimization.md`；首次性能证据见
`docs/discussions/2026-10-23-joint-training-performance-analysis.md`，实施记录见
`docs/reports/2026-10-27-performance-optimization-implementation.md`。

## 下一步

1. 用户用 Ref 20000 saved iteration 重新运行 joint，比较实际单轮时间和峰值显存。
2. 确认输出、loss 和梯度行为没有运行时回归。
3. 只有本轮优化仍不足时，才讨论 MV 独立分辨率或冻结 encoder feature cache。

## 当前边界

- 不修改 `/home/disk9/wsq/ENVS/mvinverse` 或 `/home/disk9/wsq/ENVS/ref-gaussian` 原始基线。
- 不改变 Ref loss、PBR 公式、相机采样、checkpoint 格式或 CUDA rasterizer。
- 不提前实现独立 MV 分辨率、encoder cache、低频 joint step 或新的训练框架。
- 任何新增假设先汇报并确认，不基于未确认假设继续编码。
