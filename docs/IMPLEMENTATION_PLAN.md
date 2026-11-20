# 实施方案索引

> **工作原则：**一定要尽量与基线保持一致，能复用就复用；实现时结构保持简单，不过度设计。有任何不确定的问题或假设，必须及时汇报并讨论确认，不能基于未确认假设继续实现。

本文件是兼容入口和索引，不再承载会被覆盖的方案正文。

## 当前方案

- 2026-11-20：[参考 PGSR 的联合训练视角选择](plans/2026-11-20-pgsr-view-selection.md)（待实施）

- 2026-11-14：[材质 GT loss 实施方案](plans/2026-11-14-material-gt-loss.md)（增量实验功能）
- 2026-10-27：[Renderer 与 MVInverse 性能优化方案](plans/2026-10-27-renderer-mvinverse-performance-optimization.md)
- 状态：已实现，尚未由用户重新运行；实施记录见 [`reports/2026-10-27-performance-optimization-implementation.md`](reports/2026-10-27-performance-optimization-implementation.md)。

## 历史方案

- 2026-10-23：[首轮联合闭环修订方案](archive/plans/2026-10-23-revised-first-round-implementation-plan.md)
- 2026-10-20：[首轮冻结实施方案](archive/plans/2026-10-20-initial-frozen-implementation-plan.md)

历史方案用于回溯当时的设计和约束，不自动覆盖当前方案。若当前方案发生实质变化，创建新的 `YYYY-MM-DD-topic.md`，更新本索引，并把被替代方案保留到 `docs/archive/plans/`；不得直接用新正文覆盖旧方案。
