# 短期记忆

## 2026-10-27

- 首轮联合闭环已运行，主要瓶颈是 `N=2` 的四次 Gaussian rasterization 和 MVInverse 未使用的 normal/shading head。
- 当前已确认的下一步只有两项：沿原边界拆出 `rasterize_surfel`/`shade_surfel` 并复用 rasterization；joint 只执行 A/M/R head。
- 默认工作流是先完成 Ref iteration 20000，再固定 Gaussian topology 启动 joint；开发副本 `densify_until_iter=20000`。
- 当前唯一实施依据是 `docs/IMPLEMENTATION_PLAN.md` 指向的 2026-10-27 日期化方案；历史讨论、旧方案和性能报告只作证据。
- 不由 AI 启动训练、自动化测试或实验。

## 使用规则

- 每次追加都带日期。
- 只保留当前阶段真正会影响下一步工作的内容。
- 完成一个阶段后，把稳定结论整理到 `long-term.md`，把过期内容移入 `archive/`。
- 不把命令输出、完整日志或大段历史讨论复制进这里。
