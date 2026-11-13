# 短期记忆

## 2026-10-27

- 首轮联合闭环已运行；针对 `N=2` 四次 rasterization 和未使用 heads 的两项优化已经完成代码实现，尚未重新运行。
- renderer 沿原边界拆为 `rasterize_surfel`/`shade_surfel`，默认 Ref wrapper 保留；joint 在 densification 结束后复用同一次 rasterization。
- joint 只执行 MVInverse A/M/R heads；MVInverse 默认五输出路径保持不变。
- 默认工作流是先完成 Ref iteration 20000，再固定 Gaussian topology 启动 joint；开发副本 `densify_until_iter=20000`。
- 当前唯一实施依据是 `docs/IMPLEMENTATION_PLAN.md` 指向的 2026-10-27 日期化方案；历史讨论、旧方案和性能报告只作证据。
- 下一步由用户从 Ref 20000 saved iteration 运行，比较速度、显存和运行时行为；不由 AI 启动训练或实验。

## 使用规则

- 每次追加都带日期。
- 只保留当前阶段真正会影响下一步工作的内容。
- 完成一个阶段后，把稳定结论整理到 `long-term.md`，把过期内容移入 `archive/`。
- 不把命令输出、完整日志或大段历史讨论复制进这里。
