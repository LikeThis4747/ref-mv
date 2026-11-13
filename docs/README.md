# 文档导航与留痕规范

## 当前入口

- `STATUS.md`：当前阶段、下一步和边界的短摘要。
- `IMPLEMENTATION_PLAN.md`：当前方案与历史方案的索引。
- `DECISIONS.md`：按日期追加的已确认决定。
- `REFERENCE.md`：已核对的稳定事实。
- `plans/`：当前有效的日期化方案正文。
- `GROUP_MEETING_PROPOSAL.md`：日期化组会方案的兼容索引。
- `discussions/`：日期化的讨论、分析、候选方案和未决问题。
- `reports/`：日期化的实现或实验结果记录；只陈述实际完成和实际验证的范围。
- `memory/`：当前短期记忆、长期记忆及其归档。
- `archive/`：已经被替代但必须保留的原始文档快照。

## 文档生命周期

1. 讨论先写入 `discussions/YYYY-MM-DD-topic.md`，不把未确认内容写成决定。
2. 方案确认后创建 `plans/YYYY-MM-DD-topic.md`，并由 `IMPLEMENTATION_PLAN.md` 指向最新方案。
3. 新方案替代旧方案时，把旧方案原文移入 `archive/plans/`；不覆盖、不删除旧版。
4. 更新 `STATUS.md`、短期记忆或其他可变入口前，若旧内容具有回溯价值，先保存日期化快照。
5. 决定发生变化时，在 `DECISIONS.md` 追加修订或撤回记录，引用被替代决定；不静默改写历史。

## 命名与状态

- 日期化文件统一使用 `YYYY-MM-DD-topic.md`。
- 同一天有多个版本时，在主题后增加清晰后缀，不使用 `final-v2-new` 一类名称。
- 方案正文必须写明日期和状态，例如“讨论中”“已确认”“实施中”“已替代”。
- 归档目录中的文件默认只读；必要勘误应另加说明，不改写原始正文。

## 事实来源

- 当前工作以 `STATUS.md`、`IMPLEMENTATION_PLAN.md` 指向的最新方案和 `DECISIONS.md` 中未被撤回的决定为准。
- 历史归档只用于解释当时为什么这样设计，不能自动恢复为当前方案。
- 训练结论必须绑定日期、命令、运行参数、输入 checkpoint、输出目录和代码版本或 dirty 状态。
