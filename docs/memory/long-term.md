# 长期记忆

## 2026-10-14

- `ref-mv` 是独立研究工作区；`mvinverse` 和 `ref-gaussian` 是只读基线。
- 项目联合代码放 `ref_mv/`，必要的上游接口修改放 `third_party/` 开发副本；临时脚本和中间结果放 `ai-workspace/`。
- 根目录保持极简；当前状态、决定、稳定参考、讨论和记忆分开维护。
- 讨论、推断和实现授权必须与代码事实和已确认决定分开。
- 任何训练结论都必须绑定具体配置、代码版本、数据、checkpoint 和验证范围。

## 2026-10-27

- 默认实验工作流先完成 Ref iteration 20000，再从完整 saved iteration 启动 joint；joint 默认固定 Gaussian topology。
- 性能优化必须先删除已确认的重复计算，不先改变分辨率、视角数、loss 或训练频率。
