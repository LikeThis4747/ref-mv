# Ref-Gaussian × MVInverse

这是 Ref-Gaussian 与 MVInverse 结合研究的独立工作区。

当前目标是研究一条可验证的端到端路径：

```text
Gaussian feature
    → 相机 rasterization / feature map
    → MVInverse token 条件
    → A / M / R
    → Ref-Gaussian 渲染
    → RGB loss
```

首轮接口与实施方案已经确认，当前仅完成仓库脚手架和环境准备，尚未实现联合模型。

## 目录

- `ref_mv/`：本项目自己的联合逻辑。
- `third_party/ref_gaussian/`：Ref-Gaussian 开发副本。
- `third_party/mvinverse/`：MVInverse 开发副本。
- `configs/`：场景与实验配置。
- `scripts/`：环境、smoke test 和训练命令。
- `tests/`：导入、相机、renderer 和梯度回归测试。
- `docs/`：状态、稳定事实、决定、讨论、记忆和历史归档。
- `ai-workspace/`：AI 工作时使用的临时脚本和中间结果，不作为正式接口。

## 外部基线

- MVInverse：`/home/disk9/wsq/ENVS/mvinverse`
- Ref-Gaussian：`/home/disk9/wsq/ENVS/ref-gaussian`
- Conda 环境：`/home/disk9/wsq/conda/envs/mv_refgaussian`
- Hugging Face 缓存：`/home/disk9/wsq/.cache/huggingface`

上述原始上游目录始终只读；必要接口修改只进入 `third_party/` 开发副本。

## 环境检查

```bash
cd /home/disk9/wsq/ENVS/ref-mv
source scripts/env.sh
scripts/check_environment.sh
```

检查必须确认 Ref、MVInverse 和 CUDA 扩展均从本项目开发副本加载。

## 开始阅读

1. `AGENTS.md`：AI 工作边界。
2. `docs/STATUS.md`：当前状态和下一步。
3. `docs/REFERENCE.md`：与当前任务相关的稳定事实。
4. `docs/DECISIONS.md`：已确认决定。
5. `docs/IMPLEMENTATION_PLAN.md`：首轮代码实施计划。
6. `docs/discussions/`：仍在讨论的内容。

不要默认读取 `docs/archive/` 或全部记忆文件。
