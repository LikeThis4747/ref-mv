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

首轮联合训练已经实际运行；针对重复 rasterization 和未使用 MVInverse heads 的性能优化已完成代码实现，等待用户重新运行确认实际速度与显存。

## 目录

- `ref_mv/`：本项目自己的联合逻辑。
- `third_party/ref_gaussian/`：Ref-Gaussian 开发副本。
- `third_party/mvinverse/`：MVInverse 开发副本。
- `configs/`：正式入口所需运行参数说明。
- `scripts/`：已有辅助入口；本轮不新增测试或实验脚本。
- `tests/`：保留目录；本轮未新增自动化测试。
- `docs/`：日期化方案、状态、稳定事实、决定、讨论、记忆和历史归档。
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

## 训练入口

正式入口为根目录 `train.py`。学习率、视角数和邻居数量不设研究默认值，运行前
按 `configs/README.md` 显式提供；从已有 Ref saved iteration 启动时，输入基线目录
必须与 joint 输出目录分开。

用 `--gs_feature_dim 8` 选择每个 Gaussian 的条件特征维度。支持 `1–16` 的整数，
默认 `1`。续训和通道预算见 [运行配置](configs/README.md)。

## 开始阅读

0. `docs/WINDOWS_MIGRATION.md`：Windows 迁移、完整 clone、环境重建和 AI 交接说明。
1. `AGENTS.md`：AI 工作边界。
2. `docs/README.md`：文档导航、日期命名和归档规则。
3. `docs/STATUS.md`：当前状态和下一步。
4. `docs/IMPLEMENTATION_PLAN.md`：当前方案与历史方案索引。
5. `docs/REFERENCE.md`：与当前任务相关的稳定事实。
6. `docs/DECISIONS.md`：已确认决定和修订关系。
7. `docs/discussions/`：保留的讨论过程和未决内容。

日常工作不默认读取 `docs/archive/` 或全部记忆文件；需要回溯设计变化时再按索引读取对应日期版本。
