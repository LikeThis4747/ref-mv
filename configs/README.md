# 运行配置

本轮不提供实验脚本。正式入口为根目录 `train.py`，首跑默认值为：

- `mv_checkpoint=maddog241/mvinverse`；
- `gs_feature_lr=0.001`；
- `gs_patch_lr=5e-5`，沿用 MVInverse 训练的 optimizer 学习率量级；
- `joint_envmap_lr` 不提供时沿用 Ref 的 `envmap_cubemap_lr=0.01`，joint 开始后可单独覆盖；
- `neighbor_k=8`；
- `num_views=2`。

数据路径没有通用默认值，必须提供 `-s/--source_path`。未提供 `-m/--model_path`
时继续复用 Ref 的时间戳输出目录。从已有 Ref saved iteration 启动时，同时提供
`--ref_model_path` 与 `--ref_iteration`；只有明确续接 joint 状态时才提供
`--joint_checkpoint`。
