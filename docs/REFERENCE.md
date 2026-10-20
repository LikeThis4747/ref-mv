# 稳定参考

本文件只记录已经从代码或论文核对过、对当前工作有用的事实。候选解释和未完成核对放在 `docs/discussions/`。

## MVInverse

- 当前核对基准 commit：`6172ff9a437444df028ed67523badfa523173f21`。

- MVInverse 接收多视角 RGB，输出每个视角的二维 A/M/R 图。
- DINOv2 ViT-L/14 的基线 patch embedding 是 `Conv2d(3, 1024, kernel_size=14, stride=14)`；MVInverse 随后读取完整 DINO encoder 输出的 `x_norm_patchtokens`，再传给自身 `decode()`。
- 当前代码没有一个现成的外部 `image token` 条件接口；若要加入 Gaussian feature 条件，必须明确实际 token 流程、注入层、token 网格和维度。
- 已检查的 kitchen 输入约为 `308×630`；patch size 14 时每视角约为 `22×45=990` 个 patch token，每个 token 1024 维。
- 冻结 MVInverse 参数不等于可以对完整前向使用 `torch.no_grad()`。如果 RGB loss 要更新 feature 或条件投影层，梯度必须保留经过 `A/M/R` 的路径。

## Ref-Gaussian

- 当前核对基准 commit：`af7691aa1097cc6bf89dab6de0fcd358e8830bd8`。

- 当前 rasterizer 支持对 Gaussian 的额外 feature 做可微 splatting。
- 已检查的 `render_surfel` 路径会 rasterize SH `base_color` 和额外 feature，再结合 `refl_strength` 与 specular 结果生成 RGB。
- 当前活跃路径使用的是 `refl_strength`；不能未经核对就把 MVInverse metallic 直接映射到保存但未使用的 `_metalness`。
- 当前 rasterizer 的 feature 通道有上限，现有 surfel 路径已经使用部分 feature 通道；新增 feature 前必须检查完整调用链和通道预算。
- Ref-Gaussian 的论文 deferred PBR 描述不能直接等同于当前 checkout 的实际 renderer；论文和代码要分开核对。
- `GaussianModel.capture()` 不保存环境贴图或 ray-tracing mesh；完整 surfel/indirect 加载应沿用 `eval.py` 的 saved-iteration 路径，并在 indirect 模式加载同 iteration 的 mesh。

## 数据和实验边界

- 当前 Ref-Gaussian checkout 没有现成 kitchen checkpoint/PLY。
- Ref-Gaussian loader 与现有 kitchen 的 `train/transforms.json`、`val/transforms.json` 和 EXR 数据接口并不天然一致。
- FIPT 的连续 `position → hash-grid/MLP → material` 场与 per-Gaussian 显式 feature 不是同一种三维表示；不能把两者的研究含义直接等同。

## 实施前必须重新核对

- 以上路径和行号属于前期源码审计结果；正式实现前必须在当前 checkout 再检查具体代码。
- 必须明确 `render_volume`、`render_surfel`、A/M/R shader 路径、emission、可见性和 loss 颜色空间。
