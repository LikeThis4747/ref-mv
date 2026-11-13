# Renderer 与 MVInverse 性能优化实施报告

日期：2026-10-27

状态：代码实现完成；尚未运行训练或性能实验

## 实施范围

本次严格按 `docs/plans/2026-10-27-renderer-mvinverse-performance-optimization.md` 实现两项优化：

1. 同一视角的 GS feature 与最终材质 shading 复用一次 Gaussian rasterization。
2. joint 路径只执行 MVInverse 的 albedo、metallic 和 roughness heads。

未加入独立 MV 分辨率、encoder cache、低频 joint step、activation checkpoint、全模型 FP16、新 CLI 参数或测试框架。

## 代码修改

- `third_party/ref_gaussian/gaussian_renderer/__init__.py`
  - 从原 `render_surfel()` 提取 `rasterize_surfel()` 和 `shade_surfel()`。
  - `render_surfel()` 保留为依次调用两者的兼容 wrapper，原 PBR、normal/depth regularization、indirect、背景合成和结果字典不变。
  - `rasterize_surfel()` 可同时返回原 rasterizer 中间量和 `gs_feature`。
- `ref_mv/joint.py`
  - 每个视角先保存 rasterization 结果，再生成 GS tokens 和 A/M/R。
  - densification 结束后直接把同一结果传给 `shade_surfel()`。
  - densification 结束前仍调用第二次完整 `render_surfel()`，保留原统计所使用的完整 render 语义。
- `third_party/mvinverse/mvinverse/models/mvinverse.py`
  - `forward()` 增加可选 `output_names`。
  - 未传时仍执行并返回原五项；joint 显式请求 A/M/R，跳过 normal/shading heads。
- `third_party/ref_gaussian/train.py`
  - 只增加内部 `reuse_rasterization=iteration >= opt.densify_until_iter` 传递。

## 保持不变

- 原始 `/home/disk9/wsq/ENVS/ref-gaussian` 和 `/home/disk9/wsq/ENVS/mvinverse` 未修改。
- Ref loss、PBR 公式、相机与邻居采样、CUDA rasterizer、PLY、`capture()/restore()` 和 `joint.pt` 格式未修改。
- 默认 Ref 调用仍从 `render_surfel()` 进入。
- MVInverse 的 DINO、ResNeXt 和共享 decoder 仍完整执行。

## 已完成检查

- 四个修改文件通过 Python 静态语法编译检查。
- 相关 diff 通过 whitespace/error 检查。
- 全仓静态搜索确认已移除旧 `return_gs_feature` 调用，没有遗留旧接口引用。

## 尚未验证

按项目边界，本次没有加载模型、启动训练、运行自动化测试或执行性能实验。以下内容必须由用户实际运行确认：

- Ref 20000 saved iteration 启动 joint 是否正常。
- 单步时间和峰值显存的实际下降幅度。
- loss、渲染结果和梯度行为是否与修改前一致。
- indirect shading 与实际 mesh/checkpoint 组合下的运行行为。
