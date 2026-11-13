# 上游开发副本

以下目录是可编辑的开发副本。`/home/disk9/wsq/ENVS/` 下的原始检出目录保留为只读基线。

| 本地目录 | 原始检出目录 | 基准提交 |
|---|---|---|
| `third_party/ref_gaussian/` | `/home/disk9/wsq/ENVS/ref-gaussian` | `af7691aa1097cc6bf89dab6de0fcd358e8830bd8` |
| `third_party/mvinverse/` | `/home/disk9/wsq/ENVS/mvinverse` | `6172ff9a437444df028ed67523badfa523173f21` |

复制 `ref_gaussian` 时保留了已有的本地 CUDA 编译产物，以及原先对
`submodules/raytracing/setup.py` 所做的环境适配。这些属于环境状态，不是
Ref-MV 的模型改动。

后续 Ref-MV 改动必须尽可能小，并在修改处附近明确标注：

```python
# [UPSTREAM Ref-Gaussian or MVInverse] source: <commit>/<file>
# [REF-MV MODIFICATION] reason: ...
```

## 本地兼容性修改

- `ref_gaussian/utils/refl_utils.py`：改为相对开发副本定位已有 BSDF LUT，
  不再依赖进程的当前工作目录；LUT 内容与渲染计算均未改变。

## Ref-MV 接口修改

- `ref_gaussian/utils/camera_utils.py`：在原 `-r` resize 后按 joint patch size
  从右/下裁切，并同步相机尺寸与投影参数。
- `ref_gaussian/scene/__init__.py`：将 Ref saved iteration 输入目录与 joint 输出目录分开。
- `ref_gaussian/scene/gaussian_model.py`：加入独立 `gs_feature` 参数及原 topology
  clone/split/prune 的同步更新；不改变 PLY 和 `capture()/restore()` 格式。
- `ref_gaussian/gaussian_renderer/__init__.py`：沿原计算边界拆出
  `rasterize_surfel`/`shade_surfel`，复用同一次 rasterization 的 GS feature 与
  A/M/R shading；`render_surfel` 保留为原入口 wrapper。
- `ref_gaussian/train.py`：仅在原循环选择 `render_surfel` 时接入 joint step；其余
  stage、保存和 mesh 调度保持原位置。
- `mvinverse/mvinverse/models/mvinverse.py`：在 DINO patch tokens 与原 decoder
  之间增加可选 residual，并允许调用方选择输出 heads；默认不传时仍执行并返回原五项。
