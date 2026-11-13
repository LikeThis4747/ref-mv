# Ref-MV 当前实施方案

> **工作原则：**一定要尽量与基线保持一致，能复用就复用；实现时结构保持简单，不过度设计。有任何不确定的问题或假设，必须及时汇报并讨论确认，不能基于未确认假设继续实现。在哪一步就只做好哪一步的事；当前职责是实现代码，不提前实现自动化测试、实验或训练辅助功能，也不代替用户启动训练。

状态：首轮联合闭环已实现并完成首次运行；本文件中的性能优化方案已确认，尚未编码。  
最近修订：2026-10-27

## 1. 当前目标和默认工作流

当前闭环保持不变：

```text
camera RGB -> frozen DINO -> hidden_patches ------------------┐
                                                               + -> frozen MV decoder -> A/M/R
Gaussian feature -> Ref rasterization -> patch Conv ----------┘
                                                                        |
                                                                        v
GT RGB <- Ref 原 loss <- Ref 原 surfel shading（A/M/R override）
```

固定映射：

```text
MV albedo    -> Ref albedo
MV metallic  -> Ref refl_strength
MV roughness -> Ref roughness
```

保留 Ref 原公式：

```text
final_image = (1 - refl_strength) * base_color + specular
```

默认运行流程是：

1. 使用开发副本按原 Ref 流程训练到 iteration 20000。
2. 从完整 Ref saved iteration 20000 加载 Gaussian、环境贴图和同轮 mesh。
3. Ref 输入目录和 joint 输出目录分开。
4. 从 iteration 20001 开始 joint，Gaussian topology 固定。

开发副本的 `densify_until_iter` 默认值为 20000。原判断是
`iteration < densify_until_iter`，所以从头训练 Ref 时仍执行 20000 之前的原
clone/split/prune；从 Ref 20000 加载后不再继续 densify。

## 2. 保持不变的已有实现

- 每个 Gaussian 使用 `C=1`、零初始化的显式 `gs_feature`。
- `Conv2d(1, 1024, kernel_size=14, stride=14, bias=False)` 将二维 feature map 转成 patch tokens。
- GS tokens 在 DINO 最终 `hidden_patches` 后直接相加。
- MVInverse 参数全部冻结并保持 `eval()`，但不使用 `torch.no_grad()`，以保留 residual 输入梯度。
- `N=2` 时随机选择 anchor，再从相机中心距离 top-K 邻居中选择另一个视角。
- 两个视角分别使用 Ref 原 loss，取平均后统一 backward。
- Ref PLY/capture 格式不变；`gs_feature` 和 patch convolution 单独保存在 `joint.pt`。
- 输入 Ref saved iteration 与 joint 输出目录分离。
- 图像先复用 Ref 的 `-r` resize，再从右侧和下侧裁到 14 的倍数。

## 3. 已确认的性能优化

首轮实测在 `N=2`、`980x742` 下约占用 35.4 GiB 显存，joint step 约
2.28 秒。已确认先做两项不改变当前 loss 和材质映射的优化：

1. 将每个视角重复两次的 Gaussian rasterization 合并为一次。
2. joint 调用 MVInverse 时只执行 A/M/R 输出 head，跳过未使用的 normal/shading head。

不在本轮同时加入独立 MV 分辨率、encoder cache、低频 joint step、activation
checkpoint 或全模型 FP16。

## 4. Renderer 拆分与 rasterization 复用

### 4.1 函数边界和命名

修改 `third_party/ref_gaussian/gaussian_renderer/__init__.py`，沿原
`render_surfel()` 的实际计算边界拆成：

```python
rasterize_surfel(...)
shade_surfel(...)
render_surfel(...)
```

- `rasterize_surfel`：负责原 Gaussian 投影和 rasterizer 调用。
- `shade_surfel`：负责原 normal/depth regularization、specular、indirect、背景和最终合成。
- `render_surfel`：保留原公开入口，依次调用前两个函数。

不新增 renderer 类、cache dataclass、callback 或第二套 PBR 实现。

### 4.2 `rasterize_surfel`

该函数直接移动原 `render_surfel()` 从开头到 `rasterizer(...)` 结束的代码：

```python
def rasterize_surfel(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    override_color=None,
    gaussian_feature=None,
):
    # 原 screenspace_points、raster settings、covariance、SH、indirect
    # 原 feature_parts 和 rasterizer(...) 调用

    rasterized = {
        "base_color": rendered_image,
        "rendered_features": rendered_features,
        "allmap": allmap,
        "radii": radii,
        "viewspace_points": means2D,
    }

    if gaussian_feature is not None:
        feature_dim = gaussian_feature.shape[1]
        rasterized["gs_feature"] = rendered_features[8:8 + feature_dim]

    return rasterized
```

继续保留 `8 + feature_dim <= 24`、Gaussian 数量和 feature shape 检查。

### 4.3 `shade_surfel`

该函数直接移动原 rasterizer 之后的代码，不改变量语义和公式：

```python
def shade_surfel(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    rasterized,
    srgb=False,
    opt=None,
    material_maps=None,
):
    base_color = rasterized["base_color"]
    rendered_features = rasterized["rendered_features"]

    refl_strength = rendered_features[0:1]
    roughness = rendered_features[1:2]
    albedo = rendered_features[2:5]
    indirect_light = rendered_features[5:8]

    # 原 compute_2dgs_normal_and_regularizations(...)
    # 原 normal_map / render_alpha / surf_depth

    if material_maps is not None:
        # 沿用已有 shape 检查
        albedo = material_maps["albedo"].permute(2, 0, 1).to(base_color)
        refl_strength = material_maps["metallic"].permute(2, 0, 1).to(base_color)
        roughness = material_maps["roughness"].permute(2, 0, 1).to(base_color)

    # 原 get_specular_color_surfel、indirect、sRGB 和背景合成
    # 返回键保持与原 render_surfel 一致
    return results
```

### 4.4 保留原 `render_surfel`

```python
def render_surfel(..., material_maps=None):
    rasterized = rasterize_surfel(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
    )
    return shade_surfel(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        rasterized,
        srgb=srgb,
        opt=opt,
        material_maps=material_maps,
    )
```

默认 Ref 路径仍调用 `render_surfel()`，调用位置、参数默认值和结果字典保持不变。

### 4.5 joint 前向

`ref_mv/joint.py` 改为先为每个视角保存 rasterization 结果：

```python
rasterized_views = []
feature_maps = []

for camera in cameras:
    rasterized = rasterize_surfel(
        camera,
        self.gaussians,
        pipe,
        background,
        gaussian_feature=self.gaussians.get_gs_feature,
    )
    rasterized_views.append(rasterized)
    feature_maps.append(rasterized["gs_feature"])

material = self.mvinverse(...)

render_pkgs = []
for camera, rasterized, material_maps in zip(...):
    render_pkgs.append(
        shade_surfel(
            camera,
            self.gaussians,
            pipe,
            background,
            rasterized,
            material_maps=material_maps,
            srgb=opt.srgb,
            opt=opt,
        )
    )
```

`N=2` 时由：

```text
2 x feature rasterization + MVInverse + 2 x material rasterization
```

变为：

```text
2 x rasterization + MVInverse + 2 x image-space shading
```

同一 rasterizer 输出同时接收最终 shading 路径和 `GS feature -> MVInverse` 路径的梯度；在 topology 固定的默认 joint 流程中，不再读取它用于 densification heuristic。

### 4.6 从头 joint 的兼容路径

从头 joint 在 iteration 20000 前仍可能 densify。为保持现有 densification 统计只读取第二次完整 render 的语义，训练循环传入：

```python
reuse_rasterization = iteration >= opt.densify_until_iter
```

joint 中：

```python
if reuse_rasterization:
    render_pkg = shade_surfel(..., rasterized, material_maps=maps)
else:
    render_pkg = render_surfel(..., material_maps=maps)
```

因此默认的 Ref 20000 -> joint 路径立即复用；从头 joint 在 densification 结束前暂时保留旧的第二次 rasterization，不暗改 topology 统计。

## 5. MVInverse 只执行 A/M/R head

修改 `third_party/mvinverse/mvinverse/models/mvinverse.py`，只增加可选输出选择：

```python
def forward(self, imgs, hidden_patch_residual=None, output_names=None):
    ...
    intermediates = self.decode(hidden_patches, N, H, W)

    if output_names is None:
        output_names = {
            "albedo", "roughness", "metallic", "normal", "shading"
        }

    outputs = {}
    if "albedo" in output_names:
        outputs["albedo"] = self.albedo_head(...)
    if "roughness" in output_names:
        outputs["roughness"] = self.roughness_head(...)
    if "metallic" in output_names:
        outputs["metallic"] = self.metallic_head(...)
    if "normal" in output_names:
        normal = self.normal_head(...)
        outputs["normal"] = F.normalize(normal, ...)
    if "shading" in output_names:
        outputs["shading"] = self.shading_head(...)
    return outputs
```

joint 明确请求：

```python
material = self.mvinverse(
    images.unsqueeze(0),
    hidden_patch_residual=gs_tokens.unsqueeze(0),
    output_names=("albedo", "metallic", "roughness"),
)
```

不传 `output_names` 时仍执行并返回原五项，保证 MVInverse 原推理入口不变。DINO、ResNeXt 和 36 层共享 decoder 仍完整运行；只跳过未被 joint 使用的 `normal_head` 和 `shading_head`。

## 6. 文件和预计修改量

| 文件 | 修改 | 预计规模 |
|---|---|---:|
| `third_party/ref_gaussian/gaussian_renderer/__init__.py` | 提取 `rasterize_surfel`、`shade_surfel`，保留 `render_surfel` wrapper | 约 100 行机械移动；20–30 行新逻辑 |
| `ref_mv/joint.py` | 保存并复用 rasterization；请求 A/M/R | 25–40 行 |
| `third_party/mvinverse/mvinverse/models/mvinverse.py` | 增加 `output_names` 并条件执行 heads | 20–30 行 |
| `third_party/ref_gaussian/train.py` | 传入 `reuse_rasterization` | 2–4 行 |

生产代码预计涉及 4 个现有文件，不新增文件。diff 可能有 150–200 行，但主要是移动原 `render_surfel()` 后半段；真正新增控制逻辑预计 50–70 行。

## 7. 必须保持的边界

- 不修改原始 `/home/disk9/wsq/ENVS/ref-gaussian` 和 `/home/disk9/wsq/ENVS/mvinverse`。
- 不修改 CUDA rasterizer。
- 不修改 Ref loss、PBR 公式、相机和邻居采样。
- 不修改 PLY、`capture()/restore()` 或 `joint.pt` 格式。
- 不增加新的 CLI 参数；`output_names` 和 `reuse_rasterization` 是内部接口。
- 不为本次改动新增 renderer 类、配置层、trainer、cache 类型或自动化测试框架。
- 不启动训练、短程实验或额外评估；代码完成后由用户运行。

## 8. 暂不实施的后续候选

以下仍是候选，不属于当前实施范围：

- 为 MVInverse 使用独立的约 518 分辨率；需要先确认图像、feature、内参和材质回采样。
- 缓存冻结 DINO/ResNeXt 的相机特征；需要评估 CPU/GPU 缓存和传输成本。
- 减少 joint step 频率、改为 `N=1`、activation checkpoint 或整模型 FP16。

只有完成本文件第 4、5 节并获得用户实际运行结果后，才讨论是否进入这些候选。
