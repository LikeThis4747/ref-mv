# Ref-MV 首轮冻结实施方案

状态：**已冻结，可进入实现**  
冻结日期：2026-10-20  
首轮目标：先完成 `C=1, N=1` 可微闭环，再用 `N=2` 验证跨视角路径。除第 12 节列出的运行参数外，编码阶段不再临时改变模型连接、训练边界、loss 或保存格式。

## 1. 冻结范围

### 1.1 新增可学习参数

```text
gs_feature    : [G, C]，C=1，零初始化，随 Gaussian topology 更新
gs_patch_embed: Conv2d(C, 1024, kernel_size=14, stride=14, bias=False)
```

`G` 是当前 Gaussian 数量。`gs_feature` 加入 Ref 原 Adam；`gs_patch_embed` 使用独立 Adam。MVInverse 的 DINO、ResNeXt、decoder、register token 和全部 heads 均 `requires_grad_(False)`，并始终保持 `eval()`。

Ref-Gaussian 不额外冻结，原 optimizer、stage schedule、normal/depth/distortion loss、densification/pruning、opacity reset、mesh 更新和环境贴图训练均继续生效。首轮不加入 LoRA、gate、额外 normalization、alpha normalization、材质 anchor、新 PBR 公式或第二套 topology 逻辑。

### 1.2 冻结数据流

```text
camera RGB ─→ frozen DINO ─→ hidden_patches ───────────────┐
                                                            + ─→ frozen MV decoder/heads ─→ A/M/R
gs_feature ─→ Ref rasterizer ─→ [C,H,W] ─→ patch Conv ─────┘
                                                                          │
                                                                          ↓
GT RGB ←── Ref 原 loss ←── Ref 原 surfel shading（A/M/R override）─────────┘
```

固定映射为：

```text
MV albedo    -> Ref albedo
MV metallic  -> Ref refl_strength
MV roughness -> Ref roughness
```

保留 Ref 原公式 `final_image = (1 - refl_strength) * base_color + specular`。因此首轮中 MV albedo 主要进入 specular 的 `F0`，不会替换 `base_color`；这是已知语义限制，不在实现时暗改。

### 1.3 两条启动流程

只实现以下两种模式，共用同一个 joint surfel step：

1. `saved-surfel`：从完整 Ref saved iteration 加载；联合训练的第一个绝对迭代号为 `loaded_iteration + 1`。
2. `after-initial`：从头执行 Ref initial stage；在 `init_until_iter + 1` 跳过 volume stage，直接进入 joint surfel。

迭代相关 schedule 均使用 Ref 的**绝对迭代号**。`joint_steps` 只决定本次还运行多少步，不把 schedule 重新从 1 开始。若 `saved-surfel` 的起点已超过 densification 或 indirect 阈值，则相应状态按绝对迭代恢复；不人为重新开启已经结束的 schedule。

## 2. 文件级改动与预计代码量

估算口径：非空逻辑行，包含必要注释，不含纯格式化；“修改”包含被替换的原行。实现后允许约 `±25%` 浮动。

| 文件 | 主要职责 | 预计新增 | 预计修改 |
|---|---|---:|---:|
| `third_party/ref_gaussian/utils/camera_utils.py` | joint 模式下 resize 后裁到 patch 倍数，同步 RGB/alpha/refl mask 与相机参数 | 25–35 | 8–12 |
| `third_party/ref_gaussian/scene/__init__.py` | 分离 `load_model_path` 与输出 `model_path` | 10–15 | 8–12 |
| `third_party/ref_gaussian/scene/gaussian_model.py` | `gs_feature` 初始化、getter、optimizer、clone/split/prune | 45–65 | 20–30 |
| `third_party/ref_gaussian/gaussian_renderer/__init__.py` | `render_surfel()` 的 feature-only 返回和 A/M/R override | 30–45 | 8–15 |
| `third_party/ref_gaussian/train.py` | 可选 joint runtime、多视角 loss、统计/step/save/validation 接点 | 100–150 | 50–80 |
| `third_party/mvinverse/mvinverse/models/mvinverse.py` | DINO 最终 token residual 与可选 detached 统计入口 | 16–24 | 8–12 |
| `ref_mv/config.py` | joint 参数、模式与路径校验 | 70–100 | 0 |
| `ref_mv/loaders.py` | 完整 Ref iteration 与 MV checkpoint 加载/冻结 | 80–120 | 0 |
| `ref_mv/joint_model.py` | patch embedding 和联合前向 | 110–150 | 0 |
| `ref_mv/trainer.py` | 邻居采样、joint step、日志与上游循环适配对象 | 130–190 | 0 |
| `ref_mv/checkpoint.py` | joint 状态保存、哈希校验与恢复 | 80–120 | 0 |
| `train.py` | 唯一 CLI 入口，复用 Ref 参数组 | 80–120 | 0 |
| `configs/` | `C=1,N=1` smoke 与 `C=1,N=2` 配置/命令样例 | 40–70 | 0–10 |
| `tests/` | 独立自动化验证目录；只覆盖相机对齐、上游默认回归、topology 对齐和完整梯度链 | 250–400 | 0 |
| `scripts/experiments/` | 短程 overfit、`N=2` smoke 等实验启动脚本；不包含单元测试 | 30–60 | 0 |

预计合计：生产代码、配置与实验脚本新增约 **850–1,260 行**、上游代码修改约 **100–170 行**、自动化验证代码新增约 **250–400 行**；总变更约 **1,200–1,830 行**。不修改原始目录 `/home/disk9/wsq/ENVS/ref-gaussian` 和 `/home/disk9/wsq/ENVS/mvinverse`。

自动化验证代码全部放在根目录 `tests/`，不得写入 `train.py`、`ref_mv/trainer.py`、`scripts/experiments/` 或上游训练文件。它遵循最小必要原则：不追求覆盖率数字，不为简单 getter、参数解析、日志字段或第三方上游内部实现编写重复验证；能在一个端到端 smoke 验证中可靠覆盖的路径，不再拆成多份文件。真实 CUDA 性能、loss 下降和视觉结果通过 `scripts/experiments/` 下的短程实验及实验记录验证，不包装成自动化测试代码。

术语固定如下，避免“训练集测试”“测试代码”和实验运行混在一起：

- **自动化验证**：`tests/` 中由 pytest 执行的代码，只判断接口和梯度不变量。
- **训练**：正式优化流程，只存在于根 `train.py`、`ref_mv/` 和必要的上游接点。
- **实验验证**：短程 overfit、`N=2` smoke、显存和视觉检查，入口放在 `scripts/experiments/`。
- **holdout 评估**：训练期间对未参与当前优化的相机进行指标/可视化评估；联合代码不用 `test` 为其新增对象命名。

上游 Ref 已有的 `testing_iterations`、`getTestCameras()` 等名称为兼容性保留，不做无关重命名；联合新增代码统一使用 `validation_iterations`、`holdout_cameras`、`run_validation()` 等名称。

所有开发副本改动在附近标注：

```python
# [UPSTREAM Ref-Gaussian or MVInverse] source: <commit>/<file>
# [REF-MV MODIFICATION] reason: ...
```

## 3. 相机与像素对齐

### 3.1 接口

不修改 Ref 参数类。根入口在 `ModelParams.extract()` 之后动态设置：

```python
dataset.joint_patch_size = 14 if joint.enabled else 0
```

`loadCam()` 用 `getattr(args, "joint_patch_size", 0)` 读取；字段不存在或为 0 时严格走原路径。

### 3.2 精确伪代码

```python
def loadCam(args, id, cam_info, resolution_scale):
    # [UPSTREAM] 原分辨率与 scale 计算
    resolution = ref_resolution(...)
    scale = ref_scale(...)

    K = None
    if cam_info.K is not None:
        K = cam_info.K.copy()
        K[:2] /= scale

    # [UPSTREAM] 先按 resolution resize RGB/alpha
    rgb, alpha = load_ref_image_and_alpha(cam_info.image, resolution)

    # refl mask 当前上游直接读原尺寸；joint 路径必须先 resize 到同一 resolution
    refl_mask = load_optional_refl_mask(...)
    if refl_mask is not None and refl_mask.shape[-2:] != (resolution[1], resolution[0]):
        refl_mask = resize_mask_nearest(refl_mask, resolution)

    patch = getattr(args, "joint_patch_size", 0)
    if patch:
        width = (resolution[0] // patch) * patch
        height = (resolution[1] // patch) * patch
        if width <= 0 or height <= 0:
            raise ValueError(...)
        rgb = rgb[..., :height, :width]
        alpha = crop_if_present(alpha, height, width)
        refl_mask = crop_if_present(refl_mask, height, width)
        resolution = (width, height)

    HWK = None if K is None else (resolution[1], resolution[0], K)
    fov_x = cam_info.FovX if K is None else focal2fov(K[0, 0], resolution[0])
    fov_y = cam_info.FovY if K is None else focal2fov(K[1, 1], resolution[1])
    return Camera(..., FoVx=fov_x, FoVy=fov_y, image=rgb,
                  gt_alpha_mask=alpha, HWK=HWK, gt_refl_mask=refl_mask)
```

右/下裁切不移动左上角原点，因此已缩放 `K` 的 `cx/cy` 不变。必须同时更新 FoV，因为 rasterizer 用 FoV 计算 `tanfovx/y`；`Camera` 仍用裁后 `H/W/K` 调用 `getProjectionMatrixCorrect()`。无 `K` 的数据暂不改内参语义，只要求裁后 RGB 与 FoV 路径的 renderer 回归通过；首轮目标数据应使用带 `K` 的 Ref 数据。

## 4. Gaussian feature 与 topology

### 4.1 生命周期

`GaussianModel.__init__()` 增加未启用哨兵，而不是立刻创建可学习参数：

```python
self._gs_feature = None
self.gs_feature_dim = 0

@property
def get_gs_feature(self):
    if self._gs_feature is None:
        raise RuntimeError("gs_feature is not initialized")
    return self._gs_feature
```

在 `load_ply()/create_from_pcd()` 和原 `training_setup()` 完成后启用：

```python
def initialize_gs_feature(self, feature_dim, lr, initial_value=None):
    if self._gs_feature is not None:
        raise RuntimeError("gs_feature already initialized")
    value = initial_value if initial_value is not None else zeros(G, feature_dim)
    validate(value.shape == (self.get_xyz.shape[0], feature_dim))
    self._gs_feature = nn.Parameter(value.to(self.get_xyz))
    self.gs_feature_dim = feature_dim
    self.optimizer.add_param_group({
        "params": [self._gs_feature], "lr": lr, "name": "gs_feature"
    })
```

首版不修改 PLY 字段和 `capture()/restore()` tuple。feature 始终由独立 joint checkpoint 保存。

### 4.2 clone/split/prune

原 optimizer helper 会遍历所有非 `mlp/env/env2` 参数组，因此一旦加入 `gs_feature`，传给 `cat_tensors_to_optimizer()` 的字典必须总含同名项：

```python
# split
new_gs_feature = self._gs_feature[selected_pts_mask].repeat(N, 1)

# clone
new_gs_feature = self._gs_feature[selected_pts_mask]

# densification_postfix
d = { ... }
if self._gs_feature is not None:
    d["gs_feature"] = new_gs_feature
optimizable = self.cat_tensors_to_optimizer(d)
if self._gs_feature is not None:
    self._gs_feature = optimizable["gs_feature"]

# prune_points
optimizable = self._prune_optimizer(valid_points_mask)
if self._gs_feature is not None:
    self._gs_feature = optimizable["gs_feature"]
```

`gs_feature is None` 时所有函数的默认行为与上游一致。测试必须覆盖 clone、split、prune 后 feature 行数、值复制规则、optimizer 参数引用和 Adam state 形状。

## 5. Renderer 最小接口

### 5.1 签名与约束

```python
def render_surfel(
    viewpoint_camera, pc, pipe, bg_color,
    scaling_modifier=1.0, override_color=None, srgb=False, opt=None,
    gaussian_feature=None,
    material_maps=None,
    return_gs_feature=False,
):
```

约束：

- `return_gs_feature=True` 时必须传 `gaussian_feature`。
- `material_maps` 必须同时含 `albedo/metallic/roughness`，形状分别为 `[H,W,3]`、`[H,W,1]`、`[H,W,1]`。
- 默认参数全部为空/False 时，返回键和值与上游逐项一致。
- feature-only pass 只返回 `{"gs_feature": [C,H,W]}`，不参与 densification 统计。

### 5.2 精确伪代码

```python
feature_parts = [refl, roughness, ori_color, indirect]  # 1 + 1 + 3 + 3 = 8
if gaussian_feature is not None:
    validate(gaussian_feature.ndim == 2)
    validate(gaussian_feature.shape[0] == pc.get_xyz.shape[0])
    feature_parts.append(gaussian_feature)

contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
    ...,
    features=torch.cat(feature_parts, dim=-1),
    ...,
)

refl_strength = rendered_features[0:1]
roughness = rendered_features[1:2]
albedo = rendered_features[2:5]
indirect_light = rendered_features[5:8]

if return_gs_feature:
    c = gaussian_feature.shape[1]
    return {"gs_feature": rendered_features[8:8 + c]}

# [UPSTREAM] compute_2dgs_normal_and_regularizations 保持原位置
...

if material_maps is not None:
    validate_material_shapes(material_maps, H, W)
    albedo = material_maps["albedo"].permute(2, 0, 1).to(rendered_image)
    refl_strength = material_maps["metallic"].permute(2, 0, 1).to(rendered_image)
    roughness = material_maps["roughness"].permute(2, 0, 1).to(rendered_image)

# [UPSTREAM] get_specular_color_surfel、srgb、背景合成和结果字典不改
```

材质 override 必须位于 normal/alpha 计算之后、`get_specular_color_surfel()` 之前。`.to(rendered_image)` 只做可微 dtype/device 转换，不 detach、不 clamp、不 gamma 转换。

每个 joint 视角调用两次 renderer：第一次 feature-only，第二次 material override 的完整 surfel render。首轮接受两次 rasterization。

## 6. MVInverse residual 接口

```python
def forward(self, imgs, hidden_patch_residual=None, return_token_stats=False):
    B, N, _, H, W = imgs.shape
    if H % self.patch_size or W % self.patch_size:
        raise ValueError(...)

    imgs_flat = imgs.reshape(B * N, 3, H, W)
    imgs_norm = (imgs_flat - self.image_mean) / self.image_std

    # 原 DINO/ResNeXt 路径
    hidden_patches = self.encoder(
        imgs_norm, is_training=True
    )["x_norm_patchtokens"]                       # [B*N,L,1024]
    res_features = self._get_resnext_features(imgs_norm)

    token_stats = None
    if hidden_patch_residual is not None:
        residual = hidden_patch_residual.reshape(B * N, -1, hidden_patches.shape[-1])
        if residual.shape != hidden_patches.shape:
            raise ValueError(...)
        if return_token_stats:
            token_stats = {
                "hidden_rms": hidden_patches.detach().float().square().mean().sqrt(),
                "residual_rms": residual.detach().float().square().mean().sqrt(),
            }
        hidden_patches = hidden_patches + residual.to(hidden_patches)

    intermediates = self.decode(hidden_patches, N, H, W)
    outputs = original_heads(intermediates, imgs, res_features)
    if return_token_stats:
        outputs["token_stats"] = token_stats
    return outputs
```

统计值必须 detach，不进入 loss。调用 `model(imgs)` 与原版一致；零 residual 的五个原输出必须一致。冻结后不能把整个 MV 前向放进 `torch.no_grad()`，因为 residual 经 decoder/head 到 A/M/R 的梯度图必须保留。

## 7. 联合模型

```python
class RefMVModel(nn.Module):
    def __init__(self, gaussians, mvinverse, feature_dim):
        super().__init__()
        self.gaussians = gaussians       # GaussianModel 不是 nn.Module
        self.mvinverse = mvinverse.eval().requires_grad_(False)
        self.gs_patch_embed = nn.Conv2d(
            feature_dim, 1024, kernel_size=14, stride=14, bias=False
        )

    def train(self, mode=True):
        super().train(mode)
        self.mvinverse.eval()
        return self

    def forward(self, cameras, pipe, background, opt):
        images, feature_maps = [], []
        for camera in cameras:
            validate(camera.original_image.shape[-2:] ==
                     (camera.image_height, camera.image_width))
            out = render_surfel(
                camera, self.gaussians, pipe, background,
                srgb=opt.srgb, opt=opt,
                gaussian_feature=self.gaussians.get_gs_feature,
                return_gs_feature=True,
            )
            images.append(camera.original_image[:3])
            feature_maps.append(out["gs_feature"])

        images = torch.stack(images)             # [N,3,H,W]
        features = torch.stack(feature_maps)     # [N,C,H,W]
        H, W = images.shape[-2:]
        validate(H % 14 == 0 and W % 14 == 0)

        gs_tokens = self.gs_patch_embed(features)
        gs_tokens = gs_tokens.flatten(2).transpose(1, 2)  # [N,L,1024]

        material = self.mvinverse(
            images.unsqueeze(0),
            hidden_patch_residual=gs_tokens.unsqueeze(0),
            return_token_stats=True,
        )

        render_pkgs = []
        for i, camera in enumerate(cameras):
            maps = {
                "albedo": material["albedo"][0, i],
                "metallic": material["metallic"][0, i],
                "roughness": material["roughness"][0, i],
            }
            render_pkgs.append(render_surfel(
                camera, self.gaussians, pipe, background,
                srgb=opt.srgb, opt=opt, material_maps=maps,
            ))

        token_stats = material.pop("token_stats")
        return JointOutput(render_pkgs, material, features, gs_tokens, token_stats)
```

`token_stats` 是可选 detached side-channel；默认关闭，因此不改变上游输出字典。不得为了日志复制一遍 DINO。

梯度预期不是“只有新增参数有梯度”：

- feature pass 的 splat 权重使条件分支也可对 Gaussian 位置、尺度、旋转、opacity 求导；
- material override 后，joint RGB 不再通过原二维 `refl/roughness/ori_color` map 回传到 `_refl_strength/_roughness/_ori_color`；
- SH `base_color`、几何、opacity、环境光仍沿 Ref 原公式获得梯度；
- MV 全部参数必须无 `.grad`，但 residual 输入必须有梯度。

## 8. 加载、输出目录与冻结

### 8.1 输入输出分离

不能让 joint 结果覆盖 Ref 基线。给 `Scene` 增加可选 `load_model_path`：

```python
class Scene:
    def __init__(..., load_iteration=None, load_model_path=None, ...):
        self.model_path = args.model_path             # joint 输出
        self.load_model_path = load_model_path or self.model_path
        # search/load PLY 与 maps 使用 load_model_path
        # save() 始终使用 model_path
```

`saved-surfel` 启动顺序：

```python
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(
    dataset, gaussians,
    load_iteration=cfg.ref_iteration,
    load_model_path=cfg.ref_model_path,
    shuffle=False,
)
loaded_iteration = scene.loaded_iter

if loaded_iteration >= opt.indirect_from_iter:
    opt.indirect = 1
if opt.indirect:
    require(ref_model_path / f"test_{loaded_iteration:06d}.ply")
    gaussians.load_mesh_from_ply(ref_model_path, loaded_iteration)

gaussians.training_setup(opt)
gaussians.initialize_gs_feature(C, gs_feature_lr, restored_feature_or_none)
```

`Scene.load_ply()` 已从同一目录读取 `point_cloud.ply`、`point_cloud1.map`、`point_cloud2.map`。加载器额外校验 env maps 非空、source path/分辨率与当前配置一致，不使用单独 `chkpnt*.pth` 代替完整 iteration。

MV 严格复用 `inference.py::load_model()`；随后执行：

```python
mvinverse.eval()
mvinverse.requires_grad_(False)
```

### 8.2 迭代边界

```python
if mode == "saved-surfel":
    first_iteration = loaded_iteration + 1
    last_iteration = loaded_iteration + joint_steps
    force_surfel_from = first_iteration
elif mode == "after-initial":
    first_iteration = 1
    last_iteration = opt.init_until_iter + joint_steps
    force_surfel_from = opt.init_until_iter + 1
```

`after-initial` 在 `force_surfel_from` 初始化 joint 参数与 MV；initial 阶段不加载 MV、不创建 `gs_feature`，避免无用显存占用。两种模式都从进入 joint 的第一步开始启用 patch 对齐相机，因此 `after-initial` 必须从数据加载起就设置 `joint_patch_size=14`，不能阶段中途改变相机尺寸。

## 9. 采样与一次 joint step

### 9.1 邻居表

```python
centers = torch.stack([cam.camera_center.detach().cpu() for cam in train_cameras])
dist = torch.cdist(centers, centers)
k_eff = min(K, len(train_cameras) - 1)
neighbor_table = dist.topk(k_eff + 1, largest=False).indices[:, 1:]
```

要求 `1 <= N <= k_eff + 1`。anchor 在一个 epoch 内采用 Ref 原“无放回随机栈”语义；其余 `N-1` 个视角从 anchor 的 top-K 中无放回采样。`N=1` 不访问邻居表。

### 9.2 精确训练伪代码

```python
anchor_idx = anchor_sampler.pop_or_refill()
camera_indices = sample_anchor_and_neighbors(anchor_idx, N, neighbor_table)
cameras = [train_cameras[i] for i in camera_indices]

gaussians.optimizer.zero_grad(set_to_none=True)
patch_optimizer.zero_grad(set_to_none=True)

joint = model(cameras, pipe, background, opt)
per_view_losses, per_view_tb = [], []
for camera, render_pkg in zip(cameras, joint.render_pkgs):
    loss_i, tb_i = calculate_loss(camera, gaussians, render_pkg, opt, iteration)
    per_view_losses.append(loss_i)
    per_view_tb.append(tb_i)

total_loss = torch.stack(per_view_losses).mean()
if use_env_scope:
    # 与上游同样只加一次 Gaussian-space regularizer，不能每视角重复
    total_loss = total_loss + ref_env_scope_regularizer(...)

total_loss.backward()

with torch.no_grad():
    if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
        for pkg in joint.render_pkgs:             # 只统计第二次完整 render
            visible = pkg["visibility_filter"]
            gaussians.max_radii2D[visible] = maximum(
                gaussians.max_radii2D[visible], pkg["radii"][visible]
            )
            grad = pkg["viewspace_points"].grad
            if grad is None:
                raise RuntimeError("missing viewspace gradient")
            grad.mul_(N)                          # 只校正 densification heuristic
            gaussians.add_densification_stats(pkg["viewspace_points"], visible)

        run_original_densify_prune_reset_schedule_once(...)

    run_original_mesh_update_schedule_once(...)

gaussians.optimizer.step()
patch_optimizer.step()
```

loss 对视角取平均；Ref 的 photometric、normal、depth、distortion 项均由原 `calculate_loss()` 为每个视角计算后一起平均。`viewspace_points.grad *= N` 发生在 backward 后，只影响随后读取的 densification 统计，不改变已经累计到 Gaussian 参数的平均梯度。第一次 feature-only pass 不更新 `max_radii2D`、不调用 `add_densification_stats()`。

### 9.3 原循环中的接点

`third_party/ref_gaussian/train.py::training()` 增加默认 `joint_runtime=None`：

- `None`：逐行保持原单视角路径；用于基线回归。
- 非空且尚在 initial：仍走原 initial 路径。
- 非空且进入 joint：相机选择、forward、loss 聚合、densification 多视角统计、两个 optimizer step 和 joint save 委托给 runtime；SH 升阶、env mip、reset、mesh、report 的触发顺序不变。

验证/可视化在 joint 阶段使用 `JointRuntime.render_single(camera)`（即 `N=1` 条件渲染），不能退回未 conditioned 的原 `render_surfel()`。跨视角质量另用固定 `N=2` holdout 组记录，不混入训练 loss。

## 10. 保存与恢复

在 Ref 原 `scene.save(iteration)` 写出 joint 输出目录下的 PLY/env maps/mesh 后，再写同 iteration 的 joint checkpoint。二者都位于新输出目录，不覆盖输入基线。

```python
joint_state = {
    "schema_version": 1,
    "absolute_iteration": iteration,
    "gaussian_feature": gaussians.get_gs_feature.detach().cpu(),
    "gs_patch_embed": model.gs_patch_embed.state_dict(),
    "patch_optimizer": patch_optimizer.state_dict(),
    "joint_config": asdict(cfg),
    "ref_source_model_path": str(ref_model_path),
    "ref_source_iteration": loaded_iteration_or_none,
    "saved_ply_sha256": sha256(current_output_ply),
    "mv_checkpoint": str(mv_checkpoint),
    "num_gaussians": gaussians.get_xyz.shape[0],
    "feature_dim": C,
}
atomic_torch_save(joint_state, joint_checkpoint_path)
```

恢复顺序固定为：

1. 从 joint 输出目录加载 checkpoint 对应的完整 saved iteration；
2. 校验 schema、绝对 iteration、PLY SHA256、Gaussian 数量、feature_dim 和 MV 身份；
3. `training_setup()` 后用 checkpoint tensor 初始化 `gs_feature` 并加入 Ref optimizer；
4. 创建 `gs_patch_embed` 和 patch optimizer，再加载二者 state；
5. 恢复 anchor sampler RNG/栈（若 checkpoint 中存在），否则明确记录采样序列不精确续接。

首版仍不承诺 Ref Adam 动量的精确续接，因为 saved iteration 不包含其 optimizer state，且上游 `restore()` 当前注释掉了 `load_state_dict()`。这一限制必须打印到日志。

## 11. 日志与验收顺序

### 11.1 每步/定期日志

- mean total loss、各 Ref loss 分量、train/holdout PSNR；
- A/M/R 的 min/max/mean 与直方图；
- `gs_feature`、`gs_patch_embed.weight` 的值 RMS 和梯度范数；
- `hidden_patches`、`gs_tokens` RMS 及比值（仅 detached 统计）；
- Gaussian 数量、每次 clone/split/prune 数量；
- 每步耗时与峰值显存；
- 固定视角的 RGB、A/M/R、feature map；固定 `N=2` holdout 组的输出。

### 11.2 最小自动化验证与实验验证

`tests/` 只保留四组自动化验证：

1. **相机对齐验证**：一次覆盖裁后尺寸、RGB/alpha/refl mask、`K`、FoV、HWK 和 projection；不为每个字段单独建文件。
2. **上游接口回归验证**：一次覆盖 renderer 默认参数与原输出一致，以及 MV 零 residual 与原五个输出一致。
3. **Gaussian topology 验证**：用一个小型构造样例覆盖 feature 初始化和 clone/split/prune 后的行数、复制值及 optimizer 引用；不验证上游已有 topology 数学。
4. **联合梯度验证**：`C=1,N=1` 连续两步，同时验证 rasterizer→patch→MV→renderer 的梯度链、MV 无参数梯度、patch 首步/次步梯度预期，以及 feature-only pass 不进入 densification 统计。只有多视角统计的 `1/N` 校正确实无法在这里覆盖时，才补一个小验证。

`scripts/experiments/` 只放两类实验验证入口：

1. **短程 overfit**：`C=1,N=1` 检查无 NaN、loss 可下降、checkpoint 可保存恢复。
2. **跨视角 smoke**：`C=1,N=2` 检查 decoder、平均 loss、显存和视觉输出。

两类实验的输出写入独立实验目录并记录配置，不进入 `tests/`，也不由 pytest 启动。任何一项验证失败都停在对应层修复，不直接启动长训练。

## 12. 实现前只需提供的运行参数

以下不是架构未决项，不阻塞编码与自动化验证：

- `ref_model_path + ref_iteration`（`saved-surfel` 模式）；
- MV checkpoint 路径或 Hugging Face repo id；
- 新的 joint 输出目录；
- `gs_feature_lr` 与 `gs_patch_lr`：没有基线值，先由两步梯度和极短 overfit 选定；
- `neighbor_k` 与正式 `N`：smoke 固定 `N=1/2`，正式值依据显存；
- MV albedo 首轮直接传入 Ref shader，不做隐式 sRGB→linear；若异常，后续另立对照实验；
- top-K 首轮只用相机中心距离；若出现近位置、反方向相机，再新增方向约束实验，不改变当前采样接口。

上述参数必须写入配置和 checkpoint。除这些数值外，当前讨论已冻结；如需改变模型结构、loss、冻结范围、renderer 公式或保存语义，应先新增决策记录并更新本文件。
