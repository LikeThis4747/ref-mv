# Ref-MV 首轮冻结实施方案

> **工作原则：**一定要尽量与基线保持一致，能复用就复用；实现时结构保持简单，不过度设计。有任何不确定的问题或假设，必须及时汇报并讨论确认，不能基于未确认假设继续实现。在哪一步就只做好哪一步的事；当前职责是实现代码，不提前实现自动化测试、实验或训练辅助功能，也不代替用户启动训练。

状态：**已按本文完成代码实现，尚未运行**
冻结日期：2026-10-20  
最近修订：2026-10-23（删除两套启动模式；完成实施细节基线复核）
首轮目标：实现已确认的 `C=1` 多视角可微闭环。当前阶段只写实现代码，不安排或启动训练、实验及自动化测试。除第 12 节列出的运行参数外，编码阶段不再临时改变模型连接、训练边界、loss 或保存格式。

## 0. 2026-10-23 复核结论

1. **训练期间查看结果。** 这里只要求保留 Ref 上游定期渲染训练/测试相机、计算 PSNR 和保存图像的原有逻辑，不另造新的相机配对、指标或运行流程。本阶段不替用户启动任何 `N=1`、`N=2` 或正式训练。
2. **图像尺寸处理。** 完全复用 Ref 原 `-r` 先降分辨率，然后再从右边/下边裁到 14 的倍数；不增加第二套 resize、padding 或相机分组。只在把 `N` 张图叠成一个 tensor 前检查它们的 `H/W` 是否一致；不一致就报出原始尺寸并停止。该检查只需几行代码；如实现时发现需要大范围改相机逻辑，先汇报。
3. **joint 续训。** 保存位置固定为 `point_cloud/iteration_<I>/joint.pt`。加载时使用可选的显式 `joint_checkpoint` 参数：未传则创建新 joint 参数，传入才恢复，不根据文件是否存在猜测用户意图。
4. **indirect mesh。** 上游 indirect shading 需要同 iteration 的 ray-tracing mesh，而上游出于保存成本考虑不会在每次保存 Gaussian 时都写 mesh。这里不修改上游保存频率，也不增加额外的运行前检查；后续运行时由用户选择本来就带有对应 mesh 的 saved iteration。
5. **训练入口。** 这不作为研究或架构问题。实现时只做能复用上游 parser 和 `training()` 的最小接入，不复制参数系统，不引入 CLI 框架。如果实际改动超出第 2 节估算，先汇报。
6. **`refl_mask`。** 它是上游相机可选加载的“反射区域 mask”。当前上游训练、loss 和本联合路径都没有读取它，因此对首轮结果没有影响。为减少无关改动，首轮保留上游行为，不额外 resize/裁切它；未来如果真正用到该 mask，再与 RGB 同步处理。

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

### 1.3 两个使用入口，一个 renderer 切换规则

这里不实现两套训练模式或额外状态机。唯一规则是：**原 Ref 循环选择 `render_surfel` 时，联合入口把这次调用替换为 `joint_surfel`；`render_initial` 和 `render_volume` 均保持原样。**

用户可按原训练方式从两个入口开始：

1. **加载现成 Ref Gaussian**：默认该 saved iteration 已经跑过 surfel。加载完整 Gaussian、环境贴图和必要 mesh 后，第一步原本会调用 `render_surfel`，因此直接进入 `joint_surfel`。
2. **从头训练**：先按原流程运行一次 initial 阶段；训练命令把 `volume_render_until_iter` 设为 `0`，initial 结束后原循环自然选择 `render_surfel`，此时自动改走 `joint_surfel`。

不新增 `saved-surfel`、`after-initial`、`force_surfel_from` 或 `joint_steps` 等概念。开始/结束迭代仍使用 Ref 原 `first_iter/iterations`；加载 saved iteration 时从 `loaded_iteration + 1` 继续，从头训练时按原循环从第 1 轮开始。

## 2. 文件级改动与预计代码量

估算口径：非空逻辑行，包含必要注释，不含纯格式化；“修改”包含被替换的原行。实现后允许约 `±25%` 浮动。

| 文件 | 主要职责 | 预计新增 | 预计修改 |
|---|---|---:|---:|
| `third_party/ref_gaussian/utils/camera_utils.py` | joint 模式下 resize 后裁到 patch 倍数，同步 RGB/alpha 与相机参数 | 12–20 | 5–8 |
| `third_party/ref_gaussian/scene/__init__.py` | 分离 `load_model_path` 与输出 `model_path` | 10–15 | 8–12 |
| `third_party/ref_gaussian/scene/gaussian_model.py` | `gs_feature` 初始化、getter、optimizer、clone/split/prune | 40–55 | 15–25 |
| `third_party/ref_gaussian/gaussian_renderer/__init__.py` | `render_surfel()` 的 feature-only 返回和 A/M/R override | 25–35 | 8–12 |
| `third_party/ref_gaussian/train.py` | 在原循环的 surfel 分支接入多视角 joint forward/loss/step | 70–100 | 35–55 |
| `third_party/mvinverse/mvinverse/models/mvinverse.py` | DINO 最终 token residual 可选入口 | 6–10 | 4–8 |
| `ref_mv/joint.py` | patch embedding、联合前向、相机邻居采样和最小 joint 状态保存 | 180–260 | 0 |
| `train.py` | 唯一 CLI 入口，直接复用 Ref 参数组并增加少量 joint 参数 | 40–70 | 0 |
| `configs/` | 两条命令样例，不另建配置框架 | 10–20 | 0–5 |

预计合计：生产代码与配置新增约 **390–590 行**、上游代码修改约 **75–120 行**；总变更约 **465–710 行**。不修改原始目录 `/home/disk9/wsq/ENVS/ref-gaussian` 和 `/home/disk9/wsq/ENVS/mvinverse`。

首轮不创建 `config.py`、`loaders.py`、`trainer.py`、`checkpoint.py` 等分层模块。少量 joint 参数直接加到根入口；联合前向、采样和简单保存集中在 `ref_mv/joint.py`。只有单文件明显失控时，才在实现后按实际职责拆分。

本轮不新增 `tests/` 内容或 `scripts/experiments/`，也不为测试便利改造生产代码。实现结束后只交付代码改动说明和运行所需参数；训练、实验及其执行顺序由用户另行安排。上游 Ref 已有的 `testing_iterations`、`getTestCameras()` 等名称和行为原样保留。

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

    patch = getattr(args, "joint_patch_size", 0)
    if patch:
        width = (resolution[0] // patch) * patch
        height = (resolution[1] // patch) * patch
        if width <= 0 or height <= 0:
            raise ValueError(...)
        rgb = rgb[..., :height, :width]
        alpha = crop_if_present(alpha, height, width)
        resolution = (width, height)

    HWK = None if K is None else (resolution[1], resolution[0], K)
    fov_x = cam_info.FovX if K is None else focal2fov(K[0, 0], resolution[0])
    fov_y = cam_info.FovY if K is None else focal2fov(K[1, 1], resolution[1])
    return Camera(..., FoVx=fov_x, FoVy=fov_y, image=rgb,
                  gt_alpha_mask=alpha, HWK=HWK, gt_refl_mask=original_refl_mask)
```

右/下裁切不移动左上角原点，因此已缩放 `K` 的 `cx/cy` 不变。必须同时更新 FoV，因为 rasterizer 用 FoV 计算 `tanfovx/y`；`Camera` 仍用裁后 `H/W/K` 调用 `getProjectionMatrixCorrect()`。当前 `refl_mask` 不参与本方案的 loss 或 joint forward，首轮不顺手改它。无 `K` 的数据暂不改内参语义；首轮目标数据使用带 `K` 的 Ref 数据。

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

`gs_feature is None` 时所有函数的默认行为与上游一致。实现必须保持 clone、split、prune 后 feature 行数、值复制规则、optimizer 参数引用和 Adam state 形状一致，不为此额外建设自动化测试。

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
- `8 + gaussian_feature.shape[1] <= 24`；这是当前 CUDA rasterizer 的 `MAX_FEATURES` 硬上限，超出时必须在 Python 入口 fail fast。
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
def forward(self, imgs, hidden_patch_residual=None):
    B, N, _, H, W = imgs.shape
    imgs_flat = imgs.reshape(B * N, 3, H, W)
    imgs_norm = (imgs_flat - self.image_mean) / self.image_std

    # 原 DINO/ResNeXt 路径
    hidden_patches = self.encoder(
        imgs_norm, is_training=True
    )["x_norm_patchtokens"]                       # [B*N,L,1024]
    res_features = self._get_resnext_features(imgs_norm)

    if hidden_patch_residual is not None:
        residual = hidden_patch_residual.reshape(B * N, -1, hidden_patches.shape[-1])
        if residual.shape != hidden_patches.shape:
            raise ValueError(...)
        hidden_patches = hidden_patches + residual.to(hidden_patches)

    intermediates = self.decode(hidden_patches, N, H, W)
    return original_heads(intermediates, imgs, res_features)
```

调用 `model(imgs)` 与原版一致；零 residual 的五个原输出必须一致。首轮不为日志给 MV 增加 token 统计 side-channel。冻结后不能把整个 MV 前向放进 `torch.no_grad()`，因为 residual 经 decoder/head 到 A/M/R 的梯度图必须保留。
联合调用沿用 MVInverse 官方 `inference.py` 的 CUDA fp16 autocast；只冻结参数，不切断 residual 输入的梯度。

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

        return {
            "render_pkgs": render_pkgs,
            "material": material,
            "feature_maps": features,
            "gs_tokens": gs_tokens,
        }
```

首轮直接返回普通字典，不新增 `JointOutput` dataclass，也不为了日志复制或暴露 DINO 中间结果。

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

加载现成 Ref Gaussian 时沿用以下顺序：

```python
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(
    dataset, gaussians,
    load_iteration=ref_iteration,
    load_model_path=ref_model_path,
    shuffle=False,
)
loaded_iteration = scene.loaded_iter
first_iter = 0 if loaded_iteration is None else loaded_iteration

if loaded_iteration >= opt.indirect_from_iter:
    opt.indirect = 1
if opt.indirect:
    gaussians.load_mesh_from_ply(ref_model_path, loaded_iteration)

gaussians.training_setup(opt)
```

`Scene` 会调用 `GaussianModel.load_ply()`，上游同时从 PLY 旁读取 `point_cloud1.map` 和 `point_cloud2.map`。这里沿用上游加载行为，不额外增加 mesh 存在性检查，也不修改 mesh 保存调度；需要 indirect 时由用户选择带对应 mesh 的 saved iteration。不使用单独 `chkpnt*.pth` 代替完整 iteration。

`first_iter` 必须显式取 `loaded_iteration`；上游当前只在加载 `chkpnt*.pth` 时更新 `first_iter`，仅加载 PLY 不会自动续上绝对迭代号。保留上游随后的 `first_iter += 1`，因此第一个新 step 是 `loaded_iteration + 1`。

这里仅恢复 Ref 状态，不提前创建 joint 参数。第一次原循环选择 `render_surfel` 时，调用 `ref_mv/joint.py` 中的简单初始化函数创建 `gs_feature` 和 patch optimizer。MV 严格复用 `inference.py::load_model()`，并在同一次初始化中执行：

```python
mvinverse.eval()
mvinverse.requires_grad_(False)
```

### 8.2 统一 renderer 切换

```python
render_fn = select_render_method(iteration, opt, initial_stage)

if joint_enabled and render_fn is render_surfel:
    if joint_model is None:
        joint_model, patch_optimizer = initialize_joint(
            gaussians, mv_checkpoint, C, gs_feature_lr, gs_patch_lr
        )
    run_joint_step(joint_model, patch_optimizer, ...)
else:
    # render_initial / render_volume 以及未启用 joint 时均走原路径
    run_original_ref_step(render_fn, ...)
```

`joint_model is None` 是唯一的一次性懒加载判断：第一次遇到 `render_surfel` 时，按当时的 Gaussian 数量创建零初始化 `gs_feature`、加载冻结 MV 并创建 patch optimizer。首轮不为这一判断设计 runtime 类、状态机或生命周期框架。

两个入口都从数据加载起启用 `joint_patch_size=14`，保证 initial 与后续 joint 使用同一批相机尺寸，不能在阶段切换时改变分辨率。从头训练时只需在原命令中设置 `volume_render_until_iter=0`；代码不负责强制跳过 volume。

## 9. 采样与一次 joint step

### 9.1 邻居表

```python
centers = torch.stack([cam.camera_center.detach().cpu() for cam in train_cameras])
dist = torch.cdist(centers, centers)
k_eff = min(K, len(train_cameras) - 1)
if k_eff == 0:
    neighbor_table = torch.empty((len(train_cameras), 0), dtype=torch.long)
else:
    dist.fill_diagonal_(float("inf"))
    neighbor_table = dist.topk(k_eff, largest=False).indices
```

要求 `1 <= N <= k_eff + 1`。先把对角线设为无穷大，避免存在重合相机中心时依赖 `topk` 的 tie 顺序并把 anchor 自身选回。anchor 在一个 epoch 内采用 Ref 原“无放回随机栈”语义；其余 `N-1` 个视角从 anchor 的 top-K 中无放回采样。`N=1` 不访问邻居表，单相机数据不构建 `topk(0)`。

### 9.2 精确训练伪代码

```python
anchor_idx = anchor_sampler.pop_or_refill()
camera_indices = sample_anchor_and_neighbors(anchor_idx, N, neighbor_table)
cameras = [train_cameras[i] for i in camera_indices]

gaussians.optimizer.zero_grad(set_to_none=True)
patch_optimizer.zero_grad(set_to_none=True)

joint = model(cameras, pipe, background, opt)
per_view_losses, per_view_tb = [], []
for camera, render_pkg in zip(cameras, joint["render_pkgs"]):
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
        for pkg in joint["render_pkgs"]:          # 只统计第二次完整 render
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

`third_party/ref_gaussian/train.py::training()` 只增加 `joint_enabled` 和必要 joint 参数：

- `joint_enabled=False`：逐行保持原单视角路径。
- `render_initial` 或 `render_volume`：仍走原路径。
- `render_surfel` 且启用 joint：在原位置执行相关视角采样、joint forward、多视角 loss 和两个 optimizer step；不委托给新的 trainer/runtime 框架。SH 升阶、env mip、reset、mesh、report 的触发顺序不变。

训练期间原有的结果查看逻辑不能退回未 conditioned 的原 `render_surfel()`。它只复用上游已有调用位置，不新增测试相机配对、额外评估或训练运行。

## 10. 保存与恢复

在 Ref 原 `scene.save(iteration)` 写出 joint 输出目录下的 PLY/env maps/mesh 后，再将 Ref PLY 不包含的新增状态保存到同一 iteration 目录下的 `joint.pt`。二者都位于新输出目录，不覆盖输入基线。只有显式传入 `joint_checkpoint` 时才恢复已有 joint 状态。

```python
joint_state = {
    "iteration": iteration,
    "gaussian_feature": gaussians.get_gs_feature.detach().cpu(),
    "gs_patch_embed": model.gs_patch_embed.state_dict(),
    "num_gaussians": gaussians.get_xyz.shape[0],
    "feature_dim": C,
}
torch.save(joint_state, joint_checkpoint_path)
```

恢复时先加载同 iteration 的完整 Ref saved iteration，再检查 `iteration`、Gaussian 数量和 `feature_dim`，最后恢复 feature 与 patch convolution。由于 Ref Adam 本身不能精确续接，首轮也不单独保存 patch optimizer 动量，避免形成一半续接、一半重置的状态。首轮不加入 schema version、PLY SHA256、atomic save、MV 身份校验、完整配置快照或相机采样 RNG 恢复；这些只有在实际出现错配或需要严格复现实验续跑时再增加。

首版仍不承诺 Ref Adam 动量或相机采样序列的精确续接，因为 saved iteration 不包含前者的有效恢复路径；日志中说明一次即可，不为此扩展格式。

## 11. 日志与实现交付边界

### 11.1 每步/定期日志

- 复用 Ref 原 loss、PSNR、Gaussian 数量日志以及原保存时机；
- 不额外记录梯度范数、耗时或峰值显存；
- 不额外保存 RGB、A/M/R、feature map 或测试相机结果。

首轮不加入直方图、DINO token RMS side-channel、额外测试相机分组管理或每次 clone/split/prune 的额外计数。

### 11.2 交付边界

本轮只实现生产代码，不新增自动化测试、短程训练脚本、overfit 脚本或额外评估入口，也不启动任何训练。实现完成后只汇报修改文件、与上游的差异、尚未确定的运行参数及需要用户运行时关注的事项。

## 12. 实现前只需提供的运行参数

以下不是架构未决项，不阻塞代码实现，但在用户实际运行前需要由用户提供或确认：

- `ref_model_path + ref_iteration`（选择加载现成 Ref Gaussian 时）；
- 可选 `joint_checkpoint`（只在明确续训已有 joint 状态时提供）；
- MV checkpoint 路径或 Hugging Face repo id；
- 新的 joint 输出目录；
- `gs_feature_lr=0.0075`：默认沿用 Ref 的 feature 学习率量级；
- `gs_patch_lr=5e-5`：默认沿用 MVInverse 训练的 optimizer 学习率量级；
- `neighbor_k=8`、`N=2`：作为首跑默认值，均可由训练命令覆盖；
- MV albedo 首轮直接传入 Ref shader，不做隐式 sRGB→linear；若异常，后续另立对照实验；
- top-K 首轮只用相机中心距离；若出现近位置、反方向相机，再新增方向约束实验，不改变当前采样接口。

上述参数必须保留在训练命令中；joint checkpoint 只保存恢复张量所必需的 iteration、`C` 和 Gaussian 数量。除这些数值外，当前讨论已冻结；如需改变模型结构、loss、冻结范围、renderer 公式或保存语义，应先新增决策记录并更新本文件。
