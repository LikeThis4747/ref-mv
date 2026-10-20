# Ref-Gaussian × MVInverse 方向讨论

状态：已于 2026-10-20 收敛并冻结。正式实现以 `docs/IMPLEMENTATION_PLAN.md` 和 `docs/DECISIONS.md` 为准；本文件保留讨论背景，不再作为未决方案继续扩展。  
更新日期：2026-10-20

## 2026-10-19 用户补充原话

> 方案再修改一下。现在暂时还不能冻结
>
> 1. 关于冻结ref-gaussian，我觉得可以暂时先不做。我目前假定训练流程是首先完整训练一遍ref-gaussian保存checkpoint，后续直接从rendersurfel开始加载check point联合训练。第二种可能的流程是直接render_initial之后，render_surfel就开始联合训练（这两个方案怎么兼容后续详细讨论一下，我可能都想尝试一下）
> 2. 关于视角选取，一次选取N张（1,2,4等）训练算平均loss没问题，但要选相关的视角，比如提前按相机参数给视角分组，或者先选一个视角然后选最近视角，我们要综合计算量来考虑一下实现。

这次补充重新打开两个此前暂定的问题：联合阶段 Ref-Gaussian 的训练范围与启动时机，
以及多视角相机的相关性采样。后续方案不能继续默认“Ref 全冻结”或“完全随机无放回
选取 N 个视角”。

2026-10-20 进一步确认：初版不为冻结或 topology 设计额外模式，直接沿用 Ref 原
训练流程。原流程继续 densify/prune，新增 feature 仅接入已有的
clone/split/prune；这属于维持 per-Gaussian 对齐所必需的机械修改，不引入第二套
拓扑方法。相关视角使用随机 anchor 加预计算 top-K 邻居表，表中只保存相机索引。

## 1. 核心研究想法

原始想法来自 FIPT：FIPT 通过三维位置 `x` 查询材质场得到 A/M/R；候选改动是增加一个独立的 feature MLP，得到由三维位置索引的 latent feature，再投影到各相机的二维空间，用它修正 MVInverse 的 image tokens。MVInverse 与 renderer 保持冻结，通过 RGB loss 将梯度传回 feature MLP。

切换到 Ref-Gaussian 后，核心动机不变，只把连续的 `x → feature MLP` 换成每个 Gaussian 携带的可训练 feature：

```text
per-Gaussian latent feature
  → 随相机可微 splat 为二维 feature map
  → 对齐并映射为 MVInverse patch-token 条件
  → 冻结的 MVInverse 输出逐视角 A/M/R
  → 冻结的 Ref-Gaussian surfel renderer 输出 RGB
  → RGB loss
  → 梯度回到 Gaussian feature 和必要的 token 映射层
```

新增 feature 不是 A/M/R，也不是额外材质标签，而是共享于多视角的三维场景条件。研究假设是：相比修改 MVInverse 内部权重的 LoRA，从输入端学习一个受三维场景约束的条件，可能更容易保留 MVInverse 原本预测较好的部分。这个假设仍需实验验证。

## 2. 当前已核对的事实

### 2.1 MVInverse token 路径

- DINOv2 ViT-L/14 将每个输入视角编码成 patch tokens，token 维度为 1024。
- tokens 随后进入交替的单视角/跨视角 attention decoder，再由不同 DPT heads 输出 A/M/R、normal 和 shading。
- 当前代码没有外部条件接口。用户提出的研究目标是修改图中 DINOv2 产生的 image tokens；最终确认的注入位置是 DINOv2 最终 `x_norm_patchtokens` 之后、MVInverse 交替 attention decoder 之前。该注入不是基线已有功能。
- 冻结 MVInverse 参数不能对完整前向使用 `no_grad()`；RGB loss 到输入条件的梯度仍须穿过冻结网络。

### 2.2 Ref-Gaussian 的训练与 deferred renderer

Ref-Gaussian 存在 `render_initial`、`render_volume` 和 `render_surfel`。实际 real-scene 命令（如 garden）采用：

```text
render_initial → render_surfel
```

其中 `render_surfel` 确实属于 deferred 形式：先将 Gaussian 属性 alpha-blend/splat 为二维属性图，再在像素级计算 shading。`render_volume` 则先在每个 Gaussian 上着色，再 splat 已着色的颜色。

因此，MVInverse 输出的二维材质图与 surfel 的像素级 shading 接口在结构上更接近。当前优先保留用户已有经验中效果较好的 `initial + surfel` 基线，不把引入 `render_volume` 或重写 diffuse 作为首轮必要改动。

### 2.3 Gaussian feature 的 splat 语义

对一个像素 `p`，Gaussian 按照从相机近到远的顺序混合。第 `i` 个 Gaussian 的最终权重为：

```text
w_i(p) = T_i(p) · alpha_i(p)
T_i(p) = product_{j<i} (1-alpha_j(p))
```

其中 `alpha_i(p)` 是该 Gaussian 的投影椭圆在像素处的透明度贡献，`T_i(p)` 是光线到达它之前尚未被前方 Gaussian 遮挡的比例。若每个 Gaussian 携带 `C` 维 feature `f_i`，屏幕空间 feature 为：

```text
F_2d(p) = sum_i w_i(p) · f_i
```

rasterizer 不理解 feature 的物理语义，只对每个通道执行相同的投影、深度排序和 alpha blending。当前 surfel 已经通过这条通用 feature 路径传递 `refl_strength(1) + roughness(1) + ori_color(3) + indirect(3)`，合计 8 个通道；CUDA 上限为 24，因此新增 feature 最多仍有 16 个通道的预算。

这证明新增 Gaussian latent 在机制上可行，而且 backward 能返回对其的梯度。但“可以 splat”不代表任何属性都能被物理正确地线性混合：rasterizer 可以传递 latent，却不保证这种 latent 混合一定最优；roughness 等非线性 BRDF 参数先混合再 shading 仍是 deferred approximation。

### 2.4 两个模型的材质语义

在论文定义上，MVInverse 和 Ref-Gaussian 都采用 metallic workflow：

```text
diffuse coefficient = A(1-M)
F0 = 0.04(1-M) + A·M
R 控制 GGX specular lobe
```

当前 Ref-Gaussian 代码的名称容易误导：

- `_metalness` 被保存和优化器登记，但没有进入活跃的 `render_volume`/`render_surfel` 公式。
- `refl_strength` 实际承担 metallic `M` 的数学作用。
- `ori_color`/`albedo` 进入 specular Fresnel 的 `F0`。

因此当前接口关系可暂按下式理解：

```text
MV A ↔ Ref ori_color/albedo
MV M ↔ Ref refl_strength
MV R ↔ Ref roughness
```

### 2.5 Volume 与 surfel 的 diffuse 差异

`render_volume` 使用较标准的形式：

```text
Cdiff = Ldiff(N) · A(1-M)
Cspec = Lspec(N,V,R) · split_sum(A,M,R)
```

当前 `render_surfel` 使用：

```text
C = (1-M) · C_SH + Cspec(A,M,R)
```

这里 `C_SH` 是 Gaussian SH rasterization 的外观颜色。Ref-Gaussian 论文第 7 页 implementation details 明确说明：作者发现 SH 比 integrated diffuse lighting 拟合能力更强，因此用 SH 替代后者。这是实现选择，不是 deferred shading 的必要条件。

代价是 `C_SH` 可以吸收 albedo、diffuse lighting、阴影和其他未建模外观，导致单独输出的 `base_color_map` 缺少完整 diffuse 约束，视觉上可能不像可解释的 albedo。论文也没有用材质 GT 证明该图具有严格可辨识性。

MaterialRefGS 仍保留相同的 surfel 主公式；它主要增加多视角一致性、法线/反射先验和间接光处理，没有改成标准的 `A(1-M)Ldiff` diffuse。

分析判断（不是论文或代码自述）：Ref-Gaussian 的 `indirect` 更接近提高反射重建质量的可学习补偿项，而不是严格求解的物理间接光。代码中它是每个 Gaussian 上学习的、随反射方向查询的 3 通道 SH；没有显式计算多次反射，并且对粗糙表面仍主要使用单一反射方向和二值 visibility，不能完整表示 roughness-dependent 的入射光积分。联合实验保留该公式；`indirect` 参数是否继续训练取决于重新讨论后的 Ref 训练范围。

## 3. 当前首轮联合实验方案（连接接口已确认，训练日程重新讨论）

首轮目标是以尽量小的改动跑通完整梯度链，而不是同时修复 Ref-Gaussian 的材质分解：

1. 需要兼容两种启动方式：完整训练 `render_initial + render_surfel` 后加载 saved iteration；或完成 `render_initial` 后直接进入联合 surfel 阶段。
2. 初版不额外冻结 Ref-Gaussian，继续使用原 regularization、optimizer 和 densification/pruning；新增 feature 只接入已有 topology 更新路径。
3. 为每个 Gaussian 增加独立显式 latent feature；不复用 SH 或 A/M/R 属性充当条件。首轮 `C=1`，随后再比较 `C=3`。
4. 将 latent feature 可微 splat 为每个 MVInverse 输入视角的二维 feature map。
5. 使用 `Conv2d(C,1024,kernel_size=14,stride=14,bias=False)` 转为 patch tokens，在 DINO 最终 `hidden_patches` 后直接残差相加；Gaussian feature 零初始化。
6. 冻结 MVInverse 主体和 heads，得到逐视角 A/M/R。
7. 在联合路径中，用 MVInverse 的 A/M/R 替换 surfel shader 接收的二维 `albedo/refl_strength/roughness`，但首轮保持原 surfel 公式不变：

```text
C = (1-M_mv) · C_SH + Cspec(A_mv, M_mv, R_mv)
```

8. RGB loss 通过 renderer 和冻结 MVInverse 回传；初版不额外冻结 Ref 参数，继续沿用其原 optimizer、stage schedule 与 densification/pruning。

这条链路不应被描述为严格完成了物理一致的 A/M/R 替换：MV 的 A 在当前公式中主要参与 specular `F0`，没有进入标准 diffuse 项。它的定位是最小连接实验，用于验证三维条件能否稳定影响冻结 MVInverse 并改善渲染。

未来可以单独研究将 surfel diffuse 改为 `A(1-M)Ldiff`；这会涉及 alpha-premultiplied 属性归一化、环境光、颜色空间和基线重训，不属于当前最小改动范围。

## 4. MVInverse 基线接口与 Token 融合决定

### 4.1 基线代码已有的处理

MVInverse 的 RGB 输入先进入 DINOv2 ViT-L/14。DINO 的基线 patch embedding 是：

```text
Conv2d(in_channels=3, out_channels=1024, kernel_size=14, stride=14)
```

它直接把每个 `14×14` RGB 区域转换成一个 1024 维初始 patch embedding；经过 DINO 内部 transformer 后，MVInverse 读取最终的 `x_norm_patchtokens`，再送入自己的 `decode()`。MVInverse 基线不存在 Gaussian feature 分支，也不存在 add、multiply、gate 或 cross-attention 等外部条件融合算子。

### 4.2 屏幕空间 feature 的 patch 化（本项目新增，已确认）

为了尽量复用基线处理二维像素图的方式，首轮为屏幕空间 Gaussian feature 设置平行的 patch embedding：

```text
Conv2d(in_channels=C, out_channels=1024, kernel_size=14, stride=14)
```

这只是沿用基线的 patch 大小、步长和输出维度；“增加平行分支”本身仍是本项目的新设计。

### 4.3 注入位置比较（均非基线已有，方案 A 已确认）

**方案 A：DINO 后注入。** RGB 独立走完冻结的 DINO，得到最终 `hidden_patches`；二维 GS feature 只经过平行 patch embedding，然后在 MVInverse `decode()` 之前与 `hidden_patches` 融合：

```text
RGB → DINO 24 层 → T_image ─────┐
                                 融合 → MVInverse decode
GS feature → GS patch embed → T_GS ┘
```

优点是完全保留 DINO 的原始 RGB 编码过程、改动位置清楚，并且 DINO 可在 `no_grad()` 下运行以节省显存；缺点是 GS token 不经过 DINO 的 24 层单图 attention。该位置最接近最初“修改图中 DINO 输出的 image tokens”的表述，已确认为首轮方案。

**方案 B：DINO 前注入。** RGB 与 GS feature 分别经过 patch embedding，在初始 patch embedding 处融合，再共同经过冻结的 DINO 24 层：

```text
RGB → RGB patch embed ──┐
                        融合 → DINO 24 层 → hidden_patches
GS feature → GS patch embed ┘
```

这让 GS feature 走完整 DINO 流程，但会改变冻结 DINO 从第一层开始接收的 token 分布；训练时也必须保留 DINO 24 层的反向计算图。当前认为它比方案 A 风险更大，暂作为后续对照，不作为首轮倾向。

### 4.4 首轮已确认的融合算子

首轮确认采用方案 A 的直接残差加法：

```text
T_conditioned = T_image + T_GS
```

具体使用 `Conv2d(C,1024,kernel_size=14,stride=14,bias=False)`，Gaussian feature 零初始化，卷积使用非零默认权重初始化。首轮不增加 gate、额外 LayerNorm、MLP、乘法、拼接或 cross-attention。实现时监控 GS tokens 与 DINO tokens 的 RMS 比值，但不预先增加新的尺度 loss。

Gaussian feature 维度采用逐步实验：

```text
C=1：先验证完整数据流、梯度链和标量条件是否已经有效
C=3：流程成立后增加条件自由度
```

暂不考虑 `C=8`。这里的 `C=1` 表示每个 Gaussian、以及 splat 后每个像素只有一个 feature 通道；经过与基线一致的 `14×14` patch convolution 后，每个 patch 仍可得到 1024 维 token，因此不能把它误解成“每个 patch 最终只有一个标量”。feature 维度应作为配置项，而不是写死。

## 5. 已确认接口与剩余风险

### 已确认：Gaussian feature 与 token 条件

- feature 确认为每个 Gaussian 的显式向量，不增加小网络。
- feature 维度首轮按 `C=1 → C=3` 进行，不考虑 8 维。
- 已确认首轮直接使用 rasterizer 的 alpha-premultiplied feature，不除累计 alpha、不额外拼 alpha；背景因无覆盖自然为零。
- 已确认方案 A（DINO 后注入）、平行 feature patch embedding 和直接残差相加；实现时仍需核对精确张量接口。
- 已确认复用 Ref 的 `-r` 缩放，并在相机加载阶段从右/下最小裁切到 14 的倍数；保留裁后像素坐标不变的 `K,R,T`，同步裁 RGB/mask，按裁后尺寸重算 FoV 与投影矩阵。
- 实现时验证零 feature 是否严格复现原 MVInverse，并监控 token 尺度与梯度；若失败才重新讨论尺度控制，而不是预先增加新模块。

### 部分已确认：联合阶段的数据流；Ref 训练边界重新讨论

- MVInverse 保持冻结；Ref 初版继续按原流程训练。Gaussian latent 加入 Ref 原 optimizer，GS patch convolution 使用联合代码管理的 optimizer。
- MV A/M/R 直接对应 `albedo/refl_strength/roughness`。初版不拆 renderer：给原 `render_surfel()` 增加可选 feature 返回和材质覆盖入口，每个联合视角调用两次，保留默认行为和原 surfel 公式；只有实测成为瓶颈后才考虑拆分。
- 每轮仍使用小规模 `N` 个视角、一次 MVInverse 前向，并对全部视角分别 Ref render 后平均 RGB loss；但 `N>1` 时改为 anchor 加相关邻域采样，不再采用完全随机视角。仍先 `N=1` smoke test，再用 `N=2` 验证跨视角，正式 `N` 根据显存决定。
- 联合训练使用的 RGB、相机、裁剪/缩放和背景必须与 feature splat 及 MV 输出逐像素一致。
- photometric 主项复用 Ref 的 L1 与 DSSIM，颜色空间、背景和 `srgb` 配置跟随 Ref；已确认继续服从原 normal/depth/distortion regularization 的配置与启用条件。

### 剩余研究风险：监督与材质退化

仅用 RGB loss 无法唯一确定 A/M/R：A 与光照、M 与 Fresnel、R 与环境高频/法线都可互相补偿。MV 虽然冻结，Gaussian condition 仍可能把输出推向“RGB 更好、材质更差”；若 Ref 同时训练，SH、环境光或几何还可能进一步吸收误差，使该歧义更强。

首轮先不增加新 loss，但必须记录：

- 如何监控 token 修正幅度、A/M/R 变化量和各通道梯度。
- 除训练视角 RGB 外，使用哪些 holdout、多视角一致性或材质指标判断改善，而不是只看重建 loss。

若 RGB-only 实验出现明显材质漂移，再讨论 conditioned A/M/R 对原始 MV 输出的弱 anchor；它不是首轮默认项。

### 已发现的加载风险

Ref 的训练 checkpoint `capture()` 不包含环境贴图和 `ray_tracer/mesh`。首轮联合训练应复用 `eval.py` 的已保存 iteration 加载方式：读取 PLY 和环境贴图；若启用 indirect，还要读取同 iteration 的 `test_XXXXXX.ply` 重建 ray tracer。不能只加载 `chkpnt*.pth` 后假定 surfel indirect 状态完整。

首版还应保持原 PLY 和 `capture()/restore()` 格式不变：完整 saved iteration 加载并执行原 `training_setup()` 后，再把 `gs_feature` 加入 optimizer；joint checkpoint 单独保存 feature 与 patch mapping。多视角 densification 只统计第二次正常 surfel 渲染的输出，不能把第一次 feature-only 渲染重复计入；同时应抵消平均 loss 带来的 `1/N` 屏幕空间梯度缩放，以保持原 densification 阈值含义。

## 6. 当前边界

- 首轮接口已经形成可实现方案，并于 2026-10-20 冻结；下一步进入正式编码。
- 不修改 `/home/disk9/wsq/ENVS/mvinverse` 或 `/home/disk9/wsq/ENVS/ref-gaussian` 原始基线；必要接口修改放在 `third_party/` 开发副本，联合逻辑放在 `ref_mv/`。
- 先完成回归与梯度 smoke test，再启动长训练。
- 不把旧 FIPT/V6 LoRA 结果直接当作 Ref+MV 方案证据，但保留其“RGB 改善不等于材质改善”的经验。
- 讨论文档只在形成阶段性稳定理解时整体整理，不逐轮追加聊天内容。
- 详细的文件级接口、联合前向伪代码、加载/保存与验收顺序已收敛到
  `docs/IMPLEMENTATION_PLAN.md`；本文件继续保留研究动机、代码事实和风险分析，
  不重复维护实现细节。
