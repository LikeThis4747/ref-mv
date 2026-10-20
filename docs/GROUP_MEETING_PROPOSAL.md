# Ref-Gaussian × MVInverse 联合优化方案

## 1. 核心想法

MVInverse 已具备多视角材质预测能力，但直接用 LoRA 修改网络内部参数可能破坏其原有能力。我们的思路是：**冻结 MVInverse，通过 Ref-Gaussian 提供一个可学习、跨视角一致的三维条件，从输入 token 端引导材质预测。** 初版继续沿用 Ref-Gaussian 原训练流程。

每个 Gaussian 新增一个低维 feature。该 feature 随相机可微投影到二维，加入 MVInverse 的 image tokens；MVInverse 输出 A（albedo）、M（metallic）、R（roughness）后，再交给 Ref-Gaussian 渲染 RGB，用重建误差反向优化 Gaussian feature。

## 2. 整体流程

```text
多视角 RGB ─→ 冻结 DINO ─→ image tokens ───────────────┐
                                                        + → 冻结 MVInverse → A/M/R
Gaussian feature → 可微 splat → 2D feature → patch tokens ┘                    │
                                                                                ↓
GT RGB ←──────── RGB loss ←──────── Ref surfel shading ←────────────────────────┘
```

梯度路径为：

```text
RGB loss → Ref shading → MVInverse → patch mapping → 2D feature → Gaussian feature
```

MVInverse 保持冻结。Ref-Gaussian 不增加额外冻结逻辑，原 optimizer、loss 与 densification/pruning 继续运行；新增 feature 只接入其已有 clone/split/prune 路径。

## 3. 关键改动

| 位置 | 改动 | 首轮设置 |
|---|---|---|
| Gaussian | 每个 Gaussian 增加显式 feature | `C=1`，零初始化；后续比较 `C=3` |
| Rasterizer | 将 feature 与现有属性一起 splat | 使用原始 alpha-premultiplied 输出，不除 alpha |
| MVInverse 输入 | 将二维 feature 转为 patch tokens | `Conv2d(C,1024,k=14,s=14,bias=False)` |
| Token 融合 | 在 DINO 最终 `hidden_patches` 后注入 | `T = T_image + T_GS` |
| 材质接口 | MV 输出接入 Ref surfel shader | `A→albedo`，`M→refl_strength`，`R→roughness` |
| Renderer | 原 `render_surfel()` 增加可选 feature 返回与材质覆盖入口 | 初版调用两次，不拆函数、不重写 PBR |

这里需要区分“对齐基线”和“新增设计”：MVInverse 的 DINO 原本就用 `Conv2d(3,1024,k=14,s=14)` 将每个 `14×14` RGB patch 转成一个 1024 维 token；GS 分支复用相同的 patch 网格、步长、token 数量和维度。`Conv2d(C,1024,k=14,s=14,bias=False)` 以及 `T_image + T_GS` 则是本项目新增的条件接口，原 MVInverse 中不存在。注入位置选择在 DINO 最终 `hidden_patches` 之后，是为了保持原 RGB→DINO 路径完全不变。

Ref surfel 最终公式保持不变：

```text
final_image = (1 - refl_strength) * base_color + specular
```

其中 `base_color` 仍来自 Ref 的 SH 外观，MV 的 `albedo/refl_strength/roughness` 进入原 surfel 材质入口。首轮目标是先验证闭环，不同时修改 Ref 的 diffuse 定义。

## 4. 训练方式

1. 支持两种启动方式：完整训练 `initial + surfel` 后加载 saved iteration；或完成 `initial` 后直接进入 surfel 联合阶段。
2. Ref 继续按原流程训练和更新 topology；新增 feature 与 Gaussian 同步变化。
3. 冻结 MVInverse 的 DINO、跨视角 decoder 和所有预测 heads。
4. 每轮随机选择一个 anchor，再从其相关相机邻域选出其余 `N-1` 个视角；一次预测全部视角 A/M/R，再分别渲染 RGB。
5. 复用 Ref 的 `L1 + DSSIM` RGB loss，对 `N` 个视角取平均。

实验顺序：先用 `C=1,N=1` 跑通梯度闭环，再用 `N=2` 验证跨视角路径，之后根据显存决定正式视角数。

## 5. 最小改动原则

首轮不加入 LoRA、gate、额外 attention、材质 anchor、alpha normalization 或新的 PBR 公式。原始基线保留不动；开发副本放在 `third_party/`，联合逻辑放在 `ref_mv/`。所有修改注明上游来源和新增位置。

## 6. 验证与风险

实现首先通过三项检查：

- renderer 新增参数使用默认值时与原输出一致；
- Gaussian feature 为零时，MVInverse 输出与原模型一致；
- 反传后 Gaussian feature 和 patch mapping 获得预期梯度，MVInverse 参数无梯度；Ref 参数按原训练边界继续获得或不获得梯度。

主要研究风险是仅用 RGB loss 时 A/M/R 存在分解歧义，条件可能改善 RGB 却降低材质真实性。此外，当前 surfel 中 MV albedo 主要通过 specular `F0` 起作用，而不是完整替换 diffuse albedo。首轮先记录 A/M/R 变化、token 幅度及未训练视角表现；若出现明显材质漂移，再讨论弱材质约束或 diffuse 公式改进。
