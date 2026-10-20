# 已确认决定

这里只记录已经确认的项目决定。讨论中的候选方案放在 `docs/discussions/`。

## D-001：独立工作区

- 日期：2026-10-14
- 状态：已确认
- Ref+MV 使用独立的 `ref-mv/` 目录；旧 `claude/` 只作为 FIPT/MVInverse 背景资料，不自动继承其 V6 主线。

## D-002：上游基线只读

- 日期：2026-10-14
- 状态：已确认
- `/home/disk9/wsq/ENVS/mvinverse` 和 `/home/disk9/wsq/ENVS/ref-gaussian` 作为原始基线保留且不修改；开发使用其位于 `ref-mv/third_party/` 的副本，联合逻辑放在 `ref-mv/ref_mv/`。

## D-003：轻量文档结构

- 日期：2026-10-14
- 状态：已确认
- 项目文档统一放在 `docs/` 的对应位置，不在根目录复制多套说明；根目录只保留必要入口文件以及已确认的 `ref_mv/`、`third_party/`、`configs/`、`scripts/`、`tests/` 代码结构。

## D-004：先讨论和验证，再实现

- 日期：2026-10-14
- 状态：已确认
- 在 feature 投影、token 注入、冻结边界、renderer 和 RGB loss 尚未明确前，不把候选闭环写成正式架构，也不启动长实验。

## D-005：记忆按短期/长期分层

- 日期：2026-10-14
- 状态：已确认
- 记忆放在 `docs/memory/`，短期和长期分开，每条带日期；过期内容进入 `docs/memory/archive/`。不把记忆系统作为第二套项目真相。

## D-006：基线优先与新增方法显式标注

- 日期：2026-10-15
- 状态：已确认
- 任何接口、特征变换、融合、loss、初始化和训练设计，都必须先核对 MVInverse 或 Ref-Gaussian 的已有实现。
- 基线已有做法时默认优先复用；偏离基线的内容必须明确标注为“新增候选（非基线已有）”，说明依据和额外风险，并在用户确认前保持候选状态。
- 不允许把助手自行提出的结构无标注地混入基线事实或写成默认方案。

## D-007：首轮采用 DINO 后直接残差注入

- 日期：2026-10-16
- 状态：已确认
- 首轮采用方案 A：RGB 走完冻结 DINO 后，在最终 `hidden_patches` 与 MVInverse `decode()` 之间注入 GS tokens。
- 屏幕空间 Gaussian feature 使用 `Conv2d(C, 1024, kernel_size=14, stride=14, bias=False)` 转成 patch tokens；首轮 `C=1`。
- Gaussian feature 零初始化，GS patch convolution 使用非零默认权重初始化，GS tokens 与 `hidden_patches` 直接相加。
- 首轮不增加 gate、额外 LayerNorm、MLP、乘法、拼接或 cross-attention。

## D-008：首轮使用原始 alpha-premultiplied Gaussian feature

- 日期：2026-10-16
- 状态：已确认
- 新增 Gaussian feature 直接使用 Ref rasterizer 返回的 `rendered_feature = sum_i(T_i * alpha_i * f_i)`。
- 首轮不除以累计 `render_alpha`，也不额外拼接 alpha 通道或增加低-alpha阈值/mask。
- 这沿用 Ref-Gaussian 当前通用 feature 的输出方式；若后续确认边缘或低覆盖区域存在问题，再单独比较 alpha normalization。

## D-009：首轮采用随机小规模多视角、全部视角 RGB 监督

- 日期：2026-10-16
- 状态：部分撤回（2026-10-19）
- 仍保留：每轮使用 `N` 个训练相机，分别 splat feature，一次送入 MVInverse，并对全部 `N` 个输出视角执行 Ref surfel rendering；各视角 RGB loss 求平均后统一反传。
- 先用 `N=1` 做完整链路 smoke test，再用 `N=2` 验证跨视角路径；正式 `N` 根据显存测试决定。
- 已撤回：相机不再默认完全随机选取。最终由 D-015 确认为先随机选 anchor，再从按相机中心距离预计算的 top-K 邻域中选择其余视角。

## D-010：复用 Ref 的 `-r` 缩放并最小裁切到 14 的倍数

- 日期：2026-10-16
- 状态：已确认
- 不预处理或复制数据集，不重跑 COLMAP，也不建立第二套 joint camera；继续使用 Ref-Gaussian 原有 `-r` 分辨率缩放和内参缩放流程。
- 在 `-r` 完成 resize 后，只从图像右侧和下侧裁掉不足 14 倍数的部分：`W14=(W//14)*14`、`H14=(H//14)*14`，每个方向最多损失 13 像素。
- 因裁切不移动左上角坐标原点，已缩放的 `K` 以及 `R/T` 保持不变；RGB、mask 等所有像素张量执行相同裁切。
- 必须使用裁后 `H14/W14` 与原 `K` 更新 `HWK`，由 `fx/fy` 和裁后尺寸重算 `FoVx/FoVy`，并重建 `projection_matrix/full_proj_transform`。不能只裁图像而保留旧投影参数。

## D-011：按 Ref 原计算边界拆出 rasterization 与 shading

- 日期：2026-10-16
- 状态：初版已撤回（2026-10-20）
- 原方案能让每个视角只 rasterize 一次，但需要较大范围移动 `render_surfel()` 代码。初版优先降低基线改动与回归风险，暂不拆函数。
- 初版只给原 `render_surfel()` 增加可选的 `gaussian_feature`、`material_maps` 和 `return_deferred`：第一次返回可微 `gs_feature`，第二次仍走原函数并在 shading 前覆盖 A/M/R。
- 代价是每个联合视角 rasterize 两次；先实测 `N=1/2/4` 的时间与显存，确有瓶颈后再考虑拆分优化。
- 原变量名 `base_color/albedo/refl_strength/roughness/specular/final_image` 继续保持不变。

## D-012：首轮直接用 MV A/M/R 替换 surfel 的同义二维入口

- 日期：2026-10-16
- 状态：已确认
- 联合 shading 使用 `MV albedo -> albedo`、`MV metallic -> refl_strength`、`MV roughness -> roughness`。该接口关系与原 surfel 公式继续保留；“其余 Ref 中间量冻结”不再视为已确认决定，联合阶段哪些 Ref 参数继续训练另行讨论。
- 继续使用原 surfel 公式 `final_image = (1-refl_strength) * base_color + specular`，不在首轮把 diffuse 改为 volume 形式。
- 因此 MV albedo 在首轮主要通过 specular `F0` 起作用；这是保留 Ref surfel 基线公式的已知限制，不把它误写为标准 diffuse albedo 已被完整接入。

## D-013：上游代码复用与改动标注

- 日期：2026-10-16
- 状态：已确认
- 原始上游目录继续只读。能直接 import/call 的 Ref/MV 类、函数和算子必须直接复用；没有所需插入点时，只在 `third_party/` 开发副本中做最小接口修改，联合编排放在 `ref_mv/`。
- 提取内容必须注明上游仓库、文件和基准 commit，并用注释区分“上游原代码”和“REF-MV 新增/修改”；不得自行重写 PBR 公式、相机数学或 MV heads。

## D-014：采用“项目包 + third_party 上游副本”目录结构

- 日期：2026-10-17
- 状态：已确认
- 根目录使用 `ref_mv/` 存放本项目联合代码，使用 `third_party/ref_gaussian/` 与 `third_party/mvinverse/` 存放可修改的上游开发副本；`configs/`、`scripts/`、`tests/` 分别存放实验配置、命令入口和回归测试。
- `third_party/UPSTREAMS.md` 记录两个副本的来源、基准 commit 和本地修改清单；原始 `/home/disk9/wsq/ENVS/ref-gaussian` 与 `/home/disk9/wsq/ENVS/mvinverse` 不做任何修改。
- 根目录 `train.py` 作为唯一正式联合训练入口，避免从不同上游子目录启动造成导入和相对路径混乱。

## D-015：重新打开 Ref 训练范围与相关视角采样

- 日期：2026-10-19
- 状态：初版原则已确认（2026-10-20）
- 初版不额外冻结 Ref-Gaussian，也不关闭或重写原 densification/pruning；尽量沿用 Ref 原训练循环、optimizer 和 stage schedule。
- 新增 Gaussian feature 只做必要的最小接入：随已有 Gaussian 一起 clone/split/prune，不能因新增 feature 改造出第二套 topology 流程。
- 训练入口仍需兼容两条流程：完整训练 Ref 后从 surfel saved iteration 启动联合训练；以及完成 `render_initial` 后直接进入联合 surfel 阶段。
- `N>1` 时采用随机 anchor 加预计算 top-K 相机邻居采样；邻居表只保存相机索引，不复制图像。全部视角 RGB loss 仍取平均。

## D-016：首版保持 Ref 存储格式和主训练循环

- 日期：2026-10-20
- 状态：已确认
- 首版不修改 Ref 的 PLY 字段以及 `capture()/restore()` tuple。加载完整 saved iteration、执行原 `training_setup()` 后，再以可选参数组加入 `gs_feature`；新增 feature 和 patch mapping 单独保存。
- joint surfel step 接入 Ref 原 `train.py` 循环，不另写一套 stage、loss、densification、mesh 更新和保存调度。
- 多视角反传后，只用第二次正常 `render_surfel()` 的各视角输出逐一累计原 densification 统计，再执行一次原 densify/prune；第一次仅提取 GS feature 的渲染不参与该统计。因平均 loss 会把每个视角的屏幕空间梯度缩小 `1/N`，统计前只将该临时梯度乘回 `N`，保持原阈值量纲，不改变参数的平均梯度。
- “加载 Ref checkpoint”在首版中指完整 saved iteration（PLY、环境贴图以及需要时的 mesh），不以单独 `chkpnt*.pth` 代替。当前上游 `restore()` 没有恢复 Adam state，首版不为续接 Adam 动量扩展其格式。

## D-017：冻结首轮实现方案并分离输入输出目录

- 日期：2026-10-20
- 状态：已确认
- `docs/IMPLEMENTATION_PLAN.md` 冻结为首轮实现依据；除 checkpoint 路径、输出目录、两个新增学习率、邻居 K 和正式 N 等运行参数外，不在编码阶段临时改变模型连接、loss、冻结范围或 renderer 公式。
- 支持 `saved-surfel` 与 `after-initial` 两种启动方式，均使用 Ref 的绝对迭代号和同一个 joint surfel step；后一方式在 initial 后跳过 volume stage，直接进入 joint surfel。
- Ref 输入 saved iteration 与 joint 输出目录必须分离，联合训练不得覆盖基线结果。验证和可视化在 joint 阶段也必须走 conditioned renderer。
- MVInverse 显式 `requires_grad_(False)` 且保持 `eval()`；Ref 不额外冻结，因此测试预期不能写成“只有新增参数获得梯度”。
- 自动化验证采用最小必要集并全部放入根目录 `tests/`，不得混入训练代码或实验脚本；只保护新增接口、上游默认行为、Gaussian topology 对齐和端到端梯度链。
- 短程 overfit、跨视角行为、显存和视觉质量统一称为“实验验证”，入口放在 `scripts/experiments/`，不由 pytest 启动。联合新增代码用 `holdout/validation` 表示训练外评估，不新增含混的 `test` 命名；上游已有命名仅为兼容保留。
