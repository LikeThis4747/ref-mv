# Ref-MV 首次联合训练性能与显存分析

> 本报告只基于运行中的只读状态和源码检查。没有暂停训练、附加 profiler、执行测试或修改训练代码。性能收益均为预期，未经单独实验验证。

## 结论

本次训练没有卡在数据读取或 CPU。第 3001 轮从 Ref-Gaussian 初始化阶段切换到联合分支后，每轮工作量从一次单视角渲染增加为：两次 Gaussian feature 渲染、一次双视角完整 MVInverse 前向/反向、两次材质 Gaussian 渲染。当前约 2.28 秒/轮和约 35.4 GiB 显存主要来自这条联合计算图，而不是 `neighbor_k` 或学习率配置。

当前实现可以训练，但“每轮、双视角、接近 1K 分辨率地运行完整 MVInverse”不适合作为高效训练配置。应先做不改变现有 loss 的冗余计算裁剪；如果仍不够，再讨论 MV 分支分辨率。不要直接通过 `num_views=1` 回避问题，因为这会去掉方案的多视角核心。

## 运行时观察

- 训练进程：`python train.py ... --init_until_iter 3000 ... -r 4`
- 观察位置：约第 3040 轮。
- GPU 0：约 98% 利用率，约 35.4 GiB 已用显存，功耗约 281–290 W。
- CPU：训练进程约占用 1.8 个 CPU 核；GPU 持续满载，因此不是明显的数据加载瓶颈。
- 输入尺寸：原图经 `-r 4` 后为 `988×742`，再裁为 patch size 14 可整除的 `980×742`。
- 双视角 token 数：每张图 `70×53=3710` 个 patch token；加 register token 后，跨视角 decoder 层处理约 7430 个 token。
- MVInverse checkpoint 文件约 4.06 GB；模型以 FP32 权重加载到 GPU，计算阶段使用 FP16 autocast。
- GPU 温度约 87°C，已出现软件温控降频；核心约 1695 MHz，低于最高 2100 MHz。这会造成次要损失，但无法解释主要的数量级差异。

## 已确认的主要开销

### 1. 每轮运行完整 MVInverse decoder，并对输入条件反向传播

`RefMVModel` 将 MVInverse 参数设为 `requires_grad_(False)`，但 Gaussian feature 经 `gs_patch_embed` 后加到 DINO token 上。为了把材质 loss 的梯度传回 `gs_patch_embed` 和 Gaussian feature，autograd 仍必须保存并反向经过 MVInverse 的 36 层 decoder 和相关输出 head。

因此，“冻结 MVInverse”只省掉其参数梯度和优化器状态，不会省掉 decoder 前向、输入梯度计算及主要 activation 显存。

代码位置：`ref_mv/joint.py::RefMVModel.forward`、`third_party/mvinverse/mvinverse/models/mvinverse.py::MVInverse.decode`。

### 2. `num_views=2` 时每轮有四次 Gaussian 渲染

联合分支先对两台相机分别渲染 Gaussian feature，再运行 MVInverse，最后使用预测材质对两台相机分别进行材质渲染。当前 loss 在两个视角上取平均，因此四份渲染相关计算图会共同参与一次 backward。

相比 Ref-Gaussian 原本每轮的一台相机、一次渲染，这部分计算和显存都显著增加。

代码位置：`ref_mv/joint.py` 中两个遍历 `cameras` 的循环。

### 3. 当前分辨率使 decoder 和输出 head 很重

当前 MVInverse 输入为 `980×742`。这没有超过上游推理脚本“最长边不超过 1024”的限制，但上游训练配置的 `img_size` 是 518。当前分辨率接近其训练尺寸的两倍长边，patch token 和高分辨率输出 head 的 activation 明显更大。

这里不能简单断言应该改为 518：Ref-Gaussian 的 gardenspheres 基线使用 `-r 4`，而 MV 分支若使用另一分辨率，需要明确材质图如何回到 Ref 渲染尺寸。这是集成设计选择，不应未经确认直接修改。

### 4. 计算了当前 loss 不使用的输出 head

MVInverse 每轮都会计算 `albedo`、`roughness`、`metallic`、`normal`、`shading`。当前联合渲染只读取前三项，`normal` 和 `shading` 没有进入 loss 或 renderer，但它们的完整前向和中间 activation 仍然产生。

代码事实：`ref_mv/joint.py` 构造 `material_maps` 时只读取 `albedo/metallic/roughness`。

### 5. 固定图像被重复编码

DINO 和 ResNeXt 输入是相机的固定 GT 图像，MVInverse 处于 eval 模式且参数冻结。相同相机每次被采样时，图像 encoder 仍会重新运行。其结果理论上可以复用，但 151 张图的多尺度 ResNeXt feature 很大；直接全部常驻 GPU 会恶化显存，简单的全量 cache 并不合适。

## 显存构成判断

当前约 35.4 GiB 不是由新增的标量 Gaussian feature 或 `gs_patch_embed` 参数造成的：前者对约 73.7 万个 Gaussian 只增加一个 feature channel，后者只是一个 `1→1024`、kernel/stride 均为 14 的卷积。

主要显存来源按量级判断为：

1. 双视角、36 层 decoder 及材质 head 为输入梯度保存的 activation；
2. 两次 feature 渲染与两次材质渲染同时保留的 Gaussian 计算图；
3. 约 4 GB 的 MVInverse FP32 权重，以及 autocast 计算期间的低精度张量；
4. Ref-Gaussian 自身约 73.7 万点的参数、梯度和 Adam 状态。

这是代码结构推断，不是显存 profiler 的逐算子归因；为了不干扰本次训练，没有附加 profiler。

## 建议优化（按优先级）

### 优先级 1：只执行当前训练需要的三个材质 head

让 MVInverse 保持默认完整输出不变，只为联合入口增加可选输出选择；联合训练请求 `albedo/roughness/metallic`，跳过 `normal/shading`。

- 正确性影响：不改变当前 renderer 输入和 loss，因为两项输出本来就未被使用。
- 预期收益：降低两个 DPT head 的前向时间和 activation 峰值；不会消除 36 层 decoder，因此不能期待单独恢复到 Ref 基线速度。
- 改动量：小，且可以保持 MVInverse 上游默认调用行为不变。
- 建议：本次训练结束后首先实施。

### 优先级 2：为 MV 分支明确较低的独立分辨率

Ref-Gaussian 继续沿用 gardenspheres 基线 `-r 4`，仅将进入 MVInverse 的图像和 Gaussian feature 对齐到接近上游训练配置的尺寸（例如长边 518 附近且能被 14 整除），再把预测材质可靠地对齐回 Ref 渲染尺寸。

- 正确性影响：会改变 MV 条件和材质图的空间分辨率，是新增集成选择；必须先确认缩放、相机内参和回采样方式。
- 预期收益：这是同时降低 decoder、输出 head 计算量和显存最有潜力的方向。
- 改动量：中等，不能只随手 resize；需要保证 feature、图像、材质和相机投影一致。
- 建议：先形成一个最小尺寸对齐方案并讨论，确认后再实现。

### 优先级 3：拆分并复用冻结的图像 encoder 结果

将固定 GT 图像的 DINO patch feature 与 ResNeXt feature 从每轮 decoder 调用中拆出，在相机再次出现时复用。

- 正确性影响：eval 模式、无数据增强时理论上等价。
- 预期收益：减少重复 encoder 前向时间；decoder 及其反向仍然存在，因此不能解决全部耗时。
- 显存风险：不能把全部相机的多尺度 feature 常驻 GPU。需要 CPU cache、有限 LRU 或只缓存 DINO token，都会增加传输和实现复杂度。
- 改动量：中等偏大。
- 建议：在前两项完成并测量后仍有明确瓶颈时再做，避免过度设计。

## 不建议当前优先做的事项

- `num_views=1`：会显著减负，但直接移除多视角联合的核心，不应作为默认优化。
- 降低联合训练频率：能降低平均耗时，却不能降低联合 step 的峰值显存，并且改变优化目标的采样权重。
- decoder activation checkpoint：可以换取显存，但会增加重计算，使当前已经过慢的问题更严重；除非仍然 OOM，否则不优先。
- 将完整 MVInverse 强制 `.half()`：可能减少权重显存，但偏离其上游 FP32 权重加 autocast 的执行方式，数值稳定性需要验证。
- 调整 `neighbor_k`、`gs_feature_lr` 或 `gs_patch_lr`：前者主要影响一次性的邻居候选表，后两者只影响优化数值，不是当前速度或显存瓶颈。

## 推荐执行顺序

1. 保留当前训练作为首个完整运行证据，不在运行中改代码。
2. 训练结束后实施“只计算三个已使用 head”的最小修改。
3. 用相同场景和参数观察单轮时间与峰值显存；这是性能对比，不需要为此建设自动化测试框架。
4. 如果仍不可接受，再讨论并冻结 MV 独立分辨率方案；未确认前不实现。
5. encoder cache 只在前两项仍不足时考虑。

运行环境方面可改善 GPU 散热，使其低于目标温度 84°C；这不会解决主要结构开销，但能避免额外降频。
