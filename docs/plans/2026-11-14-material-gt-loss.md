# 材质 GT loss 实施方案

日期：2026-11-14

## 冻结范围

- 数据从 `--source_path` 下的 `gt/<split>/<name>/<frame>.*` 读取，不增加绝对路径。
- 支持 albedo、metallic（对应 Ref `refl_strength`）、roughness、camera-space normal 和 camera-space Z depth。
- GT 只作为附加监督，不替换 renderer 当前材质；材质回归沿用 MVInverse 的 MSE，normal 沿用 cosine loss，depth 使用 L1。
- 每项由独立开关启用并带可调权重；启用后缺少目录或任一训练帧 GT 时立即报错。
- 额外可视化按 GT/当前值成对排列，固定每行四列。
- 不修改原始上游目录，不改 PBR 公式、optimizer 结构或 checkpoint 格式。

## 验证边界

本轮按用户要求只做 Python 语法检查，不运行 GPU、数据前向或训练验证。
