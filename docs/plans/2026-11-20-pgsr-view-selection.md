# 参考 PGSR 的联合训练视角选择

2026-11-20 确认采用 PGSR 的角度筛选和距离排序规则，作为 D-015 中纯相机中心距离排序的后续修订。当前待实施。

- 从相机外参取得世界坐标系下的相机中心和前向光轴。初始化采样器时计算两两光轴夹角与中心距离，排除自身，保留夹角小于 30° 的候选，再按中心距离升序取至多 K 个。PGSR 距离相同时优先选夹角更小的候选。
- PGSR 另有最小、最大中心距离筛选。第一版沿用 Ref-MV 现有的距离排序，暂不引入依赖场景尺度的距离阈值。
- 保留 `neighbor_k=8`，可通过现有参数改为 4 或 1。双视角训练每轮随机取一个 anchor，再从其候选中抽一个邻居。有效候选少于 K 时按实有数量采样。
- 实现建议在初始化时报告候选不足以满足 `num_views - 1` 的 anchor，避免训练途中才发现无法组成指定大小的视角组。仅调整 `ref_mv/joint.py` 的邻居表与采样校验。
- 验证朝向相反的近相机被过滤、合格相机按距离排序，以及 K 为 1、4、8 和候选不足时的行为。

参考 [PGSR 邻居选择](https://github.com/zju3dv/PGSR/blob/main/scene/__init__.py)、[默认参数](https://github.com/zju3dv/PGSR/blob/main/arguments/__init__.py)和[训练采样](https://github.com/zju3dv/PGSR/blob/main/train.py)。
