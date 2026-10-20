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
