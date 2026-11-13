# Ref-MV Windows 迁移与 AI 交接说明

本文档面向第一次在 Windows 电脑上接手本项目的人或 AI 助手。目标是说明当前仓库是什么、哪些内容在 Git 中、如何完整拉取、如何重新配置环境，以及迁移后先验证什么。

## 1. 当前项目状态

项目是 Ref-Gaussian 与 MVInverse 的联合研究工作区，不是一个已经发布的独立 Python 包。当前联合路径为：

~~~
Gaussian feature
    → Ref Gaussian rasterization / feature map
    → MVInverse token 条件
    → albedo / metallic / roughness
    → Ref-Gaussian surfel rendering
    → RGB loss
~~~

当前远程版本由三个 Git 仓库组成：

| 仓库 | 远程地址 | 当前远程分支 |
|---|---|---|
| 主项目 | https://github.com/LikeThis4747/ref-mv.git | main |
| Ref-Gaussian 开发副本 | https://github.com/LikeThis4747/ref-gaussian.git | main |
| MVInverse 开发副本 | https://github.com/LikeThis4747/mvinverse.git | main |

主项目通过 Git submodule 固定后两个仓库的 commit。不要把主项目里的 third_party/ref_gaussian 或 third_party/mvinverse 当成普通目录单独复制后再随意替换版本。

当前联合方案的稳定事实：

- Gaussian feature 维度 C=1。
- 使用 14x14 patch convolution 将 Gaussian feature 转成 MVInverse 条件 token。
- 默认多视角数量 num_views=2，按相机中心距离选择邻居。
- MVInverse 参数冻结，但为了训练 Gaussian feature 和 patch embedding，仍需经过 MVInverse 的输入梯度路径。
- 联合路径只使用 albedo、metallic、roughness 三个输出。
- 默认正式流程是先完成 Ref-Gaussian iteration 20000，再从完整 saved iteration 启动 joint。
- Ref 输入 checkpoint 目录和 joint 输出目录必须分开。
- 当前性能优化代码已经提交，但优化后的实际单步时间和峰值显存还需要重新测量。

当前服务器环境实际检查结果：

~~~
Python       3.10.22
PyTorch      2.5.1+cu118
TorchVision  0.20.1+cu118
TorchAudio   2.5.1+cu118
CUDA runtime 11.8
~~~

这是目前最重要的环境参考。Ref-Gaussian 原始 requirements.txt 仍然写着 PyTorch 2.0.0，而 MVInverse 文档使用 PyTorch 2.5.1；在新机器上不要直接无检查地按两个 requirements 文件顺序安装，否则可能把已正确安装的 PyTorch 降级。

## 2. 仓库和目录结构

~~~
ref-mv/
├── train.py                         # Ref-MV 正式联合训练入口
├── ref_mv/
│   └── joint.py                     # Gaussian feature 与 MVInverse 的联合逻辑
├── third_party/
│   ├── ref_gaussian/                # Git submodule；当前使用 LikeThis4747 fork
│   │   ├── train.py                 # Ref 单独训练入口
│   │   ├── gaussian_renderer/       # rasterization / shading / render wrapper
│   │   ├── scene/                   # Gaussian、相机、数据读取
│   │   └── submodules/              # diff-surfel、cubemap、raytracing、simple-knn
│   └── mvinverse/                   # Git submodule；当前使用 LikeThis4747 fork
├── configs/README.md                # 训练参数和首跑配置说明
├── scripts/
│   ├── env.sh                       # Linux/WSL 环境变量入口
│   ├── check_environment.sh         # CUDA 扩展和 import 检查
│   └── visualize/                   # checkpoint 可视化工具
├── run_office_experiments.sh        # 多阶段实验示例，默认按双 GPU 编写
├── monitor_office_experiments.sh    # 实验监控示例
├── docs/                            # 状态、决定、方案、报告和本迁移文档
├── data/                            # 被 .gitignore 忽略，不在远程 Git 中
├── output/                          # 被 .gitignore 忽略，不在远程 Git 中
└── pyproject.toml                   # 主项目元数据，要求 Python >=3.10
~~~

data/nvdiffrast 在当前服务器上还存在一个独立的本地 Git 检出，但它不是主项目 submodule，也不会随 git clone 出现。新机器需要安装 nvdiffrast==0.3.1；如果使用源码安装，则另外准备该源码目录。

项目外的 /home/.../ENVS/mvinverse 和 /home/.../ENVS/ref-gaussian 是原始上游基线，当前设计要求它们只读；迁移到 Windows 不需要复制这两个目录。

## 3. 完整拉取项目

### 3.1 使用 SSH

~~~
git clone git@github.com:LikeThis4747/ref-mv.git
cd ref-mv
git submodule update --init --recursive
~~~

### 3.2 没有 GitHub SSH key 时使用 HTTPS

~~~
git clone https://github.com/LikeThis4747/ref-mv.git
cd ref-mv
git submodule update --init --recursive
~~~

验证三个仓库是否完整：

~~~
git status --short --branch
git submodule status
git -C third_party/ref_gaussian status --short --branch
git -C third_party/mvinverse status --short --branch
~~~

正常情况下，三个仓库都应位于 main，并且没有未提交修改。若 submodule 行首出现 -，说明尚未初始化；若出现 +，说明本地 submodule commit 与主项目记录不一致。

## 4. Windows 环境路线

### 4.1 原生 Windows

原生 Windows 不是禁止路线。当前 CUDA/C++ 源码中已经包含部分 Windows/MSVC 分支，理论上可以直接在 Windows 编译运行。需要准备：

- NVIDIA Windows 驱动；
- Visual Studio Build Tools，勾选 Desktop development with C++；
- CUDA Toolkit，并确保 nvcc 可用；
- Python 3.10；
- Conda 或 venv；
- Ninja、Git；
- 与 PyTorch wheel 相容的 MSVC/CUDA 编译链。

原生 Windows 需要自己处理 PowerShell 环境变量，当前 scripts/env.sh 不能直接在 CMD 或 PowerShell 中执行。训练 Python 代码本身并不依赖这些 Bash 语法；Bash 脚本只是调用入口。

### 4.2 Windows + WSL2

WSL2 是更接近服务器 Linux 环境的可选路线，不是项目的硬性要求。它可以直接复用 Bash、Linux Python、GCC、nvcc 和当前 .sh 入口，通常能减少 Windows 原生编译差异。

如果使用 WSL2：

- Windows 主机安装 NVIDIA 驱动；
- WSL 内不要安装 Linux NVIDIA display driver；
- 项目和数据尽量放在 WSL 的 Linux 文件系统内，不要长期放在 /mnt/c 下训练；
- WSL 内单独安装 Python/Conda/PyTorch/CUDA 编译工具。

两条路线的核心差异是编译和脚本便利性，不是 4090 是否能被 PyTorch 使用。

## 5. 创建 Python 环境

推荐先用 Python 3.10，保持和服务器一致：

~~~powershell
conda create -n ref-mv python=3.10 -y
conda activate ref-mv
python -m pip install --upgrade pip setuptools wheel ninja
~~~

先安装当前服务器实际使用的 PyTorch CUDA 11.8 组合：

~~~powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
~~~

然后安装其余 Python 依赖。不要直接把 Ref-Gaussian 的 requirements.txt 当作完整环境锁定文件，因为其中的 torch==2.0.0 和 torchvision==0.15.0 与当前联合环境不一致。建议先安装不包含这两个 PyTorch 条目的其余依赖，再安装：

~~~
opencv-python
huggingface_hub
hydra-core
iopath
wcmatch
tensorboard
pyexr
fvcore
~~~

Ref-Gaussian 还需要其 requirements 中列出的 numpy、kornia、open3d、Pillow、plyfile、scipy、trimesh、mediapy、imageio 等依赖。若某个旧版本在 Windows 没有对应 wheel，应优先保持 API 兼容，再记录实际安装版本，不要随意升级 PyTorch 主版本。

## 6. 编译 CUDA 扩展

项目不是纯 Python。首次配置必须重新编译本机扩展，不能复制服务器上的 Linux .so 文件。

需要编译的主要扩展：

~~~
third_party/ref_gaussian/submodules/cubemapencoder
third_party/ref_gaussian/submodules/diff-surfel-rasterization
third_party/ref_gaussian/submodules/raytracing
third_party/ref_gaussian/submodules/simple-knn
~~~

在项目根目录执行：

~~~powershell
python -m pip install -e .
python -m pip install -e third_party/ref_gaussian/submodules/cubemapencoder
python -m pip install -e third_party/ref_gaussian/submodules/diff-surfel-rasterization
python -m pip install -e third_party/ref_gaussian/submodules/raytracing
python -m pip install -e third_party/ref_gaussian/submodules/simple-knn
python -m pip install nvdiffrast==0.3.1
~~~

renderutils 不是单独的 pip 包；它在第一次进入 scene.renderutils.ops 时通过 torch.utils.cpp_extension.load 动态编译。因此即使前面四个扩展安装成功，第一次完整 import 或训练仍可能触发一次额外编译。

如果出现 cl.exe not found、nvcc not found、cuda.lib 或 nvrtc.lib 找不到，优先检查 Visual Studio Developer PowerShell、CUDA Toolkit 和 PATH，不要修改训练逻辑来绕过编译错误。

## 7. 原生 Windows 环境变量

当前 scripts/env.sh 写的是服务器 Linux 路径，因此原生 Windows 需要在 PowerShell 中设置等价路径。假设项目位于 D:\work\ref-mv：

~~~powershell
$env:REF_MV_ROOT = (Get-Location).Path
$env:REF_GAUSSIAN_ROOT = Join-Path $env:REF_MV_ROOT "third_party\ref_gaussian"
$env:MVINVERSE_ROOT = Join-Path $env:REF_MV_ROOT "third_party\mvinverse"
$paths = @(
    $env:REF_GAUSSIAN_ROOT,
    (Join-Path $env:REF_GAUSSIAN_ROOT "submodules\diff-surfel-rasterization"),
    (Join-Path $env:REF_GAUSSIAN_ROOT "submodules\cubemapencoder"),
    (Join-Path $env:REF_GAUSSIAN_ROOT "submodules\raytracing"),
    (Join-Path $env:REF_GAUSSIAN_ROOT "submodules\simple-knn"),
    $env:MVINVERSE_ROOT,
    $env:REF_MV_ROOT
)
$env:PYTHONPATH = (($paths -join ";") + ";" + $env:PYTHONPATH)
~~~

PowerShell 代码块中的 `$` 是实际语法，复制到 PowerShell 时保持原样。

这些变量只对当前 PowerShell 窗口有效。也可以用 IDE 的 Python interpreter 和工作目录配置替代它们，但必须保证上述目录能被 import 到。

## 8. 环境检查

在 WSL2 中可以直接执行：

~~~
source scripts/env.sh
scripts/check_environment.sh
~~~

原生 Windows 若不能运行 .sh，至少执行以下检查：

~~~powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); import ref_mv, mvinverse, gaussian_renderer, diff_surfel_rasterization, raytracing, cubemapencoder, simple_knn; print('imports: PASS')"
~~~

检查必须同时确认：

1. torch.cuda.is_available() 为 True；
2. GPU 名称为本机 NVIDIA GPU；
3. diff_surfel_rasterization、raytracing、cubemapencoder、simple_knn 的 CUDA 扩展可以 import；
4. ref_mv、gaussian_renderer、mvinverse 来自当前 clone，而不是其他旧目录；
5. nvdiffrast 可以被 Ref-Gaussian 导入。

## 9. 数据、模型和 checkpoint

Git 只保存代码和文档。以下内容不会从远程仓库自动拉下来：

- data/ 数据集；
- output/ 训练输出；
- Ref saved iteration；
- joint.pt；
- Hugging Face/MVInverse 模型缓存；
- CUDA 编译缓存和 Python cache。

当前服务器的大致占用是：

~~~
整个工作区（含 data/output） 约 22 GB
data                         约 13 GB
output                       约 8.2 GB
Hugging Face cache            约 3.8 GB
~~~

只迁移 office 首轮实验时，复制对应的 data/office/... 即可，不需要复制全部数据集。长期使用建议 Windows 目标盘预留至少 100 GB；只做代码和小规模测试时 40–50 GB 可以启动。

MVInverse 默认 checkpoint 标识是 maddog241/mvinverse，第一次运行可能需要联网下载约 4 GB 的模型。也可以把服务器上的 Hugging Face 缓存单独复制到 Windows 后配置 HF_HOME，但不要把缓存放进 Git。

数据和 checkpoint 应使用移动硬盘、NAS、robocopy、rsync 或其他文件同步工具传输。例如 PowerShell：

~~~powershell
robocopy E:\ref-mv-data D:\work\ref-mv\data /E /Z /J
robocopy E:\ref-mv-output D:\work\ref-mv\output /E /Z /J
~~~

不要让两台机器同时写同一个 output 实验目录；每台机器使用独立实验目录，checkpoint 通过明确的单向复制同步。

## 10. 4090 上的首轮运行建议

4090 的显存是 24 GB。历史联合训练在 -r4、num_views=2 的旧版本上曾观察到约 35.4 GiB 显存，因此不要在新机器上直接照搬高分辨率配置。

先单独运行 Ref 基线：

~~~powershell
python third_party/ref_gaussian/train.py -s data/office/office_0422_refgaussian_60views -r 2 --iterations 20000 --save_iterations 20000
~~~

再从 Ref saved iteration 启动 joint。下面的路径只是示例，输入 Ref 目录和 joint 输出目录必须不同：

~~~powershell
python train.py -s data/office/office_0422_refgaussian_60views -r 2 --volume_render_until_iter 0 --ref_model_path output/office_ref_20000 --ref_iteration 20000 --model_path output/office_joint_after_ref_windows --iterations 30000 --save_iterations 30000 --gs_feature_lr 0.001 --joint_envmap_lr 0.001 --gs_patch_lr 5e-5 --neighbor_k 8 --num_views 2
~~~

4090 是单卡，当前 run_office_experiments.sh 的示例会并行启动 GPU 0 和 GPU 1，不应在单卡上直接运行。Windows 上应先串行跑 Ref，再串行跑 joint。

如果 -r2、num_views=2 仍然 OOM：

- 先记录实际峰值显存和触发阶段；
- 可以用更低 -r 做开发 smoke test；
- 可以临时用 num_views=1 验证代码链路，但不能把它当作最终多视角实验结论；
- 不要仅通过修改学习率解决 OOM，学习率不是主要显存因素。

## 11. Git 同步工作流

日常同步只提交代码、配置和文档，不提交数据与输出：

~~~
git status
git add <明确的代码或文档路径>
git commit -m "Describe the change"
git push origin main
~~~

如果修改了 submodule，必须先在 submodule 内提交并推送，再回到根仓库提交新的 submodule 指针：

~~~
git -C third_party/ref_gaussian add <paths>
git -C third_party/ref_gaussian commit -m "Describe Ref-Gaussian change"
git -C third_party/ref_gaussian push origin main

git add third_party/ref_gaussian
git commit -m "Update Ref-Gaussian submodule"
git push origin main
~~~

提交前检查大文件：

~~~
git status --short --untracked-files=all
git ls-files | xargs -I{} sh -c 'test "$(stat -c %s "{}" 2>/dev/null)" -ge 52428800 && echo "large: {}"'
~~~

Windows 原生 PowerShell 不要使用最后一条 Bash 命令；可以直接在 Git Bash、WSL2 中执行，或用资源管理器/PowerShell 检查文件大小。当前 .gitignore 已忽略 data/、output/、build/、.so、.pyd、.dll、.o 和 Python cache，但新增大模型文件仍应放在这些目录之外并在提交前人工确认。

## 12. 给后续 AI 的接手顺序

第一次接手这个 clone 时，按以下顺序阅读：

1. 本文件 docs/WINDOWS_MIGRATION.md；
2. AGENTS.md，了解代码边界和禁止事项；
3. docs/STATUS.md，确认当前研究阶段；
4. docs/DECISIONS.md，确认已冻结的接口和训练约束；
5. configs/README.md，确认命令行参数；
6. README.md 和 docs/REFERENCE.md；
7. 需要修改联合逻辑时再读 ref_mv/joint.py、根目录 train.py 和对应 third-party 文件。

后续 AI 不应：

- 把 data/、output/、模型缓存或 checkpoint 加入 Git；
- 修改项目外的原始上游目录；
- 未经确认改变 Ref loss、PBR 公式、相机采样或 checkpoint 格式；
- 把 num_views=1 的 smoke test 写成正式多视角实验结论；
- 直接运行双 GPU 实验脚本来判断单张 4090 是否支持当前配置。

迁移的第一目标是让 CUDA 扩展和 import 检查通过，第二目标是用 -r2、num_views=2 做短程训练并记录峰值显存，第三目标才是进行完整长训练。
