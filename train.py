"""Joint Ref-Gaussian × MVInverse training entry point."""

import os
import shlex
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
REF_ROOT = PROJECT_ROOT / "third_party" / "ref_gaussian"
MV_ROOT = PROJECT_ROOT / "third_party" / "mvinverse"
sys.path.insert(0, str(MV_ROOT))
sys.path.insert(0, str(REF_ROOT))

import train as ref_train
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser(description="Ref-MV joint training")
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)

    # Keep the upstream Ref CLI surface intact.
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[10000, 20000, 30000, 40000, 50000, 60000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # Joint defaults reuse the closest upstream learning-rate scales.
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--ref_iteration", type=int, default=None)
    parser.add_argument("--joint_checkpoint", type=str, default=None)
    parser.add_argument("--mv_checkpoint", type=str, default="maddog241/mvinverse")
    parser.add_argument("--gs_feature_dim", type=int, default=1,
                        help="Per-Gaussian conditioning feature dimension (1–16)")
    parser.add_argument("--gs_feature_lr", type=float, default=0.001)
    parser.add_argument("--gs_patch_lr", type=float, default=5e-5)
    parser.add_argument(
        "--joint_envmap_lr",
        type=float,
        default=None,
        help="Override the environment-map learning rate after joint training starts",
    )
    parser.add_argument("--neighbor_k", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=2)

    args = parser.parse_args(sys.argv[1:])
    if not 1 <= args.gs_feature_dim <= 16:
        parser.error("--gs_feature_dim must be between 1 and 16 (8 of 24 rasterizer channels are reserved)")
    if (args.ref_model_path is None) != (args.ref_iteration is None):
        parser.error("--ref_model_path and --ref_iteration must be provided together")
    if args.joint_checkpoint is not None and args.ref_model_path is None:
        parser.error("--joint_checkpoint requires a loaded Ref saved iteration")
    if args.start_checkpoint is not None:
        parser.error(
            "Ref-MV restores Ref state from --ref_model_path/--ref_iteration; "
            "--start_checkpoint is not a joint restore path"
        )

    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [
        i for i in range(10000, args.iterations + 1, 5000)
    ]
    args.test_iterations.append(args.volume_render_until_iter)

    if not args.model_path:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_name = os.path.basename(os.path.normpath(args.source_path))
        args.model_path = os.path.join(
            "./output", scene_name, f"{current_time}-{scene_name}"
        )
    if args.ref_model_path is not None:
        if os.path.abspath(args.ref_model_path) == os.path.abspath(args.model_path):
            parser.error("Joint output --model_path must differ from --ref_model_path")

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "run_config.txt"), "a", encoding="utf-8") as log:
        log.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n")
        log.write(f"command={shlex.join([sys.executable, *sys.argv])}\n")
        log.write(f"gs_feature_dim={args.gs_feature_dim}\n")
        log.write(f"gs_feature_lr={args.gs_feature_lr}\n")
        log.write(f"gs_patch_lr={args.gs_patch_lr}\n")
        log.write(f"joint_envmap_lr={args.joint_envmap_lr}\n")
        log.write(f"neighbor_k={args.neighbor_k}\n")
        log.write(f"num_views={args.num_views}\n")
        log.write(f"mv_checkpoint={args.mv_checkpoint}\n\n")

    dataset = model_params.extract(args)
    dataset.joint_patch_size = 14
    joint_args = Namespace(
        ref_model_path=args.ref_model_path,
        ref_iteration=args.ref_iteration,
        joint_checkpoint=args.joint_checkpoint,
        mv_checkpoint=args.mv_checkpoint,
        gs_feature_lr=args.gs_feature_lr,
        gs_patch_lr=args.gs_patch_lr,
        joint_envmap_lr=args.joint_envmap_lr,
        neighbor_k=args.neighbor_k,
        num_views=args.num_views,
        feature_dim=args.gs_feature_dim,
    )

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Ref's logger/visualization helpers read the parsed namespace directly.
    ref_train.args = args
    ref_train.training(
        dataset,
        optimization_params.extract(args),
        pipeline_params.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        None,
        args.model_path,
        joint_args=joint_args,
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
