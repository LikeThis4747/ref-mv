"""Render a compact Ref-vs-MV material comparison for one saved checkpoint."""

from __future__ import annotations

import argparse
import ast
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REF_ROOT = PROJECT_ROOT / "third_party" / "ref_gaussian"
MV_ROOT = PROJECT_ROOT / "third_party" / "mvinverse"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MV_ROOT))
sys.path.insert(0, str(REF_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import rasterize_surfel, shade_surfel  # noqa: E402
from ref_mv.joint import (  # noqa: E402
    build_neighbor_table,
    initialize_joint,
    joint_checkpoint_path,
)
from scene import GaussianModel, Scene  # noqa: E402
from utils.image_utils import visualize_depth  # noqa: E402


def read_namespace(path: Path) -> Namespace:
    return eval(path.read_text(), {"Namespace": Namespace})


def read_run_config(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if "=" not in line or line.startswith("command="):
            continue
        key, value = line.split("=", 1)
        try:
            values[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            values[key] = value
    return values


def as_image(tensor: torch.Tensor, mode: str = "rgb"):
    tensor = tensor.detach().float()
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3) and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    if mode == "normal":
        tensor = tensor[:3] * 0.5 + 0.5
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor[:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def show(ax, title, tensor=None, mode="rgb"):
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=5)
    if tensor is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16)
        return
    ax.imshow(as_image(tensor, mode))


def show_feature(ax, tensor):
    ax.axis("off")
    if tensor is None:
        ax.set_title("GS Feature (1D)", fontsize=10, pad=5)
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16)
        return
    feature = tensor.detach().float().squeeze().cpu()
    scale = max(torch.quantile(feature.abs().flatten(), 0.99).item(), 1e-8)
    ax.set_title(
        f"GS Feature (1D) | min {feature.min():.3g}, max {feature.max():.3g}",
        fontsize=10,
        pad=5,
    )
    ax.imshow(feature.numpy(), cmap="coolwarm", vmin=-scale, vmax=scale)


def load_scene(model_path: Path, iteration: int):
    cfg = read_namespace(model_path / "cfg_args")
    cfg.model_path = str(model_path)

    parser = ArgumentParser(add_help=False)
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset = model_params.extract(cfg)
    dataset.joint_patch_size = 14
    pipe = pipeline_params.extract(cfg)
    opt = optimization_params.extract(cfg)

    # cfg_args stores the initial value. Match train.py after indirect lighting starts.
    opt.indirect = int(iteration >= opt.indirect_from_iter)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=iteration,
        load_model_path=str(model_path),
        shuffle=False,
    )
    if opt.indirect:
        gaussians.load_mesh_from_ply(str(model_path), iteration)
    gaussians.training_setup(opt)
    return cfg, dataset, pipe, opt, scene, gaussians


def render_checkpoint(model_path: Path, iteration: int, camera_index: int):
    cfg, dataset, pipe, opt, scene, gaussians = load_scene(model_path, iteration)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    cameras = scene.getTrainCameras()
    if not 0 <= camera_index < len(cameras):
        raise IndexError(
            f"camera_index={camera_index} is outside the {len(cameras)} training cameras"
        )
    camera = cameras[camera_index]

    joint_path = Path(joint_checkpoint_path(str(model_path), iteration))
    mv_material = None
    feature_map = None
    init_images = None
    run_config = read_run_config(model_path / "run_config.txt")
    joint_model, _ = initialize_joint(
        gaussians=gaussians,
        mv_checkpoint=run_config.get("mv_checkpoint", "maddog241/mvinverse"),
        feature_dim=1,
        feature_lr=float(run_config.get("gs_feature_lr", 0.001)),
        patch_lr=float(run_config.get("gs_patch_lr", 5e-5)),
        joint_checkpoint=str(joint_path) if joint_path.exists() else None,
        expected_iteration=iteration if joint_path.exists() else None,
    )
    joint_model.eval()

    with torch.no_grad():
        if joint_model is not None:
            run_config = read_run_config(model_path / "run_config.txt")
            num_views = int(run_config.get("num_views", 2))
            neighbor_k = int(run_config.get("neighbor_k", 8))
            neighbor_table = build_neighbor_table(cameras, max(neighbor_k, num_views - 1))
            neighbor_ids = neighbor_table[camera_index][: max(0, num_views - 1)]
            joint_cameras = [camera] + [cameras[i] for i in neighbor_ids]
            joint_output = joint_model(joint_cameras, pipe, background, opt)
            feature_map = joint_output["feature_maps"][0]
            mv_material = {
                name: values[0, 0].float()
                for name, values in joint_output["material"].items()
            }
            input_images = torch.stack(
                [joint_camera.original_image[:3] for joint_camera in joint_cameras]
            )
            zero_features = torch.zeros_like(joint_output["feature_maps"])
            zero_tokens = joint_model.gs_patch_embed(zero_features).flatten(2).transpose(1, 2)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                init_material = joint_model.mvinverse(
                    input_images.unsqueeze(0),
                    hidden_patch_residual=zero_tokens.unsqueeze(0),
                    output_names=("albedo", "metallic", "roughness"),
                )
            init_images = {
                "Input RGB": input_images[0],
                "Init MV Albedo": init_material["albedo"][0, 0],
                "Init MV Metallic": init_material["metallic"][0, 0],
                "Init MV Roughness": init_material["roughness"][0, 0],
            }

        rasterized = rasterize_surfel(
            camera,
            gaussians,
            pipe,
            background,
            gaussian_feature=gaussians.get_gs_feature if joint_model else None,
        )
        ref_pkg = shade_surfel(
            camera,
            gaussians,
            pipe,
            background,
            rasterized,
            srgb=opt.srgb,
            opt=opt,
        )
        mv_pkg = None
        if mv_material is not None:
            mv_pkg = shade_surfel(
                camera,
                gaussians,
                pipe,
                background,
                rasterized,
                srgb=opt.srgb,
                opt=opt,
                material_maps=mv_material,
            )

        depth = visualize_depth(ref_pkg["surf_depth"])
        images = {
            "GT RGB": camera.original_image[:3],
            "Ref RGB": ref_pkg["render"],
            "MV Replaced RGB": None if mv_pkg is None else mv_pkg["render"],
            "GS Feature (1D)": feature_map,
            "Ref Albedo": ref_pkg["base_color_map"],
            "MV Albedo": None if mv_pkg is None else mv_pkg["base_color_map"],
            "Ref Diffuse": ref_pkg["diffuse_map"],
            "MV Diffuse": None if mv_pkg is None else mv_pkg["diffuse_map"],
            "Ref Reflection": ref_pkg["refl_strength_map"],
            "MV Metallic": None if mv_pkg is None else mv_pkg["refl_strength_map"],
            "Ref Specular": ref_pkg["specular_map"],
            "MV Specular": None if mv_pkg is None else mv_pkg["specular_map"],
            "Ref Roughness": ref_pkg["roughness_map"],
            "MV Roughness": None if mv_pkg is None else mv_pkg["roughness_map"],
            "Depth": depth,
            "Normal": ref_pkg["surf_normal"],
        }
        # Keep only the visualization tensors on CPU before returning. This lets
        # --iterations process several checkpoints without retaining each model
        # and its rendered images on the GPU.
        images = {
            name: None if image is None else image.detach().float().cpu()
            for name, image in images.items()
        }
        if init_images is not None:
            init_images = {
                name: image.detach().float().cpu()
                for name, image in init_images.items()
            }
    return images, init_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    iteration_group = parser.add_mutually_exclusive_group(required=True)
    iteration_group.add_argument("--iteration", type=int)
    iteration_group.add_argument("--iterations", nargs="+", type=int)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--init_only", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    iterations = args.iterations or [args.iteration]
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else model_path / "material_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    order = [
        ["GT RGB", "Ref RGB", "MV Replaced RGB", "GS Feature (1D)"],
        ["Ref Albedo", "MV Albedo", "Ref Diffuse", "MV Diffuse"],
        ["Ref Reflection", "MV Metallic", "Ref Specular", "MV Specular"],
        ["Ref Roughness", "MV Roughness", "Depth", "Normal"],
    ]
    modes = {"Normal": "normal"}
    for iteration in iterations:
        images, init_images = render_checkpoint(model_path, iteration, args.camera_index)
        if not args.init_only:
            output = output_dir / f"{iteration:06d}_cam{args.camera_index:03d}.png"
            fig, axes = plt.subplots(4, 4, figsize=(16, 12))
            fig.suptitle(
                f"{model_path.name} | iteration {iteration} | camera {args.camera_index}",
                fontsize=14,
            )
            for row, names in enumerate(order):
                for col, name in enumerate(names):
                    if name == "GS Feature (1D)":
                        show_feature(axes[row, col], images[name])
                    else:
                        show(axes[row, col], name, images[name], modes.get(name, "rgb"))
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(output, dpi=150)
            plt.close(fig)
            print(output)

        if init_images is not None:
            init_output = output_dir / (
                f"{iteration:06d}_mvinverse_init_cam{args.camera_index:03d}.png"
            )
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(
                f"{model_path.name} | iteration {iteration} | "
                "MVInverse with zero-initialized GS feature",
                fontsize=14,
            )
            for axis, name in zip(axes, init_images):
                show(axis, name, init_images[name])
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(init_output, dpi=150)
            plt.close(fig)
            print(init_output)


if __name__ == "__main__":
    main()
