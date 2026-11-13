import os
import random

import torch
from torch import nn


PATCH_SIZE = 14


class RefMVModel(nn.Module):
    """Minimal Ref-Gaussian to MVInverse conditioning bridge."""

    def __init__(self, gaussians, mvinverse, feature_dim):
        super().__init__()
        self.gaussians = gaussians
        self.mvinverse = mvinverse.eval().requires_grad_(False)
        self.gs_patch_embed = nn.Conv2d(
            feature_dim,
            1024,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            bias=False,
        )

    def train(self, mode=True):
        super().train(mode)
        self.mvinverse.eval()
        return self

    def forward(self, cameras, pipe, background, opt,
                reuse_rasterization=True):
        from gaussian_renderer import rasterize_surfel, render_surfel, shade_surfel

        if not cameras:
            raise ValueError("RefMVModel requires at least one camera")

        sizes = [tuple(camera.original_image.shape[-2:]) for camera in cameras]
        if len(set(sizes)) != 1:
            raise ValueError(f"All joint cameras must share H/W, got {sizes}")

        images = []
        rasterized_views = []
        feature_maps = []
        for camera in cameras:
            expected_size = (camera.image_height, camera.image_width)
            if tuple(camera.original_image.shape[-2:]) != expected_size:
                raise ValueError(
                    f"Camera {camera.image_name} image size does not match camera metadata: "
                    f"{tuple(camera.original_image.shape[-2:])} vs {expected_size}"
                )
            rasterized = rasterize_surfel(
                camera,
                self.gaussians,
                pipe,
                background,
                gaussian_feature=self.gaussians.get_gs_feature,
            )
            images.append(camera.original_image[:3])
            rasterized_views.append(rasterized)
            feature_maps.append(rasterized["gs_feature"])

        images = torch.stack(images)
        features = torch.stack(feature_maps)
        height, width = images.shape[-2:]
        if height % PATCH_SIZE or width % PATCH_SIZE:
            raise ValueError(
                f"Joint image size must be divisible by {PATCH_SIZE}, got {(height, width)}"
            )

        gs_tokens = self.gs_patch_embed(features).flatten(2).transpose(1, 2)
        # MVInverse's official inference path runs the frozen model in fp16 autocast.
        with torch.amp.autocast("cuda", dtype=torch.float16):
            material = self.mvinverse(
                images.unsqueeze(0),
                hidden_patch_residual=gs_tokens.unsqueeze(0),
                output_names=("albedo", "metallic", "roughness"),
            )

        render_pkgs = []
        for index, (camera, rasterized) in enumerate(zip(cameras, rasterized_views)):
            material_maps = {
                "albedo": material["albedo"][0, index],
                "metallic": material["metallic"][0, index],
                "roughness": material["roughness"][0, index],
            }
            if reuse_rasterization:
                render_pkg = shade_surfel(
                    camera,
                    self.gaussians,
                    pipe,
                    background,
                    rasterized,
                    srgb=opt.srgb,
                    opt=opt,
                    material_maps=material_maps,
                )
            else:
                render_pkg = render_surfel(
                    camera,
                    self.gaussians,
                    pipe,
                    background,
                    srgb=opt.srgb,
                    opt=opt,
                    material_maps=material_maps,
                )
            render_pkgs.append(render_pkg)

        return {
            "render_pkgs": render_pkgs,
            "material": material,
            "feature_maps": features,
            "gs_tokens": gs_tokens,
        }

    def render_camera(self, viewpoint_camera, pc, pipe, bg_color,
                      scaling_modifier=1.0, override_color=None,
                      srgb=False, opt=None):
        if pc is not self.gaussians:
            raise ValueError("RefMVModel received a different GaussianModel")
        if scaling_modifier != 1.0 or override_color is not None:
            raise ValueError("Joint rendering does not support Ref render overrides")
        return self([viewpoint_camera], pipe, bg_color, opt)["render_pkgs"][0]


def build_neighbor_table(cameras, neighbor_k):
    if neighbor_k < 0:
        raise ValueError(f"neighbor_k must be non-negative, got {neighbor_k}")
    centers = torch.stack([
        camera.camera_center.detach().cpu() for camera in cameras
    ])
    k_eff = min(neighbor_k, len(cameras) - 1)
    if k_eff == 0:
        return torch.empty((len(cameras), 0), dtype=torch.long)
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float("inf"))
    return distances.topk(k_eff, largest=False).indices


class CameraSampler:
    def __init__(self, cameras, num_views, neighbor_k):
        if not cameras:
            raise ValueError("Joint training requires at least one training camera")
        self.cameras = cameras
        self.num_views = num_views
        self.neighbors = build_neighbor_table(cameras, neighbor_k)
        max_views = self.neighbors.shape[1] + 1
        if not 1 <= num_views <= max_views:
            raise ValueError(
                f"num_views must be in [1, {max_views}] for this scene, got {num_views}"
            )
        self._anchor_stack = []

    def sample(self):
        if not self._anchor_stack:
            self._anchor_stack = list(range(len(self.cameras)))
        anchor_pos = random.randint(0, len(self._anchor_stack) - 1)
        anchor_index = self._anchor_stack.pop(anchor_pos)
        indices = [anchor_index]
        if self.num_views > 1:
            candidates = self.neighbors[anchor_index].tolist()
            indices.extend(random.sample(candidates, self.num_views - 1))
        return [self.cameras[index] for index in indices]


def initialize_joint(gaussians, mv_checkpoint, feature_dim, feature_lr,
                     patch_lr, joint_checkpoint=None, expected_iteration=None):
    from inference import load_model

    state = None
    initial_feature = None
    if joint_checkpoint is not None:
        if expected_iteration is None:
            raise ValueError("joint_checkpoint requires a loaded Ref saved iteration")
        state = torch.load(joint_checkpoint, map_location="cpu", weights_only=False)
        required = {
            "iteration", "gaussian_feature", "gs_patch_embed",
            "num_gaussians", "feature_dim",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"joint checkpoint is missing keys: {sorted(missing)}")
        if state["iteration"] != expected_iteration:
            raise ValueError(
                f"joint checkpoint iteration {state['iteration']} does not match "
                f"loaded Ref iteration {expected_iteration}"
            )
        if state["feature_dim"] != feature_dim:
            raise ValueError(
                f"joint checkpoint feature_dim {state['feature_dim']} does not match "
                f"requested {feature_dim}"
            )
        if state["num_gaussians"] != gaussians.get_xyz.shape[0]:
            raise ValueError(
                f"joint checkpoint has {state['num_gaussians']} Gaussians, loaded Ref "
                f"iteration has {gaussians.get_xyz.shape[0]}"
            )
        initial_feature = state["gaussian_feature"]

    gaussians.initialize_gs_feature(feature_dim, feature_lr, initial_feature)
    device = gaussians.get_xyz.device
    mvinverse = load_model(mv_checkpoint, device)
    model = RefMVModel(gaussians, mvinverse, feature_dim).to(device)
    if state is not None:
        model.gs_patch_embed.load_state_dict(state["gs_patch_embed"])
    patch_optimizer = torch.optim.Adam(model.gs_patch_embed.parameters(), lr=patch_lr)
    return model, patch_optimizer


def joint_checkpoint_path(model_path, iteration):
    return os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "joint.pt")


def save_joint_checkpoint(model_path, iteration, model, gaussians):
    state = {
        "iteration": iteration,
        "gaussian_feature": gaussians.get_gs_feature.detach().cpu(),
        "gs_patch_embed": {
            key: value.detach().cpu()
            for key, value in model.gs_patch_embed.state_dict().items()
        },
        "num_gaussians": gaussians.get_xyz.shape[0],
        "feature_dim": gaussians.gs_feature_dim,
    }
    torch.save(state, joint_checkpoint_path(model_path, iteration))
