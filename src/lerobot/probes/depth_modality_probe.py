"""2×2 modality probe + sensitivity estimates for the joint depth read
(depth_redesign_options.md §5.1).

Runs predict_action_chunk on the same frame(s) under the four modality
conditions — {RGB+depth, RGB-only, depth-only, neither} — with IDENTICAL
fixed-seed flow noise, and reports:

  - normalized-space MSE vs GT per condition (is depth earning its keep:
    mse(RGB) − mse(RGB+depth) is the headline number),
  - pairwise max|Δaction| between conditions (does depth influence output at all),
  - per-layer attention mass on the depth columns (via the mass-capture hook),
  - finite-difference sensitivity: ‖Δactions‖ for a 1%-of-std perturbation of the
    raw depth input vs of the wrist RGB input (a directional-Jacobian estimate —
    separates "attended but ignored" from load-bearing).

Conditions are produced within one policy build: depth− removes the depth key
(learned null bank), while RGB− applies the exact training-time wrist-patch
attention mask, which also triggers the model-side bridge kill. Rebuild-to-rebuild
nondeterminism (~1e-1, from-scratch norm buffers) would swamp the comparisons.

Replaces probes/pointmap_bit_identity.py (retired 2026-07-26 with the α gate:
there is no gate-0 no-op to verify under the joint softmax).

Run:
    uv run python -m lerobot.probes.depth_modality_probe --config config_rl.yaml \
        --frame_indices 0,500,1200
"""

import logging
import sys
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.depth_pointmap.modeling_stream import mask_camera_patch_span
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    drain_pointmap_mass_records,
    set_pointmap_mass_capture,
)
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import get_frame_data
from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

CONDITIONS = ("rgb+depth", "rgb_only", "depth_only", "neither")
CONDITION_PAIRS = tuple(combinations(CONDITIONS, 2))


@dataclass
class DepthModalityProbeConfig(TrainRLServerPipelineConfig):
    frame_indices: str = "0"  # comma-separated global dataset indices
    fd_epsilon_rel: float = 0.01  # FD perturbation, fraction of the input's std


def _frame_obs(dataset, cfg, global_idx: int):
    obs, gt_actions, state, _, task_str, ep_idx, fr_idx = get_frame_data(
        dataset, global_idx, int(cfg.policy.chunk_size)
    )
    depth_key = cfg.policy.pointmap_config.depth_key
    depth = load_depth_png(dataset.root, f"{depth_key}.depth", ep_idx, fr_idx)
    obs[f"observation.depth.{depth_key}"] = torch.from_numpy(depth.astype(np.float32)).reshape(
        1, 1, *depth.shape
    )
    return obs, gt_actions, state, task_str


def _condition_obs(obs: dict, condition: str, *, depth_obs_key: str) -> dict:
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
    if condition in ("rgb_only", "neither"):
        out.pop(depth_obs_key, None)  # encoder falls back to the learned null bank
    return out


@parser.wrap()
def cli(cfg: DepthModalityProbeConfig):
    init_logging()
    pointmap_config = getattr(cfg.policy, "pointmap_config", None)
    if pointmap_config is None:
        raise SystemExit("policy.pointmap_config is null in this config — nothing to probe.")
    device = get_safe_torch_device(try_device=cfg.policy.device)

    from lerobot.datasets.factory import make_dataset

    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None

    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)
    policy = adapter.policy
    depth_key = pointmap_config.depth_key
    depth_obs_key = f"observation.depth.{depth_key}"
    rgb_obs_key = f"observation.images.{depth_key}"
    image_patch_id, num_images, cam_index = policy._pointmap_wrist_meta()

    def predict(obs, task_str, *, rgb_on: bool = True) -> torch.Tensor:
        batch = adapter._make_batch(obs, task_str)
        if not rgb_on:
            batch["attention_mask"] = mask_camera_patch_span(
                batch["attention_mask"],
                batch["input_ids"],
                image_patch_id=image_patch_id,
                num_images=num_images,
                cam_index=cam_index,
            )
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        return (
            policy.predict_action_chunk(
                batch, inference_action_mode="continuous", generator=generator
            )
            .float()
            .cpu()
            .squeeze(0)
        )

    frame_indices = [int(s) for s in str(cfg.frame_indices).split(",") if s.strip() != ""]
    if not frame_indices:
        raise SystemExit("--frame_indices must contain at least one dataset index.")
    mse_by_condition: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    pairwise_deltas: dict[tuple[str, str], list[float]] = {pair: [] for pair in CONDITION_PAIRS}
    mass_by_layer: list[np.ndarray] = []
    sens_depth_list: list[float] = []
    sens_rgb_list: list[float] = []

    for global_idx in frame_indices:
        obs, gt_actions, state, task_str = _frame_obs(dataset, cfg, global_idx)
        gt_norm = adapter.normalize_gt_actions(gt_actions, state).float()

        mass: list[float] = []
        actions: dict[str, torch.Tensor] = {}
        for condition in CONDITIONS:
            cond_obs = _condition_obs(
                obs, condition, depth_obs_key=depth_obs_key
            )
            capture_mass = condition == "rgb+depth"
            if capture_mass:
                set_pointmap_mass_capture(True)
            try:
                actions[condition] = predict(
                    cond_obs, task_str, rgb_on=condition in ("rgb+depth", "rgb_only")
                )
            finally:
                if capture_mass:
                    mass = drain_pointmap_mass_records()
                    set_pointmap_mass_capture(False)

        horizon = min(actions["rgb+depth"].shape[0], gt_norm.shape[0])
        logging.info(f"frame {global_idx}:")
        for condition in CONDITIONS:
            mse = torch.nn.functional.mse_loss(
                actions[condition][:horizon], gt_norm[:horizon]
            ).item()
            mse_by_condition[condition].append(mse)
            logging.info(f"  mse_norm[{condition:>9s}] = {mse:.5f}")
        for left, right in CONDITION_PAIRS:
            delta = (actions[left] - actions[right]).abs().max().item()
            pairwise_deltas[(left, right)].append(delta)
            logging.info(f"  max|Δ| {left} vs {right} = {delta:.4e}")
        if mass:
            # mass has one record per (denoise step × layer); fold to per-layer means.
            layers = len(policy.depth_stream.blocks)
            if len(mass) % layers:
                raise RuntimeError(
                    f"Captured {len(mass)} depth-mass records for {layers} layers."
                )
            per_layer = np.asarray(mass, dtype=np.float64).reshape(-1, layers).mean(axis=0)
            mass_by_layer.append(per_layer)
            logging.info(
                f"  depth_attn_mass: mean={per_layer.mean():.4f} max={per_layer.max():.4f} "
                f"argmax=layer {int(per_layer.argmax())}"
            )
            for layer_idx, layer_mass in enumerate(per_layer):
                logging.info(f"  depth_attn_mass/l{layer_idx:02d} = {layer_mass:.4f}")

        # Finite-difference sensitivity: same seed, 1%-of-std input perturbation.
        eps = float(cfg.fd_epsilon_rel)
        depth_raw = obs[depth_obs_key]
        pert = torch.randn_like(depth_raw) * depth_raw.float().std() * eps
        obs_pert = {**obs, depth_obs_key: depth_raw + pert}
        sens_depth = (predict(obs_pert, task_str) - actions["rgb+depth"]).norm().item()
        rgb_raw = obs[rgb_obs_key].float()
        pert = torch.randn_like(rgb_raw) * rgb_raw.std() * eps
        obs_pert = {**obs, rgb_obs_key: (rgb_raw + pert).to(obs[rgb_obs_key].dtype)}
        sens_rgb = (predict(obs_pert, task_str) - actions["rgb+depth"]).norm().item()
        sens_depth_list.append(sens_depth)
        sens_rgb_list.append(sens_rgb)
        logging.info(
            f"  fd sensitivity (‖Δactions‖ @ {eps:.0%} input noise): depth={sens_depth:.4e} "
            f"rgb={sens_rgb:.4e} ratio={sens_depth / max(sens_rgb, 1e-12):.3f}"
        )

    logging.info("── summary over frames ──")
    for condition in CONDITIONS:
        vals = mse_by_condition[condition]
        logging.info(f"mse_norm[{condition:>9s}] mean = {sum(vals) / len(vals):.5f}")
    for left, right in CONDITION_PAIRS:
        vals = pairwise_deltas[(left, right)]
        logging.info(f"max|Δ| {left} vs {right} mean = {sum(vals) / len(vals):.4e}")
    if mass_by_layer:
        mean_mass = np.stack(mass_by_layer).mean(axis=0)
        logging.info(
            f"depth_attn_mass mean={mean_mass.mean():.4f} max={mean_mass.max():.4f}"
        )
        for layer_idx, layer_mass in enumerate(mean_mass):
            logging.info(f"depth_attn_mass/l{layer_idx:02d} mean = {layer_mass:.4f}")
    both = sum(mse_by_condition["rgb+depth"]) / len(frame_indices)
    rgb = sum(mse_by_condition["rgb_only"]) / len(frame_indices)
    logging.info(
        f"depth benefit  mse(rgb_only) − mse(rgb+depth) = {rgb - both:+.5f} "
        f"(positive ⇒ depth helps)"
    )
    logging.info(
        f"mean fd sensitivity: depth={sum(sens_depth_list) / len(frame_indices):.4e} "
        f"rgb={sum(sens_rgb_list) / len(frame_indices):.4e}"
    )


def main() -> None:
    # Same pre-parse machinery as rl_offline: register policy configs, strip inactive-model YAML fields.
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    import lerobot.rl.pi05.rl_pi05  # noqa: F401 — registers PI05RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401 — registers robot configs
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401 — registers teleop configs

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    main()
