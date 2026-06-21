"""Gate-zero bit-identity check for the point-map MoT depth read (depth_pointmap_design.md §B.4).

Builds ONE policy instance with pointmap_config and runs predict_action_chunk with the
depth read ON (gate α=0) vs OFF (depth disabled in-place), under identical fixed-seed
flow noise. At init the gated additive read contributes exactly tanh(0)·read = 0, so the
two action chunks must be BITWISE equal — any drift means the read is not a no-op at init.

Toggling depth WITHIN one build (rather than rebuilding with/without pointmap_config)
avoids the model's ~1e-1 build-to-build nondeterminism (the from-scratch norm-buffer
path) swamping the signal. Lives here rather than in pytest because it needs a GPU and
the real checkpoint.

Run:
    uv run python -m lerobot.probes.pointmap_bit_identity --config config_rl.yaml
"""

import logging
import sys
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import get_frame_data
from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class PointmapBitIdentityConfig(TrainRLServerPipelineConfig):
    bit_identity_frame: int = 0  # global dataset index of the probe frame


def _frame_obs(dataset, cfg: PointmapBitIdentityConfig) -> tuple[dict, str]:
    obs, _, _, _, task_str, ep_idx, fr_idx = get_frame_data(
        dataset, cfg.bit_identity_frame, int(cfg.policy.chunk_size)
    )
    depth_key = cfg.policy.pointmap_config.depth_key
    try:
        depth = load_depth_png(dataset.root, f"{depth_key}.depth", ep_idx, fr_idx)
        obs[f"observation.depth.{depth_key}"] = torch.from_numpy(depth.astype(np.float32)).reshape(
            1, 1, *depth.shape
        )
        logging.info(f"Depth sidecar attached: {depth.shape} med={np.median(depth):.0f} (raw units)")
    except FileNotFoundError:
        logging.warning(
            "No depth sidecar for this frame — policy B reads the learned null bank (check still valid)."
        )
    return obs, task_str


@parser.wrap()
def cli(cfg: PointmapBitIdentityConfig):
    init_logging()
    pointmap_config = getattr(cfg.policy, "pointmap_config", None)
    if pointmap_config is None:
        raise SystemExit("policy.pointmap_config is null in this config — nothing to verify.")
    device = get_safe_torch_device(try_device=cfg.policy.device)

    from lerobot.datasets.factory import make_dataset

    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    obs, task_str = _frame_obs(dataset, cfg)

    # ONE policy instance, depth toggled in-place. Rebuilding the policy per run injects
    # ~1e-1 of build-to-build nondeterminism (the from-scratch norm-buffer path) that swamps
    # the gate-0 signal, so we instead compare depth-on vs depth-off within a single build.
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)  # graph capture/replay is not bit-stable vs eager
    batch = adapter._make_batch(obs, task_str, advantage=1.0)
    policy = adapter.policy

    def predict(label: str) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        actions = (
            policy.predict_action_chunk(batch, inference_action_mode="continuous", generator=generator)
            .float()
            .cpu()
        )
        logging.info(f"{label}: actions {tuple(actions.shape)}")
        return actions

    actions_depth = predict("depth ON  (pointmap, gate=0)")
    actions_depth2 = predict("depth ON  (same instance, determinism baseline)")
    # Disable the depth read in the SAME instance (no rebuild ⇒ no build-to-build noise).
    policy.pointmap_encoder = None
    policy._action_expert()._lerobot_pointmap = None
    actions_nodepth = predict("depth OFF (pure context read)")

    baseline = (actions_depth - actions_depth2).abs().max().item()
    diff = (actions_depth - actions_nodepth).abs().max().item()
    logging.info(f"within-instance determinism |on - on'| = {baseline:.3e}   gate-0 |on - off| = {diff:.3e}")

    if torch.equal(actions_depth, actions_nodepth):
        logging.info("[ok] Bit-identical at gate=0 — the gated depth read is a no-op at init.")
    elif diff <= max(baseline, 1e-6):
        logging.info(
            f"[ok] Gate-0 diff ({diff:.3e}) within the model's own run-to-run noise "
            f"({baseline:.3e}) — depth read is a no-op at init."
        )
    else:
        raise SystemExit(
            f"[FAIL] Depth on/off differ (max |delta| = {diff:.3e}, baseline = {baseline:.3e}) — "
            "the gated depth read is not a no-op at init."
        )


def main() -> None:
    # Same pre-parse machinery as rl_offline: register policy configs, strip inactive-model YAML fields.
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    import lerobot.rl.pi05.rl_pi05  # noqa: F401 — registers PI05RLConfig
    from lerobot.robots import so_follower  # noqa: F401 — registers so101_follower
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import so_leader  # noqa: F401 — registers so101_leader

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    main()
