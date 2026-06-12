"""Gate-zero bit-identity check for the TSDF depth read (depth_tsdf_design.md §5.4, roadmap WS3).

Loads the SAME checkpoint twice:
  A) policy.tsdf_config = None        — unpatched, depth-free expert
  B) policy.tsdf_config as configured — patched expert, fresh encoder, tanh(0) = 0 gate
then runs predict_action_chunk on one dataset frame with identical fixed-seed flow
noise. At init the gated depth read must contribute exactly zero, so the two action
chunks must be BITWISE equal — any drift means the patch is not a no-op at init.

Referenced by tests/policies/test_depth_tsdf.py; lives here rather than in pytest
because it needs a GPU and the real checkpoint.

Run:
    uv run python -m lerobot.probes.tsdf_bit_identity --config config_rl.yaml
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
class TsdfBitIdentityConfig(TrainRLServerPipelineConfig):
    bit_identity_frame: int = 0  # global dataset index of the probe frame


def _frame_obs(dataset, cfg: TsdfBitIdentityConfig) -> tuple[dict, str]:
    obs, _, _, _, task_str, ep_idx, fr_idx = get_frame_data(
        dataset, cfg.bit_identity_frame, int(cfg.policy.chunk_size)
    )
    depth_key = cfg.policy.tsdf_config.depth_key
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
def cli(cfg: TsdfBitIdentityConfig):
    init_logging()
    tsdf_config = getattr(cfg.policy, "tsdf_config", None)
    if tsdf_config is None:
        raise SystemExit("policy.tsdf_config is null in this config — nothing to verify.")
    device = get_safe_torch_device(try_device=cfg.policy.device)

    from lerobot.datasets.factory import make_dataset

    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    obs, task_str = _frame_obs(dataset, cfg)

    def predict(label: str) -> torch.Tensor:
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
        # Eager attention for both runs; CUDA-graph capture/replay is not bit-stable vs eager.
        adapter._set_probe_cuda_graph_enabled(False)
        batch = adapter._make_batch(obs, task_str, advantage=1.0)
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        actions = (
            adapter.policy.predict_action_chunk(
                batch, inference_action_mode="continuous", generator=generator
            )
            .float()
            .cpu()
        )
        logging.info(f"Policy {label}: actions {tuple(actions.shape)}")
        del adapter
        torch.cuda.empty_cache()
        return actions

    cfg.policy.tsdf_config = None
    actions_a = predict("A (tsdf_config=None)")
    cfg.policy.tsdf_config = tsdf_config
    actions_b = predict("B (tsdf_config set, gate=0 init)")

    if torch.equal(actions_a, actions_b):
        logging.info("[ok] Bit-identical at gate=0 — the patched expert is a no-op at init.")
    else:
        diff = (actions_a - actions_b).abs().max().item()
        raise SystemExit(
            f"[FAIL] Action chunks differ (max |delta| = {diff:.3e}) — the depth read is not a no-op at init."
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
