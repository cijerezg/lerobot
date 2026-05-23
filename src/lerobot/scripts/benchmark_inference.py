#!/usr/bin/env python
"""
Benchmark MolmoAct2 inference latency.

Sweeps num_inference_steps and execution_horizon to find what's achievable.

The key constraint: at 30 Hz with chunk_size=30, inference must average < 0.5s
to avoid queue depletion. Target is < 0.45s to have margin.

Math reminder:
  queue_after_merge = chunk_size - ceil(latency * fps)
  steady-state works when latency < chunk_size / (2 * fps) = 0.5s

Usage:
  cd /home/user/Documents/Research/RL/LeRobot
  python lerobot/src/lerobot/rl/benchmark_inference.py --config-path lerobot/src/lerobot/rl/config_rl.yaml
"""

import logging
import time

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

import lerobot.rl.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
import lerobot.rl.gym_manipulator  # noqa: F401 — registers robots, cameras, teleops

logger = logging.getLogger(__name__)

N_WARMUP  = 5
N_MEASURE = 20

NUM_STEPS_LIST    = [1, 2, 3, 5]
EXEC_HORIZON_LIST = [5, 10, 15, 20]


def _budget_label(mean_s: float, fps: int, chunk_size: int) -> str:
    queue_left = chunk_size - int(np.ceil(mean_s * fps))
    if queue_left > 4:
        return f"OK      (+{queue_left:2d} actions)"
    if queue_left > 0:
        return f"tight   (+{queue_left:2d} actions)"
    return f"DEPLETED ({queue_left:+d} actions)"


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    logging.basicConfig(level=logging.WARNING)

    set_seed(cfg.seed)
    device = get_safe_torch_device(
        getattr(cfg.policy, "actor_device", None) or cfg.policy.device
    )
    cfg.policy.device = str(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    fps        = cfg.env.fps
    chunk_size = cfg.policy.chunk_size
    action_dim = next(iter(cfg.policy.output_features.values())).shape[0]

    print(f"\n{'='*78}")
    print(f"  MolmoAct2 inference benchmark")
    print(f"  device={device}  fps={fps}  chunk_size={chunk_size}  action_dim={action_dim}")
    print(f"  Steady-state budget: mean < {chunk_size / (2.0 * fps):.2f}s")
    print(f"{'='*78}")

    # --- Load policy + preprocessors ----------------------------------------
    print("Loading policy...")
    from lerobot.rl.rl_molmoact2_trainer import MolmoAct2Trainer
    trainer = MolmoAct2Trainer()
    policy = trainer.make_policy(cfg).to(device).eval()
    preprocessor, _ = trainer.make_processors(cfg)
    print("Done.\n")

    # --- Build one preprocessed batch (reused across all sweeps) -------------
    img_shape = list(cfg.policy.input_features["observation.images.wrist"].shape)
    dummy_obs = {
        "observation.images.wrist": torch.randint(0, 256, img_shape, dtype=torch.uint8),
        "observation.images.top":   torch.randint(0, 256, img_shape, dtype=torch.uint8),
        "observation.state":        torch.zeros(action_dim),
    }
    batch = trainer.build_inference_batch(
        dummy_obs, cfg.policy.task, cfg, preprocessor=preprocessor
    )
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # Simulate mid-episode leftover (half chunk of normalised actions)
    prev_actions = torch.randn(
        chunk_size // 2, action_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    # --- Sweep ---------------------------------------------------------------
    hdr = f"{'steps':>5}  {'exec_h':>6}  {'mean':>7}  {'p50':>7}  {'p95':>7}  {'max':>7}  budget"
    print(hdr)
    print("-" * len(hdr))

    for num_steps in NUM_STEPS_LIST:
        policy.config.num_inference_steps = num_steps

        for exec_h in EXEC_HORIZON_LIST:

            def _run():
                with torch.no_grad():
                    policy.predict_action_chunk(
                        batch,
                        inference_delay=num_steps,
                        prev_chunk_left_over=prev_actions,
                        execution_horizon=exec_h,
                    )

            for _ in range(N_WARMUP):
                _run()
            if device.type == "cuda":
                torch.cuda.synchronize()

            latencies = []
            for _ in range(N_MEASURE):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _run()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)

            arr = np.array(latencies)
            print(
                f"{num_steps:>5}  {exec_h:>6}  "
                f"{arr.mean():.3f}s  {np.percentile(arr, 50):.3f}s  "
                f"{np.percentile(arr, 95):.3f}s  {arr.max():.3f}s  "
                f"{_budget_label(arr.mean(), fps, chunk_size)}"
            )

        print()

    print(f"{'='*78}")


if __name__ == "__main__":
    main()
