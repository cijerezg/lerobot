#!/usr/bin/env python
"""Spatial memorization probe using causal Jacobian attention maps.

This is the action-Jacobian variant of :mod:`lerobot.probes.spatial_memorization_attention`.
It reuses the same aggregation/rendering code, but asks the policy adapter for
``capture_attention(..., requires_grad=True)`` so the aggregated maps are
A * |grad(A)| instead of raw softmax attention.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.spatial_memorization_attention import run_jacobian
from lerobot.probes.base import ProbablePolicy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeSpatialMemorizationActionJacobianConfig(TrainRLServerPipelineConfig):
    """Tunables live under ``cfg.probe_parameters``."""


def run(adapter, primary_dataset, cfg, output_dir):
    return run_jacobian(adapter, primary_dataset, cfg, output_dir)


@parser.wrap()
def probe_cli(cfg: ProbeSpatialMemorizationActionJacobianConfig):
    init_logging()
    p = cfg.probe_parameters
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "spatial_memorization_action_jacobian")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    from lerobot.datasets.factory import make_dataset
    primary_dataset = make_dataset(cfg)
    primary_dataset.delta_timestamps = None
    primary_dataset.delta_indices = None

    logging.info("Loading policy adapter ...")
    adapter = ProbablePolicy.for_config(cfg, device, dataset=primary_dataset)
    run(adapter, primary_dataset, cfg, output_dir)
    logging.info(f"Done. Output in {output_dir}/")


if __name__ == "__main__":
    probe_cli()
