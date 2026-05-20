#!/usr/bin/env python
"""
Generic standalone VLA inference runner.

Inference is RTC-only. RTC subsumes chunk-queue behavior while keeping one
action-queue implementation for latency-aware rollout.

Usage:
    python -m lerobot.rl.inference_async \
        --config_path lerobot/src/lerobot/rl/config_rl.yaml
"""
from __future__ import annotations

import logging
import os
import signal

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.rl_trainer import Trainer
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

import lerobot.rl.rl_molmoact2  # noqa: F401 - registers MolmoAct2RLConfig
import lerobot.rl.rl_pi05  # noqa: F401 - registers PI05RLConfig

logger = logging.getLogger(__name__)


def _configured_runtime(cfg) -> str:
    return (
        getattr(cfg, "inference_runtime", None)
        or getattr(cfg.policy, "inference_runtime", None)
        or getattr(cfg, "actor_runtime", None)
        or getattr(cfg.policy, "actor_runtime", None)
        or "rtc"
    )


def act_with_policy_async_vla(
    cfg,
    shutdown_event,
    parameters_queue=None,  # kept for old call sites; RTC inference is standalone
    transitions_queue=None,
    interactions_queue=None,
) -> None:
    _ = parameters_queue, transitions_queue, interactions_queue

    runtime = _configured_runtime(cfg)
    if runtime != "rtc":
        raise ValueError(
            f"inference_async only supports inference_runtime='rtc'; got {runtime!r}. "
            "The legacy non-RTC runtime has been removed."
        )

    from lerobot.rl.rtc_actor_runtime import act_with_policy_rtc_inference

    trainer = Trainer.for_config(cfg)
    act_with_policy_rtc_inference(
        cfg=cfg,
        trainer=trainer,
        shutdown_event=shutdown_event,
    )


@parser.wrap()
def actor_vla_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    log_dir = os.path.join(cfg.output_dir or "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    init_logging(log_file=os.path.join(log_dir, f"actor_{cfg.job_name}.log"), display_pid=False)
    logger.info("[INFERENCE] Generic RTC inference starting.")

    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    def _sigint(sig, frame):
        logger.info("[INFERENCE] SIGINT - shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _sigint)

    try:
        act_with_policy_async_vla(
            cfg=cfg,
            shutdown_event=shutdown_event,
        )
    finally:
        shutdown_event.set()
        logger.info("[INFERENCE] inference_cli finished.")


if __name__ == "__main__":
    actor_vla_cli()
