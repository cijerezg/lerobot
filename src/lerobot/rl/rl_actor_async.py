#!/usr/bin/env python
"""
Generic async actor for distributed HILSerl online RL training.

Works for any model registered with the Trainer ABC (MolmoAct2, PI05, …).
RTC is the runtime: model-specific observation preprocessing is isolated in
Trainer.build_inference_batch(), while rtc_actor_runtime owns ActionQueue,
latency-aware replanning, intervention resets, and smooth execution.

Three gRPC background threads (from actor.py):
  receive_policy    — pulls updated weights from the learner.
  send_transitions  — forwards completed episode transitions to the learner.
  send_interactions — forwards episode stats for W&B logging.

Usage:
    python -m lerobot.rl.rl_actor_async \
        --config_path lerobot/src/lerobot/rl/config_rl.yaml
"""
import logging
import os
from threading import Thread

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.actor import (
    establish_learner_connection,
    learner_service_client,
    receive_policy,
    send_interactions,
    send_transitions,
)
from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.rtc_actor_runtime import act_with_policy_rtc
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging

import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
import lerobot.rl.pi05.rl_pi05            # noqa: F401 — registers PI05RLConfig

logger = logging.getLogger(__name__)


# ── CLI entry point ───────────────────────────────────────────────────────────


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    """Entry point for distributed online RL actor."""
    cfg.validate()

    log_dir = os.path.join(cfg.output_dir or "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    init_logging(log_file=os.path.join(log_dir, f"actor_{cfg.job_name}.log"), display_pid=False)
    logger.info("[ACTOR] Starting.")

    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    trainer: Trainer = Trainer.for_config(cfg)

    # gRPC transport: receive weights from learner, send transitions and interactions.
    alc = cfg.policy.actor_learner_config
    learner_client, grpc_channel = learner_service_client(
        host=alc.learner_host, port=alc.learner_port
    )
    logger.info("[ACTOR] Establishing connection with Learner…")
    if not establish_learner_connection(learner_client, shutdown_event):
        logger.error("[ACTOR] Could not connect to Learner. Exiting.")
        return

    from torch.multiprocessing import Queue
    parameters_queue: Queue = Queue(maxsize=2)
    transitions_queue: Queue = Queue()
    interactions_queue: Queue = Queue()

    for target, name, args in [
        (receive_policy,    "recv_policy",   (cfg, parameters_queue,  shutdown_event, grpc_channel)),
        (send_transitions,  "send_trans",    (cfg, transitions_queue, shutdown_event, grpc_channel)),
        (send_interactions, "send_interact", (cfg, interactions_queue, shutdown_event, grpc_channel)),
    ]:
        Thread(target=target, args=args, daemon=True, name=name).start()

    try:
        act_with_policy_rtc(
            cfg=cfg,
            trainer=trainer,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
        )
    finally:
        shutdown_event.set()
        for q in (parameters_queue, transitions_queue, interactions_queue):
            q.close()
            q.cancel_join_thread()
        logger.info("[ACTOR] actor_cli finished.")


if __name__ == "__main__":
    actor_cli()
