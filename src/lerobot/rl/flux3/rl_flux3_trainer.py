"""Trainer seam for flux3_remote: inference methods only (rtc_actor_runtime), training raises."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lerobot.rl.flux3.rl_flux3 import Flux3RemoteConfig, Flux3RemotePolicy
from lerobot.rl.rl_trainer import Trainer


class Flux3RemoteTrainer(Trainer):
    def make_processors(self, cfg, dataset=None, is_main_process: bool = True) -> tuple:
        # The server owns the checkpoint's processors; the runtime gets raw observations in and absolute degrees out.
        return None, None

    def make_policy(self, cfg) -> nn.Module:
        if not isinstance(cfg.policy, Flux3RemoteConfig):
            raise TypeError(f"Flux3RemoteTrainer needs a flux3_remote policy config, got {type(cfg.policy).__name__}")
        return Flux3RemotePolicy(cfg.policy)

    def build_inference_batch(self, observation: dict, task_str: str, cfg, **context) -> dict[str, Any]:
        return {**observation, "task": task_str}

    def critic_value_for_logging(self, policy, transition, device, cfg) -> float | None:
        return None

    # -- training interface: not available through the fork -------------------------------------------
    def _no_training(self, *_, **__):
        raise NotImplementedError("flux3_remote is inference only; train with upstream lerobot_train (.venv-flux3)")

    freeze_model = get_optimizer_groups = update_critic = compute_advantage = _no_training
    build_training_batch = actor_forward = update_actor = update_target_networks = _no_training
    push_weights = _no_training
