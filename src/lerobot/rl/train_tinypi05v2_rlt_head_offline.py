#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline RLT actor/critic trainer for tinypi05v2.

Mirrors `train_tinypi05_rlt_head_offline.py`, swapping in the tinypi05v2 RLT
policy. The persisted `RLTReplayBuffer` format is policy-agnostic: it stores
the already-encoded `rl_token`, proprio, reference_chunk, executed_chunk and
next-state counterparts, so the loop only touches the tiny actor/critic heads.
"""

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import draccus
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.rl.rlt_buffer import RLTReplayBuffer
from lerobot.rl.rlt_pi05 import (
    rlt_actor_loss,
    rlt_critic_loss,
    save_rlt_head_checkpoint,
    soft_update_rlt_target,
)
from lerobot.rl.rlt_tinypi05v2 import TinyPI05V2RLTConfig, TinyPI05V2RLTPolicy
from lerobot.utils.utils import init_logging


@dataclass
class TrainTinyPI05V2RLTHeadOfflineConfig:
    policy_path: str = (
        "outputs/train/2026-05-02/18-34-57_tinypi05_so101_pickplace_160_bs64_anchor/checkpoints/092000/pretrained_model"
    )
    replay_buffer_path: str = "outputs/rlt_online/rlt_online_replay.pt"
    output_dir: str = "outputs/tinypi05v2_rlt_offline_head"
    device: str = "cuda"
    seed: int = 0

    steps: int = 10_000
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    discount: float = 0.985
    target_update_tau: float = 0.005
    grad_clip_norm: float | None = 20.0

    rlt_actor_hidden_dims: list[int] | None = None
    rlt_critic_hidden_dims: list[int] | None = None
    rlt_actor_hidden_dim: int = 256
    rlt_critic_hidden_dim: int = 256
    rlt_actor_residual_scale: float = 0.25
    rlt_num_critics: int = 4
    rlt_bc_beta: float = 0.1
    rlt_bc_action_weights: list[float] | None = None
    rlt_jerk_beta: float = 0.001
    rlt_reference_dropout_p: float = 0.5
    rlt_embedding_checkpoint: str | None = None
    rlt_head_checkpoint: str | None = None

    kl_sigma: float = 1.0
    log_freq: int = 50
    save_freq: int = 1_000

    wandb_enabled: bool = True
    wandb_project: str = "lerobot-rlt"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None


def gaussian_kl_to_reference(actor_actions: Tensor, reference_actions: Tensor, sigma: float = 1.0) -> Tensor:
    """KL between two fixed-variance diagonal Gaussians centered at actor/ref actions."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 0.5 * (actor_actions - reference_actions).pow(2).mean() / (float(sigma) ** 2)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clip_grad_norm(parameters: Any, max_norm: float | None) -> float:
    if max_norm is None:
        return 0.0
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(max_norm))
    return float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)


def _scalar_stats(stats: dict[str, Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu())
        for key, value in stats.items()
        if isinstance(value, Tensor) and value.numel() == 1
    }


def _build_policy(
    cfg: TrainTinyPI05V2RLTHeadOfflineConfig, replay: RLTReplayBuffer
) -> TinyPI05V2RLTPolicy:
    sample = replay.samples()[0]
    token_dim = int(sample.rl_token.shape[-1])
    chunk_size = int(sample.reference_chunk.shape[-2])

    base_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    rlt_cfg = TinyPI05V2RLTConfig.from_base_config(
        base_cfg,
        device=cfg.device,
        rlt_enabled=True,
        rlt_embedding_checkpoint=cfg.rlt_embedding_checkpoint,
        rlt_head_checkpoint=cfg.rlt_head_checkpoint,
        rlt_chunk_size=chunk_size,
        rlt_token_dim=token_dim,
        rlt_actor_hidden_dims=cfg.rlt_actor_hidden_dims,
        rlt_critic_hidden_dims=cfg.rlt_critic_hidden_dims,
        rlt_actor_hidden_dim=cfg.rlt_actor_hidden_dim,
        rlt_critic_hidden_dim=cfg.rlt_critic_hidden_dim,
        rlt_actor_residual_scale=cfg.rlt_actor_residual_scale,
        rlt_num_critics=cfg.rlt_num_critics,
        rlt_bc_beta=cfg.rlt_bc_beta,
        rlt_bc_action_weights=cfg.rlt_bc_action_weights,
        rlt_jerk_beta=cfg.rlt_jerk_beta,
        rlt_reference_dropout_p=cfg.rlt_reference_dropout_p,
        pi05_checkpoint=cfg.policy_path,
        rtc_config=None,
    )
    policy = TinyPI05V2RLTPolicy.from_pretrained(cfg.policy_path, config=rlt_cfg, strict=False)
    policy.to(cfg.device)
    policy.model.eval()
    policy.rlt_critic_target.requires_grad_(False)
    policy.rlt_critic_target.eval()
    return policy


def _init_wandb(cfg: TrainTinyPI05V2RLTHeadOfflineConfig):
    if not cfg.wandb_enabled:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config=asdict(cfg),
    )


def _log_metrics(wandb_run, step: int, metrics: dict[str, float]) -> None:
    logging.info(
        "step=%d actor_loss=%.6f critic_loss=%.6f kl_to_reference=%.6f "
        "actor_q=%.6f pred_q=%.6f action_dev_rms=%.6f",
        step,
        metrics["actor_loss"],
        metrics["critic_loss"],
        metrics["kl_to_reference"],
        metrics.get("actor_q_mean", 0.0),
        metrics.get("pred_q_mean", 0.0),
        metrics.get("action_deviation_rms", 0.0),
    )
    if wandb_run is not None:
        wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=step)


def _save_head(
    policy: TinyPI05V2RLTPolicy,
    output_dir: Path,
    step: int,
    cfg: TrainTinyPI05V2RLTHeadOfflineConfig,
) -> None:
    path = output_dir / f"rlt_head_step_{step:06d}.pt"
    latest_path = output_dir / "rlt_head_latest.pt"
    config = asdict(cfg)
    save_rlt_head_checkpoint(policy, path, step=step, config=config)
    save_rlt_head_checkpoint(policy, latest_path, step=step, config=config)
    logging.info("Saved offline tinypi05v2 RLT head checkpoint to %s", path)


@draccus.wrap()
def train_tinypi05v2_rlt_head_offline(cfg: TrainTinyPI05V2RLTHeadOfflineConfig) -> None:
    init_logging()
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = RLTReplayBuffer.load(cfg.replay_buffer_path)
    if len(replay) < cfg.batch_size:
        raise ValueError(f"Replay buffer has {len(replay)} samples, but batch_size={cfg.batch_size}")

    policy = _build_policy(cfg, replay)
    actor_optimizer = torch.optim.AdamW(policy.rlt_actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.AdamW(policy.rlt_critic.parameters(), lr=cfg.critic_lr)
    wandb_run = _init_wandb(cfg)

    logging.info(
        "Starting offline tinypi05v2 RLT head training | replay=%s | samples=%d | device=%s | steps=%d",
        cfg.replay_buffer_path,
        len(replay),
        cfg.device,
        cfg.steps,
    )

    for step in range(1, cfg.steps + 1):
        batch = replay.sample(cfg.batch_size, device=cfg.device)

        policy.rlt_actor.train()
        policy.rlt_critic.train()

        critic_loss, critic_stats = rlt_critic_loss(policy, batch, cfg.discount, return_stats=True)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = _clip_grad_norm(policy.rlt_critic.parameters(), cfg.grad_clip_norm)
        critic_optimizer.step()

        actor_loss, actor_stats = rlt_actor_loss(policy, batch, return_stats=True)
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = _clip_grad_norm(policy.rlt_actor.parameters(), cfg.grad_clip_norm)
        actor_optimizer.step()

        soft_update_rlt_target(policy, cfg.target_update_tau)

        if step % cfg.log_freq == 0 or step == 1:
            with torch.no_grad():
                actor_actions = policy.rlt_actor(
                    batch["rl_token"],
                    batch["proprio"],
                    batch["reference_chunk"],
                )
                kl_to_reference = gaussian_kl_to_reference(
                    actor_actions,
                    batch["reference_chunk"],
                    sigma=cfg.kl_sigma,
                )
            metrics = {
                **_scalar_stats(critic_stats),
                **_scalar_stats(actor_stats),
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "kl_to_reference": float(kl_to_reference.detach().cpu()),
                "critic_grad_norm": critic_grad_norm,
                "actor_grad_norm": actor_grad_norm,
                "replay_size": float(len(replay)),
            }
            _log_metrics(wandb_run, step, metrics)

        if step % cfg.save_freq == 0 or step == cfg.steps:
            _save_head(policy, output_dir, step, cfg)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train_tinypi05v2_rlt_head_offline()
