"""Trainer seam for flux3_rl: finetune of the FLUX 3 Action base in the offline loop (actor only)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F_vision
from torch import nn

from lerobot.policies.flux3 import Flux3Policy, make_flux3_pre_post_processors
from lerobot.rl.flux3 import windows
from lerobot.rl.flux3.inference import COMMAND_HISTORY, Flux3RuntimePolicy, chunk_postprocessor, state_window
from lerobot.rl.flux3.rl_flux3 import SPEED_LEVELS, render_caption
from lerobot.rl.offline_dataset_utils import (
    _idx_to_subtask_name,
    _install_buffer_column,
    get_offline_dataset_sources,
    resolve_task_strings,
)
from lerobot.rl.rl_trainer import Trainer
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.transition import move_transition_to_device

# Buffer columns of one frame's training window.
STATE_WINDOW = "flux3_state_window"  # [n_obs_steps, 21]
COMMAND_WINDOW = "flux3_command_window"  # [n_obs_steps + chunk_size, 7], absolute commands
MISTAKE = "flux3_mistake"
SPEED = "flux3_speed"


class Flux3Trainer(Trainer):
    def make_policy(self, cfg) -> nn.Module:
        roots = [source.root for source in get_offline_dataset_sources(cfg)]
        stats = windows.normalization_stats(roots, cfg.policy.delta_absolute_dims)
        cfg.policy.normalization_stats = stats
        for stream, q in stats.items():
            logging.info(f"[FLUX3] {stream} q01 {np.round(q['q01'], 2).tolist()}")
            logging.info(f"[FLUX3] {stream} q99 {np.round(q['q99'], 2).tolist()}")
        return Flux3Policy(cfg.policy).to(cfg.policy.device)

    def make_processors(self, cfg, dataset=None, is_main_process: bool = True) -> tuple:
        return make_flux3_pre_post_processors(cfg.policy)

    def freeze_model(self, policy: nn.Module, cfg) -> None:
        pass  # full finetune; the video VAE and the text encoder are not parameters of the policy

    def get_optimizer_groups(self, policy: nn.Module, cfg) -> list[dict]:
        # one optimizer, two parameter groups: the trunk, and the fresh heads at their own rate
        return [{"name": "policy", "lr": cfg.policy.optimizer_lr, "params": policy.get_optim_params()}]

    def prepare_buffer(self, buffer, cfg) -> None:
        policy_cfg = cfg.policy
        root = buffer.dataset.root
        frames = windows.read_frames(root)
        n = len(frames)
        speed = windows.speed_levels(root, n)
        subtask = buffer.complementary_info["subtask_index"][:n]
        if (speed < 0).any() or (subtask < 0).any():
            raise ValueError(
                f"{root}: {int((subtask < 0).sum())} frames without a subtask, {int((speed < 0).sum())} without a speed level"
            )
        state_window = windows.windows(windows.states(frames), frames, policy_cfg.obs_offsets)
        command_window = windows.windows(
            np.stack(frames[ACTION]).astype(np.float32),
            frames,
            range(-policy_cfg.n_obs_steps, policy_cfg.chunk_size),
        )
        _install_buffer_column(buffer, STATE_WINDOW, torch.as_tensor(state_window), fill_value=0)
        _install_buffer_column(buffer, COMMAND_WINDOW, torch.as_tensor(command_window), fill_value=0)
        _install_buffer_column(buffer, MISTAKE, torch.as_tensor(windows.mistakes(root, n)), fill_value=False)
        _install_buffer_column(buffer, SPEED, torch.as_tensor(speed))
        buffer.use_drq = False
        # the future frames the video target keeps: +1, +1 + stride, ...
        future = [1 + k for k in range(0, policy_cfg.chunk_size, policy_cfg.future_video_stride)]
        buffer.configure_future_images(future, policy_cfg.camera_order)

    def build_training_batch(
        self, raw_batch: dict, observations: dict, actions, advantage, preprocessor, dataset, cfg
    ) -> dict[str, Any]:
        info = raw_batch["complementary_info"]
        batch_size = len(info[SPEED])
        tasks = resolve_task_strings(raw_batch, dataset, cfg.policy.task, batch_size)
        subtasks = _idx_to_subtask_name(dataset)
        batch = {
            "task": [
                render_caption(task, subtasks[subtask], mistake, SPEED_LEVELS[speed])
                for task, subtask, mistake, speed in zip(
                    tasks,
                    info["subtask_index"].tolist(),
                    info[MISTAKE].tolist(),
                    info[SPEED].tolist(),
                    strict=True,
                )
            ],
            OBS_STATE: info[STATE_WINDOW],
            ACTION: info[COMMAND_WINDOW],
        }
        for camera in cfg.policy.camera_order:
            # the observations oldest first, then the future frames: [B, T, 3, H, W] in [0, 1]
            video = torch.cat(
                [observations[f"history.{camera}"], observations[camera][:, None], info[f"future.{camera}"]],
                dim=1,
            )
            batch[camera] = video.float() / 255
        batch = preprocessor(batch)
        # a window that reaches past the end of its episode is left out of the loss
        batch["action_is_pad"] = ~info["future_visual_valid"]
        return batch

    def actor_forward(self, policy: nn.Module, batch: dict, advantage, cfg) -> dict[str, Any]:
        loss, metrics = policy.forward(batch)
        return {"loss_actor": loss, **metrics}

    def update_actor(
        self, policy, optimizers, online_iter, offline_iter, preprocessor, dataset, device, cfg, **kwargs
    ) -> dict[str, Any]:
        runtime = kwargs["training_runtime"]
        steps = int(cfg.policy.gradient_accumulation_steps)
        optimizer = optimizers["policy"]
        optimizer.zero_grad()
        totals = {"loss_actor": 0.0, "video_mse": 0.0, "action_mse": 0.0, "n_valid_windows": 0.0}
        for _ in range(steps):
            raw = move_transition_to_device(next(online_iter), device)
            batch = self.build_training_batch(raw, raw["state"], None, None, preprocessor, dataset, cfg)
            out = self.actor_forward(policy, batch, None, cfg)
            runtime.backward(out["loss_actor"] / steps)
            totals["loss_actor"] += out.pop("loss_actor").item() / steps
            for key in out:
                totals[key] += float(out[key]) / steps
        if kwargs["optimization_step"] % cfg.log_freq == 0:
            logging.info("[FLUX3] " + "  ".join(f"{key} {value:.4f}" for key, value in totals.items()))
        params = [p for p in runtime.unwrap_model(policy).parameters() if p.requires_grad]
        totals["actor_grad_norm"] = runtime.clip_grad_norm_(params, cfg.policy.optimizer_grad_clip_norm).item()
        optimizer.step()
        return totals

    # -- not part of the actor-only finetune ----------------------------------------------------------
    def update_critic(self, *_, **__):
        raise NotImplementedError("flux3_rl has no critic; set skip_critic: true")

    def update_target_networks(self, policy: nn.Module) -> None:
        pass

    # -- run time (rl/rtc_actor_runtime.py) -------------------------------------------------------------
    def make_inference_policy(self, cfg) -> nn.Module:
        return Flux3RuntimePolicy.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)

    def make_inference_processors(self, cfg) -> tuple:
        # the quantiles of the training roots, saved with the checkpoint's config
        saved = json.loads((Path(cfg.policy.pretrained_path) / "config.json").read_text())
        cfg.policy.normalization_stats = saved["normalization_stats"]
        preprocessor, postprocessor = self.make_processors(cfg)
        return preprocessor, chunk_postprocessor(postprocessor)

    def build_inference_batch(self, observation: dict, task_str: str, cfg, **context) -> dict[str, Any]:
        """The training batch's inputs from the runtime's observation (batch of 1, history windows oldest first).

        context: preprocessor, subtask, velocities / torques (raw readings of the episode's last control steps),
        last_action (the command before now, None at the start of an episode).
        """
        policy_cfg = cfg.policy
        velocities = context["velocities"]
        torques = context["torques"]
        if not velocities:  # warm-up, before the first reading
            velocities = torques = [[0.0] * policy_cfg.action_dim]
        positions = torch.cat([observation[f"history.{OBS_STATE}"], observation[OBS_STATE][:, None]], dim=1)
        state = state_window(positions[0].float().cpu().numpy(), velocities, torques, policy_cfg.obs_offsets)
        last_action = context["last_action"]
        if last_action is None:  # no command yet: the arm holds its position
            last_action = observation[OBS_STATE]
        batch = {
            "task": [render_caption(task_str, context["subtask"], False, policy_cfg.eval_speed)],
            OBS_STATE: torch.as_tensor(state)[None],
            COMMAND_HISTORY: last_action.float().reshape(1, 1, -1).repeat(1, policy_cfg.n_obs_steps, 1),
        }
        for camera in policy_cfg.camera_order:
            video = torch.cat([observation[f"history.{camera}"], observation[camera][:, None]], dim=1)
            # the buffer's resize (ReplayBuffer._prepare_image_for_storage)
            batch[camera] = F_vision.resize(video[0].float(), list(policy_cfg.image_storage_size))[None]
        return context["preprocessor"](batch)

    def critic_value_for_logging(self, policy, transition, device, cfg) -> float | None:
        return None

    def push_weights(self, policy: nn.Module, parameters_queue) -> None:
        from lerobot.transport.utils import state_to_bytes
        from lerobot.utils.transition import move_state_dict_to_device

        trainable = {name: param for name, param in policy.named_parameters() if param.requires_grad}
        parameters_queue.put(state_to_bytes({"policy": move_state_dict_to_device(trainable, "cpu")}))
