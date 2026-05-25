#!/usr/bin/env python
"""
Learner script for distributed VLA online RL training.

Mirrors learner_pi05.py but dispatches through Trainer.for_config() so the same
script works for any model registered with the Trainer ABC (MolmoAct2, PI05, etc.).

Architecture:
  - Communication thread: gRPC server receives transitions / sends weights
    (from learner.start_learner).
  - Main thread: training loop.

Queues:
  transition_queue          — actor → learner: episode transitions
  interaction_message_queue — actor → learner: episode stats for W&B
  parameters_queue          — learner → actor: updated weights

Usage:
    python -m lerobot.rl.rl_learner \
        --config_path lerobot/src/lerobot/rl/config_rl.yaml
"""
from __future__ import annotations

import logging
import json
import os
import shutil
import time
from pprint import pformat
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from termcolor import colored
from PIL import Image
from torch import nn
from torch.multiprocessing import Process, Queue

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.offline_dataset_utils import load_additional_offline_datasets
from lerobot.rl.learner import (
    check_nan_in_transition,
    handle_resume_logic,
    load_training_state,
    log_training_info,
    process_interaction_messages,
    save_training_checkpoint,
    start_learner,
)
from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.utils import save_video_with_critic_overlay
from lerobot.rl.weight_anchor import build_weight_anchors, apply_weight_anchors
from lerobot.transport.utils import bytes_to_transitions
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


# ── Optimizer builder ─────────────────────────────────────────────────────────


def _build_optimizers(groups: list[dict]) -> dict[str, torch.optim.Optimizer]:
    """Convert trainer.get_optimizer_groups() list into named Adam optimizers."""
    return {g["name"]: torch.optim.Adam(g["params"], lr=g["lr"]) for g in groups}


# ── Thread/process helpers ────────────────────────────────────────────────────


def _use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    concurrency = getattr(cfg.policy, "concurrency", None)
    if concurrency is None:
        return True  # VLA models default to threads; no concurrency config field
    return concurrency.learner == "threads"


# ── Entry points ──────────────────────────────────────────────────────────────


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not _use_threads(cfg):
        mp.set_start_method("spawn")
    train(cfg, job_name=cfg.job_name)
    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None) -> None:
    cfg.validate()
    if job_name is None:
        job_name = cfg.job_name
    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")
    init_logging(log_file=log_file, display_pid=not _use_threads(cfg))
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    cfg = handle_resume_logic(cfg)
    set_seed(seed=cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = _use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=not is_threaded).shutdown_event

    start_learner_threads(cfg=cfg, wandb_logger=wandb_logger, shutdown_event=shutdown_event)


# ── Thread management ─────────────────────────────────────────────────────────


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event,
) -> None:
    transition_queue: Queue = Queue()
    interaction_message_queue: Queue = Queue()
    parameters_queue: Queue = Queue()

    concurrency_entity = Thread if _use_threads(cfg) else Process

    communication_thread = concurrency_entity(
        target=start_learner,
        args=(parameters_queue, transition_queue, interaction_message_queue, shutdown_event, cfg),
        daemon=True,
    )
    communication_thread.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training loop stopped")

    for q in (transition_queue, interaction_message_queue, parameters_queue):
        q.close()
    communication_thread.join()
    for q in (transition_queue, interaction_message_queue, parameters_queue):
        q.cancel_join_thread()
    logging.info("[LEARNER] Queues closed, communication thread joined")


# ── Main training loop ────────────────────────────────────────────────────────


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
) -> None:
    """Receive transitions from the actor, update the policy, push weights back."""
    # ── Config fields ─────────────────────────────────────────────────────────
    device_name: str = getattr(cfg.policy, "learner_device", None) or cfg.policy.device
    device = get_safe_torch_device(try_device=device_name, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)

    online_step_before_learning: int = int(getattr(cfg.policy, "online_step_before_learning", 100))
    utd_ratio: int = int(getattr(cfg.policy, "utd_ratio", 1))
    policy_update_freq: int = int(getattr(cfg.policy, "policy_update_freq", 1))
    critic_warmup_steps: int = int(getattr(cfg.policy, "critic_warmup_steps", 0))
    async_prefetch: bool = bool(getattr(cfg.policy, "async_prefetch", False))

    # Prefer explicit weights_push_interval field; fall back to nested actor_learner_config
    # for PI05 backward compatibility, then default to 180 s.
    _alc = getattr(cfg.policy, "actor_learner_config", None)
    weights_push_interval: float = float(
        getattr(cfg.policy, "weights_push_interval", None)
        or getattr(_alc, "policy_parameters_push_frequency", 180)
    )

    log_freq: int = cfg.log_freq
    save_freq: int = cfg.save_freq
    episode_save_freq: int = cfg.episode_save_freq
    saving_checkpoint: bool = cfg.save_checkpoint
    online_steps: int = int(getattr(cfg.policy, "online_steps", 1_000_000))
    skip_critic: bool = bool(getattr(cfg, "skip_critic", False))
    fps: int = cfg.env.fps

    if not _use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        init_logging(os.path.join(log_dir, f"learner_train_{os.getpid()}.log"), display_pid=True)

    # ── Trainer + policy ──────────────────────────────────────────────────────
    trainer: Trainer = Trainer.for_config(cfg)

    original_device = cfg.policy.device
    cfg.policy.device = device_name
    policy: nn.Module = trainer.make_policy(cfg)
    cfg.policy.device = original_device

    policy.train()

    if not skip_critic:
        init_critic = getattr(policy, "init_critic", None)
        if callable(init_critic):
            init_critic()

    trainer.freeze_model(policy, cfg)
    log_training_info(cfg=cfg, policy=policy)

    # ── Optimizers ────────────────────────────────────────────────────────────
    optimizers = _build_optimizers(trainer.get_optimizer_groups(policy, cfg))
    weight_anchors = build_weight_anchors(
        optimizers=optimizers,
        alpha=float(getattr(cfg.policy, "anchor_alpha", 0.0)),
        every_n_steps=int(getattr(cfg.policy, "anchor_every_n_steps", 0)),
        targets=list(getattr(cfg.policy, "anchor_targets", [])),
    )
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    # ── Processors ───────────────────────────────────────────────────────────
    offline_dataset = None
    dataset_repo_id = getattr(cfg.dataset, "repo_id", None) if cfg.dataset is not None else None

    preprocessor, postprocessor = trainer.make_processors(cfg, dataset=offline_dataset, is_main_process=True)
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    # ── Replay buffers ────────────────────────────────────────────────────────
    replay_buffer = _init_online_buffer(cfg, device, storage_device)
    replay_buffer.reward_normalization_constant = float(
        getattr(cfg.policy, "reward_normalization_constant", 1.0)
    )
    replay_buffer.terminal_failure_reward = float(
        getattr(cfg.policy, "terminal_failure_reward", -10.0)
    )

    offline_replay_buffer = None
    batch_size = cfg.batch_size

    if cfg.dataset is not None:
        offline_replay_buffer = _init_offline_buffer(cfg, device, storage_device)
        offline_dataset = offline_replay_buffer.dataset
        load_additional_offline_datasets(
            cfg=cfg,
            offline_dataset=offline_dataset,
            offline_replay_buffer=offline_replay_buffer,
            storage_device=storage_device,
            is_main_process=True,
        )
        batch_size = batch_size // 2

    # ── Initial weight push ───────────────────────────────────────────────────
    trainer.push_weights(policy, parameters_queue)
    last_weights_pushed = time.time()

    # ── Training state ────────────────────────────────────────────────────────
    optimization_step: int = resume_optimization_step or 0
    interaction_step_shift: int = resume_interaction_step or 0
    online_iterator = None
    offline_iterator = None
    interaction_message = None
    last_save_episode = 0
    episode_counter = [0]

    logging.info("[LEARNER] Starting main training loop")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        process_transitions_vla(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            shutdown_event=shutdown_event,
            episode_counter=episode_counter,
            policy=policy,
            trainer=trainer,
            device=device_name,
            cfg=cfg,
        )

        if len(replay_buffer) < online_step_before_learning:
            continue

        # Lazily create iterators once buffer is warm.
        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size,
                async_prefetch=async_prefetch,
                queue_size=2,
                action_chunk_size=cfg.policy.n_action_steps,
            )
        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size,
                async_prefetch=async_prefetch,
                queue_size=2,
                action_chunk_size=cfg.policy.n_action_steps,
            )

        t0 = time.time()

        # ── Critic update (UTD−1 extra passes) ─────────────────────────────
        if not skip_critic:
            for _ in range(utd_ratio - 1):
                trainer.update_critic(
                    policy=policy,
                    optimizers=optimizers,
                    online_iter=online_iterator,
                    offline_iter=offline_iterator,
                    device=device_name,
                    cfg=cfg,
                    preprocessor=preprocessor,
                )
                trainer.update_target_networks(policy)

        # ── Actor + final critic pass ───────────────────────────────────────
        if skip_critic:
            training_infos = trainer.update_actor(
                policy=policy,
                optimizers=optimizers,
                online_iter=online_iterator,
                offline_iter=offline_iterator,
                preprocessor=preprocessor,
                dataset=offline_dataset,
                device=device_name,
                cfg=cfg,
            )
        else:
            training_infos = trainer.update_critic(
                policy=policy,
                optimizers=optimizers,
                online_iter=online_iterator,
                offline_iter=offline_iterator,
                device=device_name,
                cfg=cfg,
                preprocessor=preprocessor,
            )
            trainer.update_target_networks(policy)

            if optimization_step >= critic_warmup_steps and optimization_step % policy_update_freq == 0:
                actor_infos = trainer.update_actor(
                    policy=policy,
                    optimizers=optimizers,
                    online_iter=online_iterator,
                    offline_iter=offline_iterator,
                    preprocessor=preprocessor,
                    dataset=offline_dataset,
                    device=device_name,
                    cfg=cfg,
                )
                training_infos.update(actor_infos)

        apply_weight_anchors(weight_anchors, optimizers, optimization_step)

        # ── Logging ────────────────────────────────────────────────────────
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            trainer.log_metrics(
                training_infos=training_infos,
                step=optimization_step,
                wandb_logger=wandb_logger,
                _policy=policy,
            )

        step_hz = 1.0 / (time.time() - t0 + 1e-9)
        if wandb_logger:
            wandb_logger.log_dict(
                {"Optimization frequency loop [Hz]": step_hz, "Optimization step": optimization_step},
                mode="train",
                custom_step_key="Optimization step",
            )

        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] step {optimization_step}  {step_hz:.2f} Hz")

        optimization_step += 1

        # ── Checkpoint ────────────────────────────────────────────────────
        if saving_checkpoint and (
            optimization_step % save_freq == 0 or optimization_step == online_steps
        ):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )

        # ── Online buffer save ─────────────────────────────────────────────
        current_episode = episode_counter[0]
        if current_episode > 0 and current_episode >= last_save_episode + episode_save_freq:
            _save_online_buffer(cfg, replay_buffer, fps, current_episode, optimization_step)
            last_save_episode = current_episode

        # ── Termination ───────────────────────────────────────────────────
        if optimization_step >= online_steps:
            logging.info("[LEARNER] Reached maximum online steps. Stopping.")
            break

        # ── Push weights ──────────────────────────────────────────────────
        if time.time() - last_weights_pushed > weights_push_interval:
            trainer.push_weights(policy, parameters_queue)
            last_weights_pushed = time.time()


# ── Replay buffer helpers ─────────────────────────────────────────────────────


def process_transitions_vla(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    shutdown_event,
    episode_counter: list,
    policy: nn.Module | None = None,
    trainer: Trainer | None = None,
    device: str | None = None,
    cfg: TrainRLServerPipelineConfig | None = None,
) -> None:
    """Drain the transition queue and add each episode's transitions to the replay buffer."""
    while not transition_queue.empty() and not shutdown_event.is_set():
        raw = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=raw)
        episode_counter[0] += 1

        log_ctx = _maybe_start_logging_episode(cfg, episode_counter[0])

        for step_idx, transition in enumerate(transition_list):
            if log_ctx is not None and policy is not None and trainer is not None and device is not None:
                _log_transition_for_episode(
                    transition=transition,
                    step_idx=step_idx,
                    log_ctx=log_ctx,
                    policy=policy,
                    trainer=trainer,
                    device=device,
                    cfg=cfg,
                )

            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN in transition, skipping")
                continue

            # Pad or truncate action dim if buffer was initialised with a different shape.
            if replay_buffer.initialized:
                buffer_action_dim = replay_buffer.actions.shape[-1]
                incoming = transition[ACTION]
                incoming_dim = incoming.shape[-1]
                if incoming_dim != buffer_action_dim:
                    if incoming_dim < buffer_action_dim:
                        pad = torch.zeros(
                            (*incoming.shape[:-1], buffer_action_dim - incoming_dim),
                            dtype=incoming.dtype,
                            device=incoming.device,
                        )
                        transition[ACTION] = torch.cat([incoming, pad], dim=-1)
                    else:
                        transition[ACTION] = incoming[..., :buffer_action_dim]

            replay_buffer.add(**transition)

        if log_ctx is not None and policy is not None:
            _finish_logging_episode(log_ctx, episode_counter[0])


def _maybe_start_logging_episode(
    cfg: TrainRLServerPipelineConfig | None,
    episode: int,
) -> dict | None:
    if cfg is None:
        return None
    freq = int(getattr(cfg, "episode_logging_freq", 0) or 0)
    if freq <= 0 or episode % freq != 0:
        return None

    logging.info(f"[LEARNER] Starting logging episode {episode}")
    log_dir = os.path.join(cfg.output_dir, "logging_episodes", f"episode_{episode:06d}")
    os.makedirs(log_dir, exist_ok=True)
    return {
        "log_dir": log_dir,
        "critic_values": [],
        "camera_names": list(getattr(cfg, "video_logging_cameras", ["top", "side"])),
        "critic_failed": False,
    }


def _log_transition_for_episode(
    transition: dict,
    step_idx: int,
    log_ctx: dict,
    policy: nn.Module,
    trainer: Trainer,
    device: str,
    cfg: TrainRLServerPipelineConfig | None,
) -> None:
    _save_transition_images(
        transition=transition,
        log_dir=log_ctx["log_dir"],
        step_idx=step_idx,
        camera_names=log_ctx["camera_names"],
    )

    if cfg is None or log_ctx["critic_failed"]:
        return

    try:
        with torch.no_grad():
            value = trainer.critic_value_for_logging(
                policy=policy,
                transition=transition,
                device=device,
                cfg=cfg,
            )
    except Exception as exc:
        log_ctx["critic_failed"] = True
        logging.warning(f"[LEARNER] Critic value logging disabled for this episode: {exc}")
        return

    if value is not None:
        log_ctx["critic_values"].append(float(value))


def _save_transition_images(
    transition: dict,
    log_dir: str,
    step_idx: int,
    camera_names: list[str],
) -> None:
    observations = transition.get("state", {})
    for key, value in observations.items():
        if "image" not in key or not isinstance(value, torch.Tensor):
            continue
        camera_name = key.split(".")[-1]
        if camera_name not in camera_names:
            continue

        img = value.detach().cpu()
        while img.ndim > 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        elif img.ndim != 2:
            logging.debug(f"[LEARNER] Skipping image with unsupported shape {tuple(value.shape)}")
            continue

        img_np = img.numpy()
        if np.nanmax(img_np) <= 1.0:
            img_np = img_np * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        if img_np.ndim == 3 and img_np.shape[-1] == 1:
            img_np = img_np[..., 0]

        img_path = os.path.join(log_dir, f"step_{step_idx:06d}_{camera_name}.png")
        Image.fromarray(img_np).save(img_path)


def _finish_logging_episode(log_ctx: dict, episode: int) -> None:
    critic_values = log_ctx["critic_values"]
    log_dir = log_ctx["log_dir"]

    if critic_values:
        with open(os.path.join(log_dir, "critic_values.json"), "w") as f:
            json.dump(critic_values, f)

        plt.figure(figsize=(10, 5))
        plt.plot(critic_values)
        plt.title(f"Critic Values - Episode {episode}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "critic_plot.png"))
        plt.close()

        try:
            save_video_with_critic_overlay(
                log_dir,
                critic_values,
                camera_names=log_ctx["camera_names"],
            )
            logging.info(f"[LEARNER] Video generated for episode {episode}")
        except Exception as exc:
            logging.error(f"[LEARNER] Failed to generate video: {exc}")

    logging.info(f"[LEARNER] Finished logging episode {episode}")


def _init_online_buffer(
    cfg: TrainRLServerPipelineConfig,
    device,
    storage_device,
) -> ReplayBuffer:
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
            image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16"),
            image_storage_size=getattr(cfg.policy, "image_storage_size", None),
        )
    logging.info("Resuming: loading online buffer from disk")
    dataset = LeRobotDataset(
        repo_id=getattr(cfg.dataset, "repo_id", None),
        root=os.path.join(cfg.output_dir, "dataset"),
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
        image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16"),
        image_storage_size=getattr(cfg.policy, "image_storage_size", (224, 224)),
    )


def _init_offline_buffer(
    cfg: TrainRLServerPipelineConfig,
    device,
    storage_device,
) -> ReplayBuffer:
    if not cfg.resume:
        offline_dataset = make_dataset(cfg)
    else:
        episodes = cfg.dataset.episodes
        if episodes is None and cfg.dataset.max_episodes is not None:
            episodes = list(range(cfg.dataset.max_episodes))
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=os.path.join(cfg.output_dir, "dataset_offline"),
            episodes=episodes,
        )

    offline_dataset.delta_timestamps = None
    offline_dataset.delta_indices = None

    buf = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        reward_normalization_constant=float(getattr(cfg.policy, "reward_normalization_constant", 1.0)),
        terminal_failure_reward=float(getattr(cfg.policy, "terminal_failure_reward", -10.0)),
        inject_complementary_info={"is_golden": cfg.treat_main_dataset_as_golden},
        cache_dir=cfg.buffer_cache_dir,
        image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16"),
        image_storage_size=getattr(cfg.policy, "image_storage_size", (224, 224)),
    )
    buf.dataset = offline_dataset
    return buf


def _save_online_buffer(
    cfg: TrainRLServerPipelineConfig,
    replay_buffer: ReplayBuffer,
    fps: int,
    episode: int,
    step: int,
) -> None:
    online_buffer_dir = os.path.join(cfg.output_dir, "online_buffer")
    logging.info(f"[LEARNER] Saving online buffer at episode {episode}, step {step}, "
                 f"buffer size {len(replay_buffer)}")
    if os.path.exists(online_buffer_dir) and os.path.isdir(online_buffer_dir):
        shutil.rmtree(online_buffer_dir)
    replay_buffer.to_lerobot_dataset(
        repo_id="online_buffer",
        fps=fps,
        root=online_buffer_dir,
        task_name=cfg.policy.task,
    )
    logging.info(f"[LEARNER] Online buffer saved to {online_buffer_dir}")


if __name__ == "__main__":
    train_cli()
