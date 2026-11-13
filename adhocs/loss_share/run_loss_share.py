"""Loss share and gradient share per data group, same frames across the checkpoints of one run.

    uv run python lerobot/adhocs/loss_share/run_loss_share.py --config_path=config_rl.yaml \
        --run_dir=outputs/rebot-corr10-nodepth-b128-20261111 --steps=400,800,1200,1600,2000 \
        --out_dir=outputs/loss_share/corr10_2026-11-12

``n_frames`` chunk starts are drawn once from the training mixture (the config's group shares, uniform
over a group's eligible starts, diverse rows from the training sampler) and held fixed, with a fixed
flow-time grid and fixed noise per frame (the ValLoss recipe). For every checkpoint each frame gets its
flow loss (the training loss_flow term, normalized anchor-encoded actions) and the norm of that loss's
gradient; with knowledge insulation the flow gradient reaches the action expert only, so the gradient
is taken over the action expert's parameters. ``batch`` frames of one source go through together; the
gradient norm is then per batch of the batch-mean loss (batch=1: per frame).

Per group k with summed-loss gradient G_k and G = sum_k G_k (N * the gradient of the mean loss):
loss share = sum of losses in k / total, norm share = ||G_k|| / sum_j ||G_j||, projection share =
G_k.G / ||G||^2 (sums to 1, the group's part of the actual update direction).

Outputs (out_dir): frames.csv, step_<step>_samples.csv, step_<step>_groups.csv,
step_<step>_cosines.csv, run.json. ``plot_loss_share.py`` draws the figures.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.objective import flow_timestep_grid
from lerobot.probes.utils import register_config_choices, suppress_pack_dropout
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.offline_dataset_utils import (
    buffer_state_keys,
    get_offline_dataset_sources,
    get_offline_dataset_weights,
    load_additional_offline_buffers,
    load_metadata_rows,
    load_offline_dataset,
    materialize_dataset_labels,
    pool_lowdim_stats,
)
from lerobot.rl.rl_trainer import Trainer
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.transition import move_transition_to_device
from lerobot.utils.utils import init_logging

COARSE = {"external_rebot": "external", "corrections": "corrections", "diverse": "diverse"}  # the rest = internal
COARSE_ORDER = ["internal", "external", "corrections", "diverse"]


@dataclass
class LossShareConfig(TrainRLServerPipelineConfig):
    run_dir: str = "outputs/rebot-corr10-nodepth-b128-20261111"
    steps: str = "400,800,1200,1600,2000"
    out_dir: str = "outputs/loss_share/corr10_2026-11-12"
    n_frames: int = 1024
    batch: int = 1  # frames per forward/backward (same source); 1 = per-frame gradient norms
    frame_seed: int = 0


def checkpoint_path(cfg, step: int) -> str:
    return os.path.join(cfg.run_dir, "checkpoints", f"{step:06d}", "pretrained_model")


def build_groups(cfg, trainer, preprocessor, offline_dataset, device):
    """The training mixture groups, built as scripts/rl_offline.py builds them."""
    from lerobot.rl.data_sources.diverse_integration import (
        align_rebot_buffers,
        build_diverse_buffer,
        build_mixture_groups,
        sample_spec_from_config,
    )

    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    fps = cfg.env.fps
    memory_cfg = getattr(cfg.policy, "memory", None)
    history_offsets = memory_cfg.history_offsets(fps) if memory_cfg is not None else None
    main_source = get_offline_dataset_sources(cfg)[0]
    main_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=buffer_state_keys(cfg, offline_dataset),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        reward_normalization_constant=cfg.policy.reward_normalization_constant,
        terminal_failure_reward=cfg.policy.terminal_failure_reward,
        cache_dir=getattr(cfg, "buffer_cache_dir", None),
        image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16"),
        image_storage_size=getattr(cfg.policy, "image_storage_size", (224, 224)),
        image_stride=getattr(cfg.policy, "image_stride", 1),
        history_offsets=history_offsets,
    )
    main_buffer.dataset = offline_dataset
    main_buffer.offline_source = main_source
    materialize_dataset_labels(
        main_buffer, offline_dataset, offline_dataset, source_index=0, is_main_process=True,
        require_depth_gripper_event_labels=False, embodiment=main_source.embodiment,
    )
    if memory_cfg is not None and memory_cfg.metadata_enabled:
        main_buffer.materialize_metadata(*load_metadata_rows(offline_dataset.root))
    additional = load_additional_offline_buffers(
        cfg=cfg, main_dataset=offline_dataset, device=device, storage_device=storage_device,
        is_main_process=True, history_offsets=history_offsets, memory_cfg=memory_cfg,
    )
    trainer.sync_subtask_vocabulary(preprocessor, offline_dataset, is_main_process=True)
    buffers = [main_buffer, *additional]
    for buffer in buffers:
        buffer.configure_critic_rewards(
            str(getattr(cfg.policy, "critic_reward_mode", "episode")),
            float(getattr(cfg.policy, "critic_mistake_penalty", 0.0)),
            bootstrap_subtasks=bool(getattr(cfg.policy, "advantage_bootstrap_subtasks", False)),
        )
    spec = sample_spec_from_config(cfg)
    diverse_buffer = build_diverse_buffer(
        cfg, cfg.diverse, main_dataset=offline_dataset, device=device, seed=int(cfg.seed or 0),
        rank=0, world_size=1, is_main_process=True,
    )
    trainer.sync_subtask_vocabulary(preprocessor, offline_dataset, is_main_process=True)
    return build_mixture_groups(
        cfg.diverse,
        align_rebot_buffers(buffers, cfg.diverse, spec, is_main_process=True),
        diverse_buffer,
        rebot_weights=get_offline_dataset_weights(cfg),
        action_chunk_size=cfg.policy.n_action_steps,
    )


def draw_frames(groups, n_frames: int, seed: int) -> pd.DataFrame:
    """One row per frame: group, coarse group, view (index into the group's buffers), source, row."""
    weights = np.array([g.weight for g in groups], dtype=np.float64)
    exact = weights / weights.sum() * n_frames
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts))[: n_frames - counts.sum()]:
        counts[i] += 1
    rng = np.random.default_rng(seed)
    rows = []
    for group, count in zip(groups, counts, strict=True):
        coarse = COARSE.get(group.name, "internal")
        if group.name == "diverse":
            for row in group.buffers[0].draw_indices(int(count)):
                rows.append((group.name, coarse, 0, "diverse", int(row)))
            continue
        starts = torch.cat([view.sampling_indices.cpu() for view in group.buffers]).numpy()
        view_of = np.concatenate([np.full(len(view.sampling_indices), i) for i, view in enumerate(group.buffers)])
        for pick in np.sort(rng.choice(len(starts), size=int(count), replace=False)):
            view = int(view_of[pick])
            rows.append((group.name, coarse, view, group.buffers[view].offline_source.name, int(starts[pick])))
    frames = pd.DataFrame(rows, columns=["group", "coarse", "view", "source", "row"])
    frames.insert(0, "frame", np.arange(len(frames)))
    return frames


def flow_losses(policy, batch, timesteps, noise):
    vision = policy._backbone().vision_backbone
    vision._lerobot_future_features = None
    vision._lerobot_capture_future_features = False
    losses, _ = policy._compute_flow_matching_loss_joint_per_layer(
        batch=batch, model_inputs=policy._model_inputs(batch), timesteps=timesteps, noise=noise, reduction="none"
    )
    return losses


def measure_checkpoint(cfg, trainer, preprocessor, offline_dataset, groups, frames, step, device):
    cfg.policy.pretrained_path = checkpoint_path(cfg, step)
    policy = trainer.make_policy(cfg)
    trainer.freeze_model(policy, cfg)
    policy.eval()
    trainable = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    expert = [(n, p) for n, p in trainable if ".action_expert." in n]
    by_name = {g.name: g for g in groups}
    grid = torch.from_numpy(flow_timestep_grid(cfg.policy, int(cfg.policy.num_flow_timesteps))).float()

    def pack(chunk):
        group = by_name[chunk.group.iloc[0]]
        if chunk.group.iloc[0] == "diverse":
            raw = group.buffers[0].collate(chunk.row.to_numpy())
        else:
            view = group.buffers[int(chunk.view.iloc[0])]
            raw = view.sample(len(chunk), action_chunk_size=cfg.policy.n_action_steps, indices=chunk.row.to_numpy())
        raw = move_transition_to_device(raw, str(device))
        observations = trainer._inject_depth_observations(raw.get("state", {}), raw.get("complementary_info"), cfg)
        with suppress_pack_dropout(preprocessor):
            batch = trainer.build_training_batch(
                raw_batch=raw, observations=observations, actions=raw[ACTION],
                preprocessor=preprocessor, dataset=offline_dataset, cfg=cfg,
            )
        action = batch[ACTION]
        noise = torch.stack([
            torch.randn((len(grid), action.shape[1], action.shape[2]),
                        generator=torch.Generator().manual_seed(cfg.frame_seed * 1_000_003 + int(frame)))
            for frame in chunk.frame
        ]).to(device)
        return batch, grid.expand(len(chunk), len(grid)).clone().to(device), noise

    # Knowledge insulation check: the flow loss must leave every non-expert parameter without gradient.
    batch, timesteps, noise = pack(frames.iloc[:1])
    grads = torch.autograd.grad(flow_losses(policy, batch, timesteps, noise).sum(), [p for _, p in trainable], allow_unused=True)
    leaked = [n for (n, _), g in zip(trainable, grads, strict=True)
              if ".action_expert." not in n and g is not None and bool(g.float().abs().sum() > 0)]
    if leaked:
        raise RuntimeError(f"flow gradient reaches {len(leaked)} parameters outside the action expert, e.g. {leaked[:3]}")
    del grads
    for name, param in trainable:
        if ".action_expert." not in name:
            param.requires_grad_(False)
    params = [p for _, p in expert]
    logging.info(f"[loss_share] step {step}: {len(params)} action-expert tensors, {sum(p.numel() for p in params):,} params")

    group_grads = {c: [torch.zeros_like(p, dtype=torch.float32) for p in params] for c in COARSE_ORDER}
    records = []
    started = time.time()
    chunks = [part.iloc[i:i + cfg.batch]
              for _, part in frames.groupby(["group", "view"], sort=False)
              for i in range(0, len(part), cfg.batch)]
    for done, chunk in enumerate(chunks):
        batch, timesteps, noise = pack(chunk)
        losses = flow_losses(policy, batch, timesteps, noise)
        grads = torch.autograd.grad(losses.sum(), params)
        square = 0.0
        for total, grad in zip(group_grads[chunk.coarse.iloc[0]], grads, strict=True):
            grad = grad.float()
            total += grad
            square += float(grad.square().sum())
        action = batch[ACTION].float()
        target_sq = action.square().flatten(1).sum(1) / (action != 0).flatten(1).sum(1).clamp(min=1)
        for i, frame in enumerate(chunk.frame):
            records.append({
                "step": step, "frame": int(frame), "loss": float(losses[i].detach()), "target_sq": float(target_sq[i]),
                "grad_norm": square ** 0.5 / len(chunk), "batch": len(chunk),
            })
        if done % 50 == 0:
            logging.info(f"[loss_share] step {step}: {done}/{len(chunks)} batches, {time.time() - started:.0f} s")

    samples = frames.merge(pd.DataFrame(records), on="frame")
    dots = np.array([[sum(float((x * y).sum()) for x, y in zip(group_grads[a], group_grads[b], strict=True))
                      for b in COARSE_ORDER] for a in COARSE_ORDER])
    norms = np.sqrt(np.diag(dots))
    total_sq = dots.sum()
    rows = []
    for i, coarse in enumerate(COARSE_ORDER):
        part = samples[samples.coarse == coarse]
        rows.append({
            "step": step, "group": coarse, "frames": len(part), "frame_share": len(part) / len(samples),
            "loss_mean": part.loss.mean(), "loss_share": part.loss.sum() / samples.loss.sum(),
            "target_sq_mean": part.target_sq.mean(),
            "grad_norm_mean": part.grad_norm.mean(),
            "grad_norm_sum_share": part.grad_norm.sum() / samples.grad_norm.sum(),
            "group_grad_norm": norms[i], "group_grad_norm_share": norms[i] / norms.sum(),
            "projection_share": dots[i].sum() / total_sq,
            "cos_to_total": dots[i].sum() / (norms[i] * total_sq ** 0.5),
        })
    samples.to_csv(os.path.join(cfg.out_dir, f"step_{step:06d}_samples.csv"), index=False)
    pd.DataFrame(dots / np.outer(norms, norms), index=COARSE_ORDER, columns=COARSE_ORDER).to_csv(
        os.path.join(cfg.out_dir, f"step_{step:06d}_cosines.csv"))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(cfg.out_dir, f"step_{step:06d}_groups.csv"), index=False)
    logging.info(f"[loss_share] step {step} ({time.time() - started:.0f} s)\n{table.round(4).to_string(index=False)}")
    del policy, group_grads, params, expert, trainable
    torch.cuda.empty_cache()


def run(cfg: LossShareConfig):
    init_logging()
    os.makedirs(cfg.out_dir, exist_ok=True)
    steps = [int(s) for s in cfg.steps.split(",")]
    cfg.policy.pretrained_path = checkpoint_path(cfg, steps[0])
    cfg.validate()
    cfg.env.task = cfg.policy.task
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = get_safe_torch_device(try_device=cfg.policy.device)
    cfg.policy.device = str(device)
    trainer = Trainer.for_config(cfg)
    offline_dataset = load_offline_dataset(cfg, get_offline_dataset_sources(cfg)[0])
    pool_lowdim_stats(cfg, offline_dataset, is_main_process=True)
    preprocessor, _ = trainer.make_processors(cfg, dataset=offline_dataset, is_main_process=True)
    groups = build_groups(cfg, trainer, preprocessor, offline_dataset, device)
    frames = draw_frames(groups, cfg.n_frames, cfg.frame_seed)
    frames.to_csv(os.path.join(cfg.out_dir, "frames.csv"), index=False)
    total = sum(g.weight for g in groups)
    with open(os.path.join(cfg.out_dir, "run.json"), "w") as f:
        json.dump({
            "run_dir": cfg.run_dir, "steps": steps, "n_frames": len(frames), "batch": cfg.batch,
            "frame_seed": cfg.frame_seed, "group_shares": {g.name: g.weight / total for g in groups},
            "frames_per_group": frames.group.value_counts().to_dict(),
        }, f, indent=2)
    logging.info(f"[loss_share] frames per group: {frames.coarse.value_counts().to_dict()}")
    for step in steps:
        if os.path.exists(os.path.join(cfg.out_dir, f"step_{step:06d}_groups.csv")):
            logging.info(f"[loss_share] step {step}: done, skipped")
            continue
        measure_checkpoint(cfg, trainer, preprocessor, offline_dataset, groups, frames, step, device)


@parser.wrap()
def cli(cfg: LossShareConfig):
    run(cfg)


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()
