"""Drift of the gradient on fixed probe frames while the model trains.

    uv run python lerobot/adhocs/grad_drift/run_grad_drift.py --config_path=config_rl.yaml \
        --out_dir=outputs/grad_drift/base_2026-11-12 --measure_at=0,1,2,4,8,16,32,64,128,256

``n_probe`` chunk starts per data group (internal / external / corrections / diverse) are drawn once from
the training mixture and packed once, with a fixed flow-time grid and fixed noise per frame (the ValLoss
recipe, eval mode, pack dropout off). The probe loss is the training loss (flow + discrete CE) and its
gradient is taken over every trainable parameter. Between measurements the model trains with the run's
own step (``trainer.update_actor``: config batch, accumulation, clip, AdamW) and the run's learning-rate
schedule from its step 0 (warmup included). The start weights are ``policy.pretrained_path`` (null = base
MolmoAct2).

A gradient is not kept. Per parameter block (``block_of``) it is reduced to a count sketch of
``sketch_dim`` numbers: every element has a fixed random bucket and sign, the sketch is linear, and the dot
product of two sketches estimates the dot product of the two vectors (cosine error about
1 / sqrt(sketch_dim) per block, less over all blocks). The displacement theta_t - theta_0 goes through the
same sketch, so g . (theta_t - theta_0) can be estimated too. Squared norms are saved exact.

Outputs (out_dir): frames.csv, run.json, step_<step>.npz (grad [group, block, dim], grad_sq
[group, block], disp [block, dim], disp_sq [block]), floor.npz (step 0 with other flow noise),
losses.csv (per measurement and frame), train.csv (per training step). ``plot_grad_drift.py`` draws the
figures.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loss_share"))

from run_loss_share import COARSE, COARSE_ORDER, build_groups, draw_frames  # noqa: E402

from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainRLServerPipelineConfig  # noqa: E402
from lerobot.probes.objective import flow_timestep_grid  # noqa: E402
from lerobot.probes.utils import register_config_choices, suppress_pack_dropout  # noqa: E402
from lerobot.rl.buffer import concatenate_variable_dim_batch_transitions  # noqa: E402
from lerobot.rl.data_sources.diverse_mixture import make_hierarchical_offline_iterator  # noqa: E402
from lerobot.rl.molmoact2.rl_molmoact2_trainer import _weight_block_of  # noqa: E402
from lerobot.rl.offline_dataset_utils import (  # noqa: E402
    get_offline_dataset_sources,
    load_offline_dataset,
    pool_lowdim_stats,
)
from lerobot.rl.rl_trainer import Trainer  # noqa: E402
from lerobot.rl.utils import build_named_adamw_optimizers, build_named_schedulers, step_named_schedulers  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from lerobot.utils.transition import move_transition_to_device  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402


@dataclass
class GradDriftConfig(TrainRLServerPipelineConfig):
    out_dir: str = "outputs/grad_drift/base_2026-11-12"
    n_probe: int = 32  # probe frames per data group, one forward/backward per group
    measure_at: str = "0,1,2,4,8,16,32,64,128,256"  # training steps taken before each measurement
    sketch_dim: int = 4096
    frame_seed: int = 0


def block_of(name: str) -> str:
    """Weight-audit block plus the first layer index in the name: vlm.07, vit.12, action_expert.03, vlm.rest."""
    layer = re.search(r"\.(\d+)\.", name)
    return f"{_weight_block_of(name)}.{int(layer.group(1)):02d}" if layer else f"{_weight_block_of(name)}.rest"


class Sketch:
    """Count sketch per parameter block; tensors come in the order of ``names``."""

    def __init__(self, names: list[str], dim: int, seed: int):
        self.blocks = sorted({block_of(name) for name in names})
        self.block_index = [self.blocks.index(block_of(name)) for name in names]
        self.dim = dim
        self.seed = seed

    def __call__(self, tensors, device):
        sketch = torch.zeros(len(self.blocks), self.dim, dtype=torch.float64, device=device)
        square = torch.zeros(len(self.blocks), dtype=torch.float64, device=device)
        for i, tensor in enumerate(tensors):
            if tensor is None:
                continue
            flat = tensor.detach().reshape(-1).double()
            # One draw per element: bucket = draw // 2, sign from the low bit. Same seed, same hash every call.
            draw = torch.randint(2 * self.dim, (flat.numel(),), device=device,
                                 generator=torch.Generator(device).manual_seed(self.seed * 1_000_003 + i))
            sketch[self.block_index[i]].index_add_(0, draw // 2, flat * (1 - 2 * (draw % 2)))
            square[self.block_index[i]] += flat.square().sum()
        return sketch.float().cpu().numpy(), square.cpu().numpy()


def draw_probe_frames(groups, n_probe: int, seed: int) -> pd.DataFrame:
    """``n_probe`` frames per coarse group, the member groups by their training weights."""
    parts = []
    for coarse in COARSE_ORDER:
        members = [g for g in groups if COARSE.get(g.name, "internal") == coarse]
        if members:
            parts.append(draw_frames(members, n_probe, seed))
    frames = pd.concat(parts, ignore_index=True)
    frames["frame"] = np.arange(len(frames))
    return frames


def pack_probes(cfg, trainer, preprocessor, offline_dataset, groups, frames, device) -> dict:
    """coarse -> (packed batch on the device, flow times, frame ids in batch order)."""
    by_name = {g.name: g for g in groups}
    grid = torch.from_numpy(flow_timestep_grid(cfg.policy, int(cfg.policy.num_flow_timesteps))).float()
    probes = {}
    for coarse, part in frames.groupby("coarse", sort=False):
        raw = None
        ids = []
        for (name, view), chunk in part.groupby(["group", "view"], sort=False):
            if name == "diverse":
                piece = by_name[name].buffers[0].collate(chunk.row.to_numpy())
            else:
                piece = by_name[name].buffers[int(view)].sample(
                    len(chunk), action_chunk_size=cfg.policy.n_action_steps, indices=chunk.row.to_numpy())
            raw = piece if raw is None else concatenate_variable_dim_batch_transitions(raw, piece)
            ids.extend(int(f) for f in chunk.frame)
        raw = move_transition_to_device(raw, str(device))
        observations = trainer._inject_depth_observations(raw.get("state", {}), raw.get("complementary_info"), cfg)
        with suppress_pack_dropout(preprocessor):
            batch = trainer.build_training_batch(
                raw_batch=raw, observations=observations, actions=raw[ACTION],
                preprocessor=preprocessor, dataset=offline_dataset, cfg=cfg,
            )
        probes[coarse] = (batch, grid.expand(len(ids), len(grid)).clone().to(device), ids)
    return probes


def measure(policy, params, probes, sketch, noise_seed: int, device):
    """Per group: the probe loss per frame, and the sketch + exact squared norm of the mean-loss gradient."""
    policy.eval()
    # Gradient checkpointing is gated on these two training flags, and without it the backward through the
    # backbone does not fit in memory. Every dropout reads the flag of its own submodule, which stays eval.
    policy.training = True
    policy._backbone().vision_backbone.training = True
    sketches, squares, rows = [], [], []
    for coarse, (batch, timesteps, ids) in probes.items():
        action = batch[ACTION]
        noise = torch.stack([
            torch.randn((timesteps.shape[1], action.shape[1], action.shape[2]),
                        generator=torch.Generator().manual_seed(noise_seed * 1_000_003 + frame))
            for frame in ids
        ]).to(device)
        policy.zero_grad(set_to_none=True)
        losses, metrics = policy.forward(batch, reduction="none", flow_timesteps=timesteps, flow_noise=noise)
        losses.mean().backward()
        one_sketch, one_square = sketch((p.grad for p in params), device)
        sketches.append(one_sketch)
        squares.append(one_square)
        for frame, loss in zip(ids, losses.detach().float().tolist(), strict=True):
            rows.append({"coarse": coarse, "frame": frame, "loss": loss, "group_flow": metrics["action_flow_loss"],
                         "group_ce": metrics["discrete_ce_loss"]})
    policy.zero_grad(set_to_none=True)
    return np.stack(sketches), np.stack(squares), rows


def run(cfg: GradDriftConfig):
    init_logging()
    os.makedirs(cfg.out_dir, exist_ok=True)
    measure_at = sorted({int(s) for s in cfg.measure_at.split(",")})
    cfg.validate()
    cfg.env.task = cfg.policy.task
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = get_safe_torch_device(try_device=cfg.policy.device)
    cfg.policy.device = str(device)
    trainer = Trainer.for_config(cfg)
    policy = trainer.make_policy(cfg)
    trainer.freeze_model(policy, cfg)
    offline_dataset = load_offline_dataset(cfg, get_offline_dataset_sources(cfg)[0])
    pool_lowdim_stats(cfg, offline_dataset, is_main_process=True)
    preprocessor, postprocessor = trainer.make_processors(cfg, dataset=offline_dataset, is_main_process=True)
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor
    optimizers = build_named_adamw_optimizers(trainer.get_optimizer_groups(policy, cfg), cfg.policy, seed=int(cfg.seed or 0))
    schedulers = build_named_schedulers(
        optimizers, cfg.policy, {"policy": cfg.policy.offline_steps, "depth": cfg.policy.offline_steps})
    groups = build_groups(cfg, trainer, preprocessor, offline_dataset, device)

    frames = draw_probe_frames(groups, cfg.n_probe, cfg.frame_seed)
    frames.to_csv(os.path.join(cfg.out_dir, "frames.csv"), index=False)
    probes = pack_probes(cfg, trainer, preprocessor, offline_dataset, groups, frames, device)
    train_iter = make_hierarchical_offline_iterator(
        groups, batch_size=cfg.batch_size, action_chunk_size=cfg.policy.n_action_steps,
        proportional=bool(cfg.diverse.rebot_group_weights), seed=int(cfg.seed or 0),
    )

    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    start = [p.detach().clone() for p in params]
    sketch = Sketch([n for n, _ in named], cfg.sketch_dim, cfg.frame_seed)
    block_params = [0] * len(sketch.blocks)
    for index, param in zip(sketch.block_index, params, strict=True):
        block_params[index] += param.numel()
    with open(os.path.join(cfg.out_dir, "run.json"), "w") as f:
        json.dump({
            "pretrained_path": cfg.policy.pretrained_path, "measure_at": measure_at, "groups": list(probes),
            "n_probe": cfg.n_probe, "sketch_dim": cfg.sketch_dim, "frame_seed": cfg.frame_seed,
            "batch_size": cfg.batch_size, "gradient_accumulation_steps": cfg.policy.gradient_accumulation_steps,
            "optimizer_lr": cfg.policy.optimizer_lr, "blocks": sketch.blocks, "block_params": block_params,
        }, f, indent=2)
    logging.info(f"[grad_drift] {sum(block_params):,} trainable params in {len(sketch.blocks)} blocks, "
                 f"probe frames per group: {frames.coarse.value_counts().to_dict()}")

    floor, floor_sq, _ = measure(policy, params, probes, sketch, cfg.frame_seed + 1, device)
    np.savez(os.path.join(cfg.out_dir, "floor.npz"), grad=floor, grad_sq=floor_sq)

    loss_rows, train_rows = [], []
    step = 0
    started = time.time()
    while True:
        if step in measure_at:
            grad, grad_sq, rows = measure(policy, params, probes, sketch, cfg.frame_seed, device)
            disp, disp_sq = sketch((p.detach().float() - s.float() for p, s in zip(params, start, strict=True)), device)
            np.savez(os.path.join(cfg.out_dir, f"step_{step:06d}.npz"), grad=grad, grad_sq=grad_sq, disp=disp, disp_sq=disp_sq)
            loss_rows.extend({"step": step, **row} for row in rows)
            pd.DataFrame(loss_rows).to_csv(os.path.join(cfg.out_dir, "losses.csv"), index=False)
            pd.DataFrame(train_rows).to_csv(os.path.join(cfg.out_dir, "train.csv"), index=False)
            logging.info(f"[grad_drift] step {step}: grad norm {dict(zip(probes, np.sqrt(grad_sq.sum(1)).round(4).tolist(), strict=True))}, "
                         f"displacement {disp_sq.sum() ** 0.5:.4f}, {time.time() - started:.0f} s")
        if step == measure_at[-1]:
            break
        policy.train()
        infos = trainer.update_actor(
            policy=policy, optimizers=optimizers, online_iter=train_iter, offline_iter=None,
            preprocessor=preprocessor, dataset=offline_dataset, device=device, cfg=cfg, optimization_step=step,
        )
        step_named_schedulers(schedulers, "policy", "depth")
        step += 1
        train_rows.append({"step": step, "lr": optimizers["policy"].param_groups[0]["lr"], **{k: infos[k] for k in ("loss_actor", "loss_flow", "loss_discrete_ce", "actor_grad_norm")}})


@parser.wrap()
def cli(cfg: GradDriftConfig):
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
