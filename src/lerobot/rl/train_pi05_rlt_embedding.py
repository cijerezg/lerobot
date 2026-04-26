#!/usr/bin/env python

import logging
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import draccus
import torch
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.processor.core import TransitionKey
from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade
from lerobot.rl.rlt_pi05 import PI05RLTConfig, PI05RLTPolicy
from lerobot.utils.constants import ACTION, OBS_PREFIX
from lerobot.utils.utils import init_logging


@dataclass
class TrainPI05RLTEmbeddingConfig:
    policy_path: str = "outputs/pi05_subtasks_good_dataset_4/checkpoints/000800/pretrained_model"
    dataset_repo_id: str = ""
    dataset_root: str | None = None
    output_dir: str = "outputs/pi05_rlt_embedding"
    task: str = ""
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 4
    steps: int = 10_000
    lr: float = 1e-4
    log_freq: int = 50
    save_freq: int = 1_000
    max_episodes: int | None = None
    rlt_token_dim: int = 2048
    rlt_token_max_seq_len: int = 1024
    rlt_token_encoder_layers: int = 2
    rlt_token_decoder_layers: int = 2
    rlt_token_num_heads: int = 8


def _build_policy(cfg: TrainPI05RLTEmbeddingConfig) -> PI05RLTPolicy:
    base_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    rlt_cfg = PI05RLTConfig.from_base_config(
        base_cfg,
        device=cfg.device,
        rlt_enabled=False,
        rlt_token_dim=cfg.rlt_token_dim,
        rlt_token_max_seq_len=cfg.rlt_token_max_seq_len,
        rlt_token_encoder_layers=cfg.rlt_token_encoder_layers,
        rlt_token_decoder_layers=cfg.rlt_token_decoder_layers,
        rlt_token_num_heads=cfg.rlt_token_num_heads,
        subtask_generation_enabled=False,
        pi05_checkpoint=cfg.policy_path,
        rtc_config=None,
    )
    policy = PI05RLTPolicy.from_pretrained(cfg.policy_path, config=rlt_cfg, strict=False)
    policy.to(cfg.device)
    policy.train()
    return policy


def _build_dataset(cfg: TrainPI05RLTEmbeddingConfig, policy: PI05RLTPolicy):
    if not cfg.dataset_repo_id:
        raise ValueError("--dataset_repo_id is required")
    dataset_cfg = DatasetConfig(
        repo_id=cfg.dataset_repo_id,
        root=cfg.dataset_root,
        max_episodes=cfg.max_episodes,
    )
    train_cfg = SimpleNamespace(
        dataset=dataset_cfg,
        policy=policy.config,
        num_workers=cfg.num_workers,
        tolerance_s=1e-4,
    )
    return make_dataset(train_cfg)


def _make_embedding_processor_policy_config(policy: PI05RLTPolicy) -> PI05RLTConfig:
    processor_policy_cfg = copy.deepcopy(policy.config)
    # The RL-token reconstruction loss only uses VLA prefix embeddings (images,
    # state-as-text, and task text). It does not consume action tokens, so avoid
    # the anchor/delta-specific action stats requirement in the shared RL utility.
    processor_policy_cfg.action_encoding = "absolute"
    processor_policy_cfg.action_encoding_stats_path = None
    return processor_policy_cfg


def _prepare_batch(raw_batch: dict, cfg: TrainPI05RLTEmbeddingConfig, device: str) -> dict:
    actions = raw_batch[ACTION].to(device)
    batch_size = actions.shape[0]
    tasks = raw_batch.get("task")
    if tasks is None:
        tasks = [cfg.task] * batch_size
    elif isinstance(tasks, str):
        tasks = [tasks] * batch_size

    batch_for_proc = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_batch.items()
        if key.startswith(OBS_PREFIX)
    }
    batch_for_proc[ACTION] = actions
    batch_for_proc[TransitionKey.COMPLEMENTARY_DATA] = {
        "task": list(tasks),
        "subtask": [""] * batch_size,
    }
    return batch_for_proc


def _save_checkpoint(policy: PI05RLTPolicy, output_dir: Path, step: int, cfg: TrainPI05RLTEmbeddingConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"rlt_embedding_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "config": draccus.encode(cfg),
            "rlt_embedding": policy.rlt_embedding.state_dict(),
        },
        ckpt_path,
    )
    latest_path = output_dir / "rlt_embedding_latest.pt"
    torch.save(
        {
            "step": step,
            "config": draccus.encode(cfg),
            "rlt_embedding": policy.rlt_embedding.state_dict(),
        },
        latest_path,
    )
    logging.info("Saved RLT embedding checkpoint to %s", ckpt_path)


def _repeat_dataloader(dataloader: DataLoader):
    while True:
        yield from dataloader


@draccus.wrap()
def train_pi05_rlt_embedding(cfg: TrainPI05RLTEmbeddingConfig) -> None:
    init_logging()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = _build_policy(cfg)
    dataset = _build_dataset(cfg, policy)
    processor_cfg = SimpleNamespace(policy=_make_embedding_processor_policy_config(policy))
    preprocessor, _ = make_pi05_full_processors_with_upgrade(
        processor_cfg,
        dataset=dataset,
        is_main_process=True,
    )

    optimizer = torch.optim.AdamW(policy.rlt_embedding.parameters(), lr=cfg.lr)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    data_iter = _repeat_dataloader(dataloader)

    policy.rlt_embedding.train()
    policy.model.eval()

    for step in range(1, cfg.steps + 1):
        raw_batch = next(data_iter)
        batch_for_proc = _prepare_batch(raw_batch, cfg, cfg.device)
        with torch.no_grad():
            batch = preprocessor(batch_for_proc)

        output = policy(batch, model="rlt_embedding")
        loss = output["loss_rlt_embedding"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.log_freq == 0:
            logging.info("step=%d loss_rlt_embedding=%.6f", step, float(loss.detach().cpu()))

        if step % cfg.save_freq == 0 or step == cfg.steps:
            _save_checkpoint(policy, output_dir, step, cfg)


if __name__ == "__main__":
    train_pi05_rlt_embedding()
