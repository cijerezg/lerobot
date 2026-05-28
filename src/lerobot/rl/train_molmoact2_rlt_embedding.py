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

"""Train the RL-token autoencoder on top of a frozen MolmoAct2 checkpoint."""

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import draccus
import torch
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.core import TransitionKey
from lerobot.rl.rlt_molmoact2 import MolmoAct2RLTConfig, MolmoAct2RLTPolicy
from lerobot.utils.constants import OBS_PREFIX
from lerobot.utils.utils import init_logging


@dataclass
class TrainMolmoAct2RLTEmbeddingConfig:
    policy_path: str = (
        "outputs/molmoact2_so101_placemotor_e51_v6/checkpoints/030000/030000/pretrained_model"
    )
    dataset_repo_id: str = ""
    dataset_root: str | None = None
    output_dir: str = "outputs/molmoact2_rlt_embedding"
    task: str = ""
    device: str = "cuda"
    batch_size: int = 4
    num_workers: int = 4
    steps: int = 10_000
    lr: float = 1e-4
    warmup_steps: int = 500
    log_freq: int = 50
    save_freq: int = 1_000
    max_episodes: int | None = None

    # Leave None to default to MolmoAct2's text hidden width.
    rlt_token_dim: int | None = None
    rlt_token_max_seq_len: int = 1024
    rlt_token_encoder_layers: int = 2
    rlt_token_decoder_layers: int = 2
    rlt_token_num_heads: int = 8


def _build_policy(cfg: TrainMolmoAct2RLTEmbeddingConfig) -> MolmoAct2RLTPolicy:
    base_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    rlt_cfg = MolmoAct2RLTConfig.from_base_config(
        base_cfg,
        device=cfg.device,
        rlt_enabled=False,
        rlt_token_dim=cfg.rlt_token_dim,
        rlt_token_max_seq_len=cfg.rlt_token_max_seq_len,
        rlt_token_encoder_layers=cfg.rlt_token_encoder_layers,
        rlt_token_decoder_layers=cfg.rlt_token_decoder_layers,
        rlt_token_num_heads=cfg.rlt_token_num_heads,
        molmoact2_checkpoint=cfg.policy_path,
        pretrained_path=Path(cfg.policy_path),
        inference_action_mode="continuous",
        enable_inference_cuda_graph=False,
        rtc_config=None,
    )
    policy = MolmoAct2RLTPolicy.from_pretrained(cfg.policy_path, config=rlt_cfg, strict=False)
    policy.to(cfg.device)
    policy.train()
    return policy


def _build_dataset(cfg: TrainMolmoAct2RLTEmbeddingConfig, policy: MolmoAct2RLTPolicy):
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


def _prepare_batch(raw_batch: dict, cfg: TrainMolmoAct2RLTEmbeddingConfig, device: str) -> dict:
    state = raw_batch.get("observation.state")
    if not torch.is_tensor(state):
        raise ValueError("MolmoAct2 RLT embedding training requires observation.state in the dataset.")
    batch_size = int(state.shape[0]) if state.ndim > 1 else 1
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
    batch_for_proc[TransitionKey.COMPLEMENTARY_DATA] = {"task": list(tasks)}
    return batch_for_proc


def _save_checkpoint(
    policy: MolmoAct2RLTPolicy,
    output_dir: Path,
    step: int,
    cfg: TrainMolmoAct2RLTEmbeddingConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": draccus.encode(cfg),
        "rlt_embedding": policy.rlt_embedding.state_dict(),
    }
    ckpt_path = output_dir / f"rlt_embedding_step_{step:06d}.pt"
    torch.save(payload, ckpt_path)
    torch.save(payload, output_dir / "rlt_embedding_latest.pt")
    logging.info("Saved MolmoAct2 RLT embedding checkpoint to %s", ckpt_path)


def _repeat_dataloader(dataloader: DataLoader):
    while True:
        yield from dataloader


@draccus.wrap()
def train_molmoact2_rlt_embedding(cfg: TrainMolmoAct2RLTEmbeddingConfig) -> None:
    init_logging()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = _build_policy(cfg)
    dataset = _build_dataset(cfg, policy)

    # Use the saved MolmoAct2 processor pipeline from the checkpoint so the
    # tokenizer, image packing, state normalizer, and camera key order match the
    # frozen VLA. Do not pass actions here: the RLT token is an inference-time
    # prompt/image/state embedding and should not reconstruct discrete action
    # label tokens.
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.policy_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    optimizer = torch.optim.AdamW(policy.rlt_embedding.parameters(), lr=cfg.lr)

    def _lr_lambda(step: int) -> float:
        if cfg.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / float(cfg.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
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
        scheduler.step()

        if step % cfg.log_freq == 0:
            logging.info(
                "step=%d loss_rlt_embedding=%.6f lr=%.2e",
                step,
                float(loss.detach().cpu()),
                scheduler.get_last_lr()[0],
            )

        if step % cfg.save_freq == 0 or step == cfg.steps:
            _save_checkpoint(policy, output_dir, step, cfg)


if __name__ == "__main__":
    train_molmoact2_rlt_embedding()
