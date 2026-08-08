#!/usr/bin/env python
r"""Flow + FAST cross-entropy on held-out frames, cheap enough to log every step.

Replaces the in-training modality ablation deltas (removed 2026-08-08). Those
estimated $I(a; x_m \mid x_{\setminus m})$ on the *training* micro-batch, which is
zero by construction once the model has memorised its episodes: the answer is
recoverable from proprio and the task string alone, so no observation is
necessary and every modality reads ~0. Measured over v5 and v6, all three
deltas started positive and decayed monotonically toward zero as memorisation
completed (v6 top: +0.00081 → −0.00109, slope $p=0.015$) — the metric was
tracking memorisation and then bottoming out, at 0.12% of the CE it sat on.

This measures the one quantity that still has range on held-out data: the
training objective itself. Val flow and val CE against the train curves is the
generalisation gap, which is what the memorisation finding says to watch.

Cost is kept to a few forwards per call by doing all the expensive work once:

* **Frames are sampled and packed at construction.** Tokenisation, the image
  processor's patchification, depth back-projection inputs and the history
  window are paid once at startup, not on every call. Each call is then pure
  GPU forward on tensors that are already in model layout. The sample lives in
  CPU RAM and costs ~2.4 GB at 128 frames — measured, and after dropping the raw
  camera tensors the pack step has already consumed (see ``_CONSUMED_BY_PACK``;
  they were 69% of it).
* **Timesteps sit on the stratified quantile grid** of the training Beta, and
  the noise is drawn once from a seeded generator. The flow loss is therefore a
  deterministic function of the weights: a change between two steps is the model
  moving and nothing else. Without this the sampler's own draw dominates — it is
  what made the ablation deltas unreadable at single-step cadence.
* **Pack dropout is suppressed and the policy runs in eval.** The number is the
  deployment regime (every camera present, subtask and metadata clauses
  rendered), so it is *not* directly comparable to ``train/loss_flow``, which is
  measured with dropout armed and reads higher.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.probes.objective import flow_timestep_grid
from lerobot.probes.utils import (
    build_episode_index,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
    suppress_pack_dropout,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


def _CONSUMED_BY_PACK(key: str) -> bool:
    """Raw camera tensors the pack step has already turned into ``pixel_values`` /
    ``history_images``, and which the model never reads back (modeling_molmoact2.py
    mentions neither family; ``_MODEL_INPUT_KEYS`` is an allowlist besides).

    Dropping them is what makes the sample cacheable: they are 69% of a packed
    batch, and they would otherwise sit in RAM for the whole run. Depth is NOT in
    here — the point-map encoder reads ``observation.depth.*`` and
    ``history.depth.*.depth`` raw, on every forward."""
    return key.startswith("observation.images.") or key.startswith("history.observation.images.")


class ValLoss:
    """Fixed held-out sample, packed once, re-scored on demand.

    Args:
        val_dataset: the held-out dataset (``cfg.val_dataset_path``).
        preprocessor: the training preprocessor — the same instance the trainer
            packs with, so the val batch goes through the identical pack step.
        cfg: the pipeline config.
        device: where the forwards run.
        n_frames: total frames in the sample, spread evenly over val episodes.
        batch_size: frames per forward. Defaults to ``cfg.batch_size``, so peak
            activation memory matches a training micro-batch.
    """

    def __init__(self, val_dataset, preprocessor, cfg, device, n_frames: int = 128, batch_size: int | None = None):
        self._device = device
        self._batches: list[dict] = []
        self._counts: list[int] = []

        chunk_size = int(cfg.policy.chunk_size)
        action_dim = int(cfg.policy.output_features[ACTION].shape[0])
        batch_size = int(batch_size or cfg.batch_size)
        seed = int(getattr(cfg.probe_parameters, "random_seed", 0))
        stride = probe_image_stride(cfg)

        n_episodes = max(len(build_episode_index(val_dataset)), 1)
        samples = sample_episodes_evenly(
            val_dataset, -(-n_frames // n_episodes), None, seed, stride
        )
        # Even spread over whatever the stride grid actually yielded, rather than
        # truncating to the first n and biasing the sample toward early episodes.
        if len(samples) > n_frames:
            samples = [samples[i] for i in np.linspace(0, len(samples) - 1, n_frames, dtype=int)]

        n_timesteps = max(1, int(cfg.policy.num_flow_timesteps))
        grid = torch.from_numpy(flow_timestep_grid(cfg.policy, n_timesteps)).float()
        generator = torch.Generator().manual_seed(seed)

        # Decode one group at a time: raw frames carry full-resolution images and
        # their history windows, so holding all n before packing costs more than
        # the packed sample it produces.
        with suppress_pack_dropout(preprocessor):
            for start in range(0, len(samples), batch_size):
                group = [
                    probe_frame_inputs(val_dataset, cfg, global_idx, chunk_size)
                    for _, _, global_idx in samples[start : start + batch_size]
                ]
                batch = self._pack(group, preprocessor, chunk_size, action_dim)
                padded = batch[ACTION]
                batch["_flow_timesteps"] = grid.expand(len(group), n_timesteps).clone()
                batch["_flow_noise"] = torch.randn(
                    (len(group), n_timesteps, padded.shape[1], padded.shape[2]),
                    generator=generator,
                    dtype=torch.float32,
                )
                self._batches.append(batch)
                self._counts.append(len(group))

        logging.info(
            f"[VAL_LOSS] {sum(self._counts)} frames over {n_episodes} episodes, "
            f"{len(self._batches)} forwards of <={batch_size}"
        )

    @staticmethod
    def _pack(frames: list[dict], preprocessor, chunk_size: int, action_dim: int) -> dict:
        """One packed batch from N probe frames, on CPU.

        The pack step is already batched over subtask texts, metadata dicts and the
        history stack, so stacking the per-frame observation tensors on dim 0 and
        passing lists for the rest is all it takes.
        """
        obs = {
            key: torch.cat([f["obs"][key] for f in frames], dim=0)
            for key in frames[0]["obs"]
        }
        flat = {
            **obs,
            "task": [f["task"] for f in frames],
            ACTION: torch.stack(
                [f["gt_actions"][:chunk_size, :action_dim] for f in frames]
            ),
            TransitionKey.COMPLEMENTARY_DATA: {
                "subtask": [f["subtask"] for f in frames],
                "metadata": [f["metadata"] for f in frames],
            },
        }
        return {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in preprocessor(flat).items()
            if not _CONSUMED_BY_PACK(k)
        }

    @torch.no_grad()
    def __call__(self, policy) -> dict[str, float]:
        """Mean flow and FAST CE over the fixed sample, weighted by frames per forward."""
        was_training = policy.training
        policy.eval()
        totals = {"val_loss_flow": 0.0, "val_loss_discrete_ce": 0.0}
        try:
            for batch, count in zip(self._batches, self._counts):
                on_device = {
                    k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                    if not k.startswith("_flow_")
                }
                _, metrics = policy.forward(
                    on_device,
                    flow_timesteps=batch["_flow_timesteps"].to(self._device),
                    flow_noise=batch["_flow_noise"].to(self._device),
                )
                totals["val_loss_flow"] += float(metrics.get("action_flow_loss", 0.0)) * count
                totals["val_loss_discrete_ce"] += float(metrics.get("discrete_ce_loss", 0.0)) * count
        finally:
            policy.train(was_training)

        n = max(sum(self._counts), 1)
        return {k: v / n for k, v in totals.items()}
