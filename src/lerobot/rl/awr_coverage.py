"""Conditional training draws for AWR groups missed by mixture calibration."""

from collections import defaultdict

import numpy as np
import torch

from lerobot.rl.awr import group_key
from lerobot.rl.buffer import concatenate_variable_dim_batch_transitions


COVERAGE_SAMPLES_PER_GROUP = 8


def rebot_group_rows(buffer, names, action_chunk_size, *, include_skipped=False):
    """Eligible starts grouped by the same identities and boundaries as training."""
    high = max(0, buffer.size - 1) if buffer.optimize_memory and buffer.size < buffer.capacity else buffer.size
    indices = torch.arange(0, high, buffer.image_stride, device=buffer.storage_device)
    valid = torch.ones_like(indices, dtype=torch.bool)
    if buffer.actions.ndim == 2:
        for offset in range(action_chunk_size - 1):
            valid &= ~buffer.dones[(indices + offset) % buffer.capacity].bool()
    comp = buffer.complementary_info
    if not include_skipped and "critic_skip" in comp:
        valid &= ~comp["critic_skip"][indices].reshape(-1).bool()
    indices = indices[valid]
    embodiment, subtask = comp.get("embodiment_index"), comp.get("subtask_index")
    embodiment = torch.full_like(indices, -1) if embodiment is None else embodiment[indices].reshape(-1).long()
    subtask = torch.full_like(indices, -1) if subtask is None else subtask[indices].reshape(-1).long()
    pairs = torch.stack((embodiment, subtask), dim=1).cpu().numpy()
    rows = defaultdict(list)
    keys = {}
    for index, pair in zip(indices.cpu().tolist(), pairs, strict=True):
        robot, label = map(int, pair)
        if label >= len(names):
            raise ValueError(f"Unknown subtask index {label} in AWR training buffer.")
        identity = (robot, label)
        if identity not in keys:
            keys[identity] = group_key(robot, names[label] if label >= 0 else "")
        rows[keys[identity]].append(index)
    return {key: np.asarray(values, dtype=np.int64) for key, values in rows.items()}


class TrainingCoverageSampler:
    """Draw from the actual training distribution conditional on a requested group.

    ReBot sources are uniform over eligible starts. Diverse sources keep their
    source -> episode -> anchor probabilities. No held-out examples are consulted.
    These draws estimate missing group means/normalizers, never the mixture scale.
    """

    def __init__(self, groups, trainer, preprocessor, cfg):
        self.groups = groups
        self.names = trainer.subtask_vocabulary(preprocessor)
        self.batch_size = cfg.batch_size
        self.action_chunk_size = cfg.policy.n_action_steps

    def _candidates(self, requested):
        candidates = defaultdict(list)
        total_weight = sum(group.weight for group in self.groups)
        for group in self.groups:
            inner_total = sum(group.inner_weights)
            for buffer, weight in zip(group.buffers, group.inner_weights, strict=True):
                mass = group.weight / total_weight * weight / inner_total
                if hasattr(buffer, "_identity") and hasattr(buffer, "collate"):
                    probabilities = np.full(buffer.size, 1.0 / buffer.size)
                    sampler = buffer._sampler
                    if sampler is not None:
                        probabilities.fill(0)
                        for source, probability in zip(sampler.groups, sampler.probabilities, strict=True):
                            episodes = sampler.episode_rows[source]
                            for rows in episodes:
                                probabilities[rows] = probability / len(episodes) / len(rows)
                    rows_by_group = defaultdict(list)
                    for row, (identity, skip) in enumerate(zip(buffer._identity, buffer._critic.skip, strict=True)):
                        if skip:
                            continue
                        label = identity.subtask_index
                        key = group_key(identity.embodiment_index, self.names[label] if label >= 0 else "")
                        if key in requested:
                            rows_by_group[key].append(row)
                    for key, rows in rows_by_group.items():
                        for row in rows:
                            candidates[key].append((buffer, row, mass * probabilities[row]))
                else:
                    rows_by_group = rebot_group_rows(buffer, self.names, self.action_chunk_size, include_skipped=True)
                    count = sum(map(len, rows_by_group.values()))
                    skip = buffer.complementary_info.get("critic_skip")
                    skip = None if skip is None else skip.cpu().numpy().reshape(-1)
                    for key in requested & rows_by_group.keys():
                        for row in rows_by_group[key]:
                            if skip is None or not skip[row]:
                                candidates[key].append((buffer, int(row), mass / count))
        missing = requested - candidates.keys()
        if missing:
            raise ValueError(f"No eligible training rows for {len(missing)} AWR coverage groups: {sorted(missing)[:5]}")
        return candidates

    def __call__(self, requests, seed):
        if not requests:
            return
        candidates = self._candidates(set(requests))
        rng = np.random.default_rng(seed)
        for start in range(0, len(requests), self.batch_size):
            selected = {}
            for key in requests[start:start + self.batch_size]:
                entries = candidates[key]
                probability = np.asarray([entry[2] for entry in entries], dtype=np.float64)
                if not np.isfinite(probability).all() or probability.sum() <= 0:
                    raise ValueError(f"Invalid conditional training probabilities for {key}")
                buffer, row, _ = entries[rng.choice(len(entries), p=probability / probability.sum())]
                selected.setdefault(id(buffer), (buffer, []))[1].append(row)
            batch = None
            for buffer, rows in selected.values():
                if hasattr(buffer, "_identity") and hasattr(buffer, "collate"):
                    part = buffer.collate(rows)
                else:
                    part = buffer.sample(len(rows), self.action_chunk_size, indices=rows)
                batch = part if batch is None else concatenate_variable_dim_batch_transitions(batch, part)
            yield batch
