"""Sampling-only whole-episode groups over a shared, complete ReBot replay cache."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

logger = logging.getLogger(__name__)


def read_episode_ranges(root: str) -> dict[int, tuple[int, int]]:
    files = sorted((Path(root) / 'meta' / 'episodes').rglob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No episode metadata under {root}')
    table = pd.concat([pd.read_parquet(p, columns=[
        'episode_index', 'dataset_from_index', 'dataset_to_index'
    ]) for p in files])
    if table.episode_index.duplicated().any():
        raise ValueError(f'Duplicate episode metadata in {root}')
    return {int(r.episode_index): (int(r.dataset_from_index), int(r.dataset_to_index))
            for r in table.itertuples()}


class EpisodeGroupBuffer:
    """Select eligible starts within whole episodes, leaving the underlying cache intact."""

    def __init__(self, buffer, indices: torch.Tensor, group: str, chunk_size: int):
        self.buffer = buffer
        self.sampling_indices = indices
        self.sampling_group = group
        self.action_chunk_size = chunk_size

    def _positions(self, batch_size: int) -> torch.Tensor:
        # iid draws over the eligible starts
        return torch.randint(len(self.sampling_indices), (batch_size,), device=self.sampling_indices.device)

    def __len__(self):
        return len(self.sampling_indices)

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def sample(self, batch_size: int, action_chunk_size: int = 30, *, indices=None):
        if action_chunk_size != self.action_chunk_size:
            raise ValueError('Episode groups must be rebuilt when action_chunk_size changes')
        if indices is None:
            indices = self.sampling_indices[self._positions(batch_size)]
        else:
            indices = torch.as_tensor(indices, device=self.sampling_indices.device)
            if not bool(torch.isin(indices, self.sampling_indices).all()):
                raise ValueError('Explicit indices fall outside the episode sampling group')
        return self.buffer.sample(batch_size, action_chunk_size=action_chunk_size, indices=indices)

    def get_iterator(self, batch_size, async_prefetch=False, queue_size=2, action_chunk_size=30):
        """Keep the view restriction even for callers using the iterator interface."""
        while True:
            yield self.sample(batch_size, action_chunk_size=action_chunk_size)


def episode_group_views(buffer, groups: dict[str, list[int]], chunk_size: int, exclusions=None):
    """Return (group, view, mass), rejecting omissions and overlaps. mass = the view's eligible-start count."""
    source = buffer.offline_source
    if source.episodes is not None:
        raise ValueError('episode_groups requires a complete dataset, not an episodes subset')
    ranges = read_episode_ranges(source.root)
    if exclusions is not None:
        exclusions.validate_root(source.root, ranges)
        if exclusions.has_root(source.root) and buffer.actions.ndim != 2:
            raise ValueError('Frame exclusions require dense per-frame actions')
    history = getattr(buffer, 'history_offsets', None) or {}
    history_frames = max((max(offsets, default=0) for offsets in history.values()), default=0)
    assigned = [ep for episodes in groups.values() for ep in episodes]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(ranges):
        raise ValueError(f'{source.name}: episode_groups must assign every episode exactly once')
    ordered = sorted(ranges.values())
    if (not ordered or ordered[0][0] != 0 or ordered[-1][1] != buffer.size
            or any(a >= b for a, b in ordered)
            or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:]))):
        raise ValueError(f'{source.name}: episode ranges do not cover the complete replay cache')
    stride = buffer.image_stride
    if chunk_size < 1 or chunk_size % stride:
        raise ValueError('action_chunk_size must be positive and divisible by image_stride')
    high = buffer.size - 1 if buffer.optimize_memory and buffer.size < buffer.capacity else buffer.size
    for group, episodes in groups.items():
        starts = []
        for ep in episodes:
            lo, hi = ranges[ep]
            # Dense action chunks must remain inside their physical episode. The
            # episode's last done is allowed only at the last action of the chunk.
            stop = min(high, hi - chunk_size + 1) if buffer.actions.ndim == 2 else min(high, hi)
            start = ((lo + stride - 1) // stride) * stride
            rows = torch.arange(start, max(start, stop), stride, device=buffer.storage_device)
            if buffer.actions.ndim == 2:
                valid = torch.ones_like(rows, dtype=torch.bool)
                for offset in range(chunk_size - 1):
                    valid &= ~buffer.dones[rows + offset].bool()
                rows = rows[valid]
            if not len(rows):
                raise ValueError(f'{source.name}: episode {ep} has no eligible chunk starts')
            if exclusions is not None:
                rows = exclusions.filter_starts(
                    rows, root=source.root, episode=ep, episode_start=lo, episode_end=hi,
                    chunk_size=chunk_size, history_frames=history_frames)
            if len(rows):
                starts.append(rows)
        if not starts and exclusions is not None:
            logger.info('[EpisodeGroups] %s / %s: no starts remain after exclusions', source.name, group)
            continue
        if not starts:
            raise ValueError(f'{source.name}: empty episode group {group!r}')
        indices = torch.cat(starts)
        logger.info('[EpisodeGroups] %s / %s: %d whole episodes, %d eligible starts',
                    source.name, group, len(episodes), len(indices))
        yield group, EpisodeGroupBuffer(buffer, indices, group, chunk_size), len(indices)
