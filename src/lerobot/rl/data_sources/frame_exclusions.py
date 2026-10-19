"""Reversible, post-cache exclusions in original episode-local frame coordinates.

Relative source_root entries use the training process working directory, exactly
like dataset source roots. Removing the config path restores the complete cache.
"""
from __future__ import annotations

import json
from pathlib import Path


class FrameExclusions:
    def __init__(self, path: str):
        data = json.loads(Path(path).read_text())
        if data.get('version') != 1 or not isinstance(data.get('exclusions'), list):
            raise ValueError('Frame exclusions require version 1 and an exclusions list')
        inactive = data.get('inactive_roots', [])
        if not isinstance(inactive, list) or any(not isinstance(root, str) or not root for root in inactive):
            raise ValueError('inactive_roots must be a list of source-root paths')
        self.inactive_roots = {str(Path(root).resolve()) for root in inactive}
        self.ranges = {}
        self.boundaries = {}
        for kind, target in [('exclusions', self.ranges), ('boundaries', self.boundaries)]:
            entries = data.get(kind, [])
            if not isinstance(entries, list):
                raise ValueError(f'{kind} must be a list')
            for entry in entries:
                root = entry.get('source_root')
                ep = entry.get('episode_index')
                if not isinstance(root, str) or not root or type(ep) is not int or ep < 0:
                    raise ValueError(f'Invalid source_root/episode_index in {kind}')
                key = (str(Path(root).resolve()), ep)
                if kind == 'exclusions':
                    lo, hi = entry.get('from_frame'), entry.get('to_frame')
                    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi:
                        raise ValueError('Exclusions need integer half-open from_frame < to_frame')
                    target.setdefault(key, []).append((lo, hi))
                else:
                    frame = entry.get('frame')
                    if type(frame) is not int or frame <= 0:
                        raise ValueError('Splice boundary frame must be a positive integer')
                    target.setdefault(key, []).append(frame)

    def validate_sources(self, active_roots):
        """Validate even dormant ledger roots, so a misspelling cannot be a no-op."""
        import logging
        from lerobot.rl.data_sources.episode_groups import read_episode_ranges
        active = {str(Path(root).resolve()) for root in active_roots}
        roots = {key[0] for key in self.ranges.keys() | self.boundaries.keys()}
        for root in sorted(roots):
            self.validate_root(root, read_episode_ranges(root))
            if root not in active and root not in self.inactive_roots:
                raise ValueError(f'Manifest root has no active buffer and is not in inactive_roots: {root}')
            if root not in active:
                logging.getLogger(__name__).info(
                    '[FrameExclusions] Validated inactive dataset root (not sampled): %s', root)

    def has_root(self, root):
        root = str(Path(root).resolve())
        return any(key[0] == root for key in self.ranges.keys() | self.boundaries.keys())

    def validate_root(self, root, episode_ranges):
        root = str(Path(root).resolve())
        for key in self.ranges.keys() | self.boundaries.keys():
            if key[0] != root:
                continue
            ep = key[1]
            if ep not in episode_ranges:
                raise ValueError(f'Exclusion references unknown episode {ep} in {root}')
            lo, hi = episode_ranges[ep]
            if any(b > hi - lo for a, b in self.ranges.get(key, [])):
                raise ValueError(f'Exclusion exceeds episode {ep} length in {root}')
            if any(b >= hi - lo for b in self.boundaries.get(key, [])):
                raise ValueError(f'Splice boundary exceeds episode {ep} interior in {root}')

    def filter_starts(self, rows, *, root, episode, episode_start, episode_end,
                      chunk_size, history_frames=0):
        key = (str(Path(root).resolve()), episode)
        exclusions = self.ranges.get(key, [])
        boundaries = self.boundaries.get(key, [])
        if not exclusions and not boundaries:
            return rows
        # Exclusions affect action supervision only. Failed-attempt observations
        # remain legitimate recovery context; no frames are removed or joined.
        right = rows + chunk_size
        valid = right <= episode_end
        for lo, hi in exclusions:
            valid &= (right <= episode_start + lo) | (rows >= episode_start + hi)
        # Existing source splices are different: neither action targets nor
        # conditioning history may straddle the discontinuity.
        left = (rows - history_frames).clamp(min=episode_start)
        for boundary in boundaries:
            boundary += episode_start
            valid &= (right <= boundary) | (left >= boundary)
        return rows[valid]


class DiverseExclusions:
    """Diverse-corpus counterpart: episode-local [from_s, to_s) ranges keyed by episode_id."""

    def __init__(self, path: str):
        data = json.loads(Path(path).read_text())
        if data.get('version') != 1 or not isinstance(data.get('exclusions'), list):
            raise ValueError('Diverse exclusions require version 1 and an exclusions list')
        self.ranges = {}
        for entry in data['exclusions']:
            episode_id, lo, hi = entry.get('episode_id'), entry.get('from_s'), entry.get('to_s')
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError('Diverse exclusions need an episode_id')
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or not 0 <= lo < hi:
                raise ValueError(f'Diverse exclusion for {episode_id} needs 0 <= from_s < to_s')
            self.ranges.setdefault(episode_id, []).append((float(lo), float(hi)))

    def excluded_rows(self, rows, window_s: float) -> set[int]:
        """Rows whose action window [anchor_s, anchor_s + window_s) overlaps an excluded range.

        Every manifest episode must be in the selection, so a stale id cannot be a silent no-op.
        """
        missing = set(self.ranges) - {str(row['episode_id']) for row in rows}
        if missing:
            raise ValueError(f'Diverse exclusions reference episodes outside the selection: {sorted(missing)[:5]}')
        out = set()
        for index, row in enumerate(rows):
            start = float(row['anchor_s'])
            if any(start < hi and start + window_s > lo for lo, hi in self.ranges.get(str(row['episode_id']), [])):
                out.add(index)
        return out


def validate_exclusion_training_mode(cfg):
    """This integration intentionally supports actor-only diverse training only."""
    diverse = getattr(cfg, 'diverse', None)
    if not (getattr(diverse, 'rebot_exclusions_manifest', None) or getattr(diverse, 'diverse_exclusions_manifest', None)):
        return
    if not getattr(diverse, 'enabled', False):
        raise ValueError('Frame exclusions require diverse.enabled; refusing to ignore manifest')
    if not getattr(cfg, 'skip_critic', False):
        raise ValueError('Frame exclusions currently require skip_critic=true; critic targets are unsupported')
    if (getattr(cfg, 'val_split', 0) or getattr(cfg, 'val_freq', 0)
            or getattr(cfg, 'val_on_start', False)):
        raise ValueError('Frame exclusions do not yet support validation; set val_split=0, val_freq=0, val_on_start=false')
