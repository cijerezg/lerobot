"""Explore saved full-input gradients by task, phase, value and TD advantage."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace


COLORS = re.compile(r'\b(?:beige|black|blue|brown|gray|green|grey|navy|orange|pink|purple|red|white|yellow)\b\s*', re.I)


def image_state_gradient_fields(gradient):
    """Compute disjoint RGB/state/depth norms and their observation norm.

    State values are the discrete <state_N> tokens, or the continuous state's
    <extra_0> position, explicitly confirmed to carry the state_projector output.
    State clause wording, delimiters, depth placeholders and other prompt tokens
    are excluded. The token blocks are disjoint, so their squared norms add.
    """
    value_token = {'discrete': r'<state_\d+>', 'continuous': r'<extra_0>'}.get(gradient.get('state_format'))
    if value_token is None:
        raise ValueError('Image/state report requires saved discrete or continuous state-value token gradients')
    if gradient.get('state_format') == 'continuous' and not gradient.get('continuous_state_consumed'):
        raise ValueError('Continuous-state gradient requires confirmation that projected state was consumed')
    state = [t['norm'] for t in gradient['tokens']
             if t['group'] == 'state' and re.fullmatch(value_token, t['text'])]
    images = [group['norm'] for name, group in gradient['groups'].items()
              if (name.startswith('img_') or name == 'other_image_patches') and group['norm'] is not None]
    if not state or not images:
        raise ValueError('Saved record has no state-value tokens or RGB patch gradients')
    if any(not math.isfinite(v) or v < 0 for v in state + images):
        raise ValueError('Invalid saved component gradient norm')
    image_norm, state_norm = math.hypot(*images), math.hypot(*state)
    combined = math.hypot(image_norm, state_norm)
    depth = gradient['groups'].get('depth', {}).get('norm')
    if gradient.get('raw_depth_consumed') and depth is None:
        raise ValueError('Consumed depth is missing its gradient group')
    depth_norm = float(depth) if gradient.get('raw_depth_consumed') else 0.0
    if not math.isfinite(depth_norm) or depth_norm < 0:
        raise ValueError('Invalid depth gradient norm')
    observation_norm = math.hypot(combined, depth_norm)
    if observation_norm > gradient['norm'] * (1 + 2e-5) + 1e-10:
        raise ValueError('Restricted gradient exceeds the full input norm')
    return dict(image_grad_norm=image_norm, state_value_grad_norm=state_norm,
                image_state_grad_norm=combined, depth_grad_norm=depth_norm,
                observation_grad_norm=observation_norm, state_value_token_count=len(state),
                raw_depth_consumed=gradient.get('raw_depth_consumed'),
                raw_history_consumed=gradient.get('raw_history_consumed'))


def refresh_image_state_gradients(output_dir):
    """Add exact restricted norms to saved points; no model or target reconstruction."""
    root = Path(output_dir)
    path = root / 'gradient_points.json'
    data = json.loads(path.read_text())
    by_episode = {}
    for row in data['records']:
        by_episode.setdefault(row['episode_ref'], []).append(row)
    for ref, points in by_episode.items():
        ep = data['episodes'][ref]
        saved = json.loads((root / ep['file']).read_text())['records']
        for point in points:
            row = saved[point['record_index']]
            if (row['global_idx'], row['episode'], row['subtask'], row['source']) != (
                    point['global_idx'], ep['episode'], point['subtask'], ep['source']):
                raise ValueError('Frame identity mismatch while restricting gradient inputs')
            point.update(image_state_gradient_fields(row['gradient']))
    scope = 'observation' if any(r['raw_depth_consumed'] for r in data['records']) else 'images_state'
    data['summary']['gradient_scope'] = scope
    data.setdefault('definitions', {})['image_state_grad_norm'] = (
        'L2 over RGB image patch embeddings and state-value embeddings (<state_N> tokens, or the continuous '
        '<extra_0> placeholder plus its state projection) entering critic fusion; '
        'excludes fixed state wording/delimiters, other text, metadata, depth/history placeholders')
    path.write_text(json.dumps(data, allow_nan=False, separators=(',', ':')))
    return {'frames': len(data['records']), 'scope': scope,
            'raw_depth_consumed': sorted({r['raw_depth_consumed'] for r in data['records']})}


def transition_metrics(row, following, reward, done, *, chunk, discount, v_min, v_max):
    """Pair only matching saved forwards; terminals need no successor value."""
    result = dict(reward=float(reward), terminal=bool(done), delta_value=None,
                  advantage=None, target_clipped=None, transition_status='missing_next_frame')
    if done:
        target = reward
        result['transition_status'] = 'terminal'
    else:
        if following is None or following['global_idx'] - row['global_idx'] != chunk:
            return result
        for field in ('source', 'episode', 'segment_end', 'task', 'subtask', 'metadata'):
            if following[field] != row[field]:
                result['transition_status'] = 'conditioning_changes'
                return result
        if [im['camera'] for im in following['images']] != [im['camera'] for im in row['images']]:
            result['transition_status'] = 'camera_inputs_change'
            return result
        result['delta_value'] = float(following['value'] - row['value'])
        target = reward + discount * following['value']
        result['transition_status'] = 'paired'
    clipped = max(v_min, min(v_max, target))
    result.update(advantage=float(clipped - row['value']), target_clipped=bool(clipped != target))
    return result


class _MetadataDataset:
    """Only frame indices, sufficient for the shared reward/terminal builder."""
    def __init__(self, root, episode_ids):
        from datasets import Dataset
        self.root = root
        self.hf_dataset = Dataset.from_dict({'episode_index': episode_ids})

    def __len__(self):
        return len(self.hf_dataset)


def _source_context(root, cfg, chunk):
    import numpy as np
    import pandas as pd
    from lerobot.probes.critic import _training_targets

    tables = sorted((root / 'meta/episodes').rglob('*.parquet'))
    episodes = pd.concat([pd.read_parquet(p, columns=['episode_index', 'dataset_from_index', 'dataset_to_index'])
                          for p in tables], ignore_index=True)
    n = int(episodes.dataset_to_index.max())
    episode_ids = np.full(n, -1, dtype=np.int64)
    bounds = {}
    for row in episodes.to_dict('records'):
        ep, start, stop = int(row['episode_index']), int(row['dataset_from_index']), int(row['dataset_to_index'])
        episode_ids[start:stop] = ep
        bounds[ep] = (start, stop)
    if (episode_ids < 0).any():
        raise ValueError(f'Incomplete episode index metadata: {root}')
    dataset = _MetadataDataset(root, episode_ids)
    targets = _training_targets(dataset, cfg, chunk)
    if targets['mode'] != 'subtask':
        raise ValueError('Saved gradient transition reconstruction requires subtask terminals')
    boundaries = np.flatnonzero(targets['episode_end'] | targets['terminals'])
    fps = float(json.loads((root / 'meta/info.json').read_text())['fps'])
    return dataset, targets, boundaries, bounds, fps


def _load_run_context(root):
    import yaml
    run = next((p for p in root.resolve().parents if (p / 'provenance.json').exists() and (p / 'config.yaml').exists()), None)
    if run is None:
        return None
    provenance = json.loads((run / 'provenance.json').read_text())
    config = yaml.safe_load((run / 'config.yaml').read_text())
    roots = {r['source']: Path(r['root']) for r in provenance['selected_episodes']}
    roots['validation'] = Path(provenance['validation_root'])
    # Config and provenance use project-relative source paths.
    for name, source in roots.items():
        if not source.exists() and not source.is_absolute():
            roots[name] = next((p / source for p in run.parents if (p / source).exists()), source)
    return SimpleNamespace(policy=SimpleNamespace(**config['policy'])), roots


def render_all_gradient_report(output_dir):
    """Rearrange saved forwards and read annotation indices; no model/image decoding."""
    root = Path(output_dir)
    catalog = json.loads((root / 'gradient_episodes.json').read_text())
    records, full_records, identities = [], [], set()
    for episode_ref, episode in enumerate(catalog['episodes']):
        path = Path(episode['file'])
        saved = json.loads((root / path).read_text())
        if catalog.get('checkpoint') and saved.get('checkpoint') != catalog['checkpoint']:
            raise ValueError(f'Checkpoint mismatch: {path}')
        for record_index, row in enumerate(saved['records']):
            identity = (row['source'], row['episode'], row['global_idx'])
            if identity[:2] != (episode['source'], episode['episode']) or identity in identities:
                raise ValueError(f'Invalid or duplicate frame identity: {identity}')
            identities.add(identity)
            point = {key: row.get(key) for key in (
                'frame_idx', 'global_idx', 'seconds', 'subtask', 'grad_norm',
                'value', 'metadata', 'seconds_to_end', 'task', 'segment_end')}
            point.update(episode_ref=episode_ref, record_index=record_index,
                         phase=(row['subtask'].split() or ['unknown'])[0].lower(),
                         family=COLORS.sub('', row['subtask']),
                         norms={key: group['norm'] for key, group in row['gradient']['groups'].items()},
                         images=[dict(image, path=(path.parent / image['path']).as_posix())
                                 for image in row['images']],
                         segment_progress=None, segment_start=None, delta_value=None, advantage=None,
                         terminal=None, reward=None, target_clipped=None, transition_status='annotations_unavailable')
            point.update(image_state_gradient_fields(row['gradient']))
            records.append(point)
            # Do not retain large prompt-token arrays from every episode.
            full_records.append({key: row[key] for key in (
                'source', 'episode', 'global_idx', 'segment_end', 'task', 'subtask', 'metadata', 'images', 'value')})
    if len(records) != catalog['summary']['grad_n_frames']:
        raise ValueError('Saved point count disagrees with the episode catalog')
    context = _load_run_context(root)
    transition_config = None
    if context is not None:
        import numpy as np
        from lerobot.probes.critic import _transition_target
        cfg, roots = context
        p = cfg.policy
        chunk = int(p.chunk_size)
        lookup = {(r['source'], r['episode'], r['global_idx']): r for r in full_records}
        transition_config = {key: getattr(p, key) for key in (
            'chunk_size', 'discount', 'critic_reward_mode', 'reward_normalization_constant',
            'critic_mistake_penalty', 'value_support_min', 'value_support_max')}
        for source in sorted({r['source'] for r in full_records}):
            dataset, targets, boundaries, bounds, fps = _source_context(roots[source], cfg, chunk)
            for point, row in zip(records, full_records, strict=True):
                if row['source'] != source:
                    continue
                g = row['global_idx']
                start, stop = bounds[row['episode']]
                b = int(np.searchsorted(boundaries, g))
                end = int(boundaries[b])
                if end != row['segment_end'] or not start <= g < stop or g - start != point['frame_idx']:
                    raise ValueError(f'Saved frame/terminal disagrees with current annotations: {source}/{g}')
                if targets['labels'].get(g, {}) != row['metadata']:
                    raise ValueError(f'Saved metadata disagrees with current annotations: {source}/{g}')
                if abs(point['seconds_to_end'] - (end - g) / fps) > 1e-5:
                    raise ValueError(f'Saved frame timing disagrees with source fps: {source}/{g}')
                segment_start = max(start, int(boundaries[b - 1]) + 1 if b else start)
                point.update(segment_start=segment_start,
                             segment_progress=(g - segment_start) / max(1, end - segment_start))
                reward, done = _transition_target(targets, dataset, g, chunk,
                                                  p.critic_mistake_penalty, p.reward_normalization_constant, cfg)
                following = lookup.get((source, row['episode'], g + chunk))
                point.update(transition_metrics(row, following, reward, done, chunk=chunk,
                             discount=p.discount, v_min=p.value_support_min, v_max=p.value_support_max))
    coverage = dict(Counter(r['transition_status'] for r in records))
    summary = dict(catalog['summary'], gradient_scope='observation' if any(r['raw_depth_consumed'] for r in records) else 'images_state', advantage_n=sum(r['advantage'] is not None for r in records),
                   delta_value_n=sum(r['delta_value'] is not None for r in records), transition_coverage=coverage)
    payload = dict(catalog, summary=summary, records=records, view='gradient_patterns',
                   transition_config=transition_config,
                   definitions={
                       'advantage': 'clip(reward + discount * (1-terminal) * V(next), support) - V(current)',
                       'delta_value': 'V(next) - V(current), one chunk, identical task/subtask/metadata/cameras, no terminal crossing',
                       'segment_progress': 'Fraction from previous critic terminal + 1 to current terminal; release folded into move',
                       'family': 'Exact subtask text with common colour words removed (display grouping only)',
                       'values': 'Original gradient-enabled forwards; no model reevaluation',
                   })
    packed = json.dumps(payload, allow_nan=False, separators=(',', ':'))
    (root / 'gradient_points.json').write_text(packed)
    template = Path(__file__).with_name('critic_gradient_all.html').read_text()
    script = (Path(__file__).with_name('critic_gradient_stats.js').read_text() + '\n'
              + Path(__file__).with_name('critic_gradient_all.js').read_text())
    (root / 'gradient_explorer.html').write_text(template.replace('__PROBE_SCRIPT__', script).replace('__PROBE_DATA__', packed.replace('<', '\\u003c')))
    manifest_path = root / 'index.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        how = ('Explore all measured frames in gradient order, coloured by task, phase, value or advantage. '
               'Switch to 2D density, choose value, time, progress, delta V or advantage, and click a bin to browse its frames. '
               'Filters and terminal selection share one frame inspector. Missing transition values are excluded and counted.')
        manifest['doc'] = __doc__ + '\n\n' + how
        for panel in manifest['panels']:
            if panel['file'] == 'gradient_explorer.html':
                panel.update(caption='Gradient patterns: sorted points and 2D density', how=how)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return {'points': len(records), 'episodes': len(catalog['episodes']), 'transitions': coverage,
            'advantage_n': summary['advantage_n'], 'delta_value_n': summary['delta_value_n']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path, help='Directory containing gradient_episodes.json')
    parser.add_argument('--components-only', action='store_true', help='Refresh image/state norms in saved points only')
    args = parser.parse_args()
    fn = refresh_image_state_gradients if args.components_only else render_all_gradient_report
    print(json.dumps(fn(args.output_dir)))
