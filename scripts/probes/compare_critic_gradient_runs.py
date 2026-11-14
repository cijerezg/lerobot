#!/usr/bin/env python3
"""Match saved gradient surveys by frame and write an inspectable HTML comparison."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def compare(old_root, new_root):
    old = json.loads((old_root / 'gradient_points.json').read_text())
    new = json.loads((new_root / 'gradient_points.json').read_text())
    def key(data, r):
        ep = data['episodes'][r['episode_ref']]
        return ep['source'], ep['episode'], r['global_idx']
    old_rows = {key(old, r): r for r in old['records']}
    new_rows = {key(new, r): r for r in new['records']}
    if len(old_rows) != len(old['records']) or set(old_rows) != set(new_rows):
        raise ValueError('Comparison requires exactly the same unique frames')
    paired = []
    for r in new['records']:
        k = key(new, r); previous = old_rows[k]
        for field in ('subtask', 'segment_progress', 'seconds_to_end', 'metadata', 'terminal'):
            if previous[field] != r[field]:
                raise ValueError(f'Comparison frame {k} differs in {field}')
        paired.append(dict(episode=f'{k[0]}/{k[1]}', phase=r['phase'], progress=r['segment_progress'],
                           old=previous['image_state_grad_norm'], new=r['observation_grad_norm'],
                           old_rgb=previous['image_grad_norm'], new_rgb=r['image_grad_norm'],
                           old_state=previous['state_value_grad_norm'], new_state=r['state_value_grad_norm'],
                           depth=r['depth_grad_norm'], depth_consumed=r['raw_depth_consumed'],
                           old_value=previous['value'], new_value=r['value'],
                           old_adv=previous['advantage'], new_adv=r['advantage'],
                           terminal=r['terminal']))
    def rho(a, b):
        valid = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(valid) < 3 or len(set(x for x, _ in valid)) < 2 or len(set(y for _, y in valid)) < 2:
            return None
        return float(spearmanr(*zip(*valid)).statistic)
    actions = ['all'] + sorted({r['phase'] for r in paired})
    summaries = []
    for action in actions:
        rows = paired if action == 'all' else [r for r in paired if r['phase'] == action]
        col = lambda k: [r[k] for r in rows]
        n = max(1, math.ceil(len(rows) / 10))
        # Stable identities break ties only for this set-overlap summary.
        top = lambda k: set(sorted(range(len(rows)), key=lambda i: (-rows[i][k], i))[:n])
        summary = dict(action=action, frames=len(rows), episodes=len({r['episode'] for r in rows}),
                       paired_rank=rho(col('old'), col('new')), rgb_paired_rank=rho(col('old_rgb'), col('new_rgb')),
                       top_decile_overlap=len(top('old') & top('new')) / n,
                       old_progress=rho(col('old'), col('progress')), new_progress=rho(col('new'), col('progress')),
                       old_rgb_progress=rho(col('old_rgb'), col('progress')), new_rgb_progress=rho(col('new_rgb'), col('progress')),
                       old_value=rho(col('old'), col('old_value')), new_value=rho(col('new'), col('new_value')),
                       old_abs_adv=rho(col('old'), [None if r['old_adv'] is None or r['terminal'] else abs(r['old_adv']) for r in rows]),
                       new_abs_adv=rho(col('new'), [None if r['new_adv'] is None or r['terminal'] else abs(r['new_adv']) for r in rows]),
                       value_mae=float(np.mean(np.abs(np.array(col('old_value')) - col('new_value')))),
                       medians={k: float(np.median(col(k))) for k in ('old', 'new', 'old_rgb', 'new_rgb', 'old_state', 'new_state', 'depth')})
        summaries.append(summary)
    parent = new_root.parents[1]
    check = json.loads((parent / 'depth_input_check.json').read_text())
    encoder = json.loads((parent / 'encoder_comparison.json').read_text())
    depth_rows = [r for r in paired if r['depth_consumed']]
    share = np.array([(r['depth'] / r['new'])**2 for r in depth_rows])
    real_depth_checks = [r for r in check['records'] if r['raw_depth_consumed']]
    delta = np.abs([r['depth_null_delta'] for r in real_depth_checks])
    info = dict(frames=len(paired), episodes=len(new['episodes']),
                depth_frames=len(depth_rows), depth_squared_share_percentiles=np.quantile(share, [.1, .5, .9, 1]).tolist(),
                depth_null_frames=len(delta), depth_null_nonzero=int(sum(delta > 0)),
                depth_near_bound_fraction_percentiles=np.quantile([r['input_scales']['depth']['fraction_near_bound'] for r in real_depth_checks], [0, .5, 1]).tolist(),
                depth_null_abs_delta_percentiles=np.quantile(delta, [.5, .9, 1]).tolist(),
                input_rms_medians={k:float(np.median([r['input_scales'][k]['rms'] for r in real_depth_checks])) for k in ('rgb', 'state', 'depth')},
                encoder_comparison=encoder, depth_checks=check["records"], actions=summaries,
                old_checkpoint=old['checkpoint'], new_checkpoint=new['checkpoint'])
    (new_root / 'architecture_comparison.json').write_text(json.dumps(info, indent=2, allow_nan=False))
    stats = Path('lerobot/src/lerobot/probes/critic_gradient_stats.js').read_text()
    template = Path(__file__).with_suffix('.html').read_text()
    body = template.replace('__STATS__', stats).replace('__DATA__', json.dumps(dict(info=info, pairs=paired)).replace('<', '\\u003c'))
    (new_root / 'architecture_comparison.html').write_text(body)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('old_root', type=Path)
    parser.add_argument('new_root', type=Path)
    args = parser.parse_args()
    compare(args.old_root, args.new_root)
