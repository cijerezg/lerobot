"""Speed labels for a new ReBot root, same two files the training roots carry:
  meta/speed.parquet          v4 work-normalized duration (speed_annotate annotate, reference rebot-annot-v3-diverse)
  meta/speed_hybrid_v1.parquet + _trace.npz + _info.json   v7 hybrid motion+duration against the frozen ReBot training pool
                                (outputs/stats/speed_reference_joint_motion_rebot_v1_pool.npz), the table the loader reads.
The reference pool is NOT refit (the new root is an applied root, like rebot_val-annotated-v4 was).
    uv run python -m lerobot.annotation.speed.rebot_speed_pass <root> [<root> ...]
"""
import collections, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from lerobot.annotation.speed import hybrid_motion_speed as hybrid
from lerobot.annotation.speed import speed_annotate as speed

REF = Path("outputs/stats/speed_reference_joint_motion_rebot_v1.json")
POOL = Path("outputs/stats/speed_reference_joint_motion_rebot_v1_pool.npz")
DURATION_REF = "outputs/stats/speed_reference_rebot-annot-v3-diverse.json"
COLUMNS = ['episode_index', 'segment_index', 'from_index', 'to_index', 'subtask', 'class', 'duration_s', 'net_displacement',
           'speed', 'speed_score', 'motion_speed', 'motion_score', 'motion_median_deg_s', 'duration_speed', 'duration_used',
           'speed_source', 'speed_default_reason', 'speed_flags', 'reference_group']

def hybrid_root(root: Path, pool):
    fps = float(json.loads((root / 'meta/info.json').read_text())['fps'])
    states = speed._rebot_states(root)[:, :6]
    segments = pd.read_parquet(root / 'meta/episode_metadata.parquet'); prior = pd.read_parquet(root / 'meta/speed.parquet')
    keys = ['episode_index', 'segment_index', 'from_index', 'to_index']
    assert (segments[keys].to_numpy() == prior[keys].to_numpy()).all(), f'{root.name}: duration labels are not row-aligned'
    n = len(states) - 1
    arrays = {k: np.full(n, np.nan) for k in ('joint_speed_deg_s', 'motion_score', 'duration_score', 'speed_score')}
    supervision = np.zeros(n, dtype=bool); labels = np.zeros(n, dtype=np.uint8); rows = []
    for episode, seg in segments.groupby('episode_index'):
        seg = seg.sort_values('segment_index'); start, stop = int(seg['from_index'].iloc[0]), int(seg['to_index'].iloc[-1])
        q = states[start:stop]; atoms = []
        for i, row in zip(seg.index, seg.itertuples(index=False), strict=True):
            old = prior.loc[i]
            atoms.append({'start_timestep': int(row.from_index) - start, 'end_timestep_exclusive': int(row.to_index) - start,
                          'duration_speed': int(old.speed), 'use_duration': old.speed_source != 'default', 'segment': row, 'prior': old})
        trace = hybrid.hybrid_trace(q, fps, atoms, pool); span = slice(start, start + len(q) - 1)
        for key, source in (('joint_speed_deg_s', 'joint_speed_rad_s'), ('motion_score', 'motion_score'), ('duration_score', 'duration_score'), ('speed_score', 'speed_score')):
            arrays[key][span] = trace[source]
        supervision[span] = trace['supervision_mask']; valid = trace['supervision_mask'] & trace['valid']
        ep_labels = np.where(trace['supervision_mask'], 3, 0).astype(np.uint8); ep_labels[valid] = hybrid.quantize(trace['speed_score'][valid]); labels[span] = ep_labels
        for atom in atoms:
            a, b = atom['start_timestep'], atom['end_timestep_exclusive']; segment, old = atom['segment'], atom['prior']
            scores = trace['speed_score'][a:b - 1]
            reason = 'fewer than two state transitions in the segment' if len(scores) < 2 else ('nonfinite state data' if not np.isfinite(scores).all() else '')
            score = 3. if reason else float(np.median(scores)); motion_score = 3. if reason else float(np.median(trace['motion_score'][a:b - 1]))
            median = None if reason else float(np.median(trace['joint_speed_rad_s'][a:b - 1])); flags = [reason] if reason else []
            if abs(int(hybrid.quantize(motion_score)) - atom['duration_speed']) >= 2:
                flags.append('motion and duration differ by at least two buckets; adjustment capped')
            rows.append({'episode_index': int(segment.episode_index), 'segment_index': int(segment.segment_index), 'from_index': int(segment.from_index),
                         'to_index': int(segment.to_index), 'subtask': segment.subtask, 'class': old['class'], 'duration_s': float(old['duration_s']),
                         'net_displacement': float(old['net_displacement']), 'speed': 3 if reason else int(hybrid.quantize(score)), 'speed_score': score,
                         'motion_speed': 3 if reason else int(hybrid.quantize(motion_score)), 'motion_score': motion_score, 'motion_median_deg_s': median,
                         'duration_speed': atom['duration_speed'], 'duration_used': bool(atom['use_duration']),
                         'speed_source': 'default_unclear' if reason else hybrid.METHOD, 'speed_default_reason': reason, 'speed_flags': flags, 'reference_group': 'rebot'})
    trace_path, table_path = root / 'meta/speed_hybrid_v1_trace.npz', root / 'meta/speed_hybrid_v1.parquet'
    assert not trace_path.exists() and not table_path.exists(), f'{root.name}: hybrid files already exist'
    np.savez_compressed(trace_path, speed=labels, valid=np.isfinite(arrays['speed_score']), supervision_mask=supervision, native_rate_hz=fps, **arrays)
    pd.DataFrame([{k: r[k] for k in COLUMNS} for r in rows]).to_parquet(table_path, engine='pyarrow')
    json.dump({'reference': str(REF), 'method': hybrid.METHOD, 'units': 'degrees/second', 'duration_prior': 'meta/speed.parquet', 'trace': 'speed_hybrid_v1_trace.npz',
               'split': 'applied', 'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'rows': len(rows),
               'shares': {str(k): int(c) for k, c in sorted(collections.Counter(r['speed'] for r in rows).items())}},
              open(root / 'meta/speed_hybrid_v1_info.json', 'w'), indent=2)
    print(f"{root.name}: hybrid {len(rows)} rows, buckets {dict(sorted(collections.Counter(r['speed'] for r in rows).items()))}")

if __name__ == '__main__':
    roots = [Path(r) for r in sys.argv[1:]]
    for root in roots:
        if not (root / 'meta/speed.parquet').exists():
            subprocess.run([sys.executable, '-m', 'lerobot.annotation.speed.speed_annotate', 'annotate', '--reference', DURATION_REF, '--root', str(root)], check=True)
    pool = np.load(POOL)['training_values_deg_s']
    for root in roots:
        hybrid_root(root, pool)
