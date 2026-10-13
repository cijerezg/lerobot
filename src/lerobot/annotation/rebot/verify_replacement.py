"""Verify an episode replacement independently of the dataset assembler.

Checks every preserved numeric row and annotation, media hardlinks, transition
masking, and actual RGB/depth/subtask loading. Missing manual labels stay missing.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

GLOBAL_COLUMNS = ('index', 'from_index', 'to_index', 'raw_from_index', 'raw_to_index', 'commit_index')


def table(path):
    return pd.concat([pd.read_parquet(p) for p in sorted(path.rglob('*.parquet'))], ignore_index=True)


def mapped(rows, episode, delta):
    rows = rows.copy()
    rows['episode_index'] = episode
    for key in GLOBAL_COLUMNS:
        if key in rows:
            nonnegative = rows[key].notna() & (rows[key] >= 0)
            rows.loc[nonnegative, key] = rows.loc[nonnegative, key] + delta
    return rows.reset_index(drop=True)


def check(root, report, config):
    root = root.resolve()
    manifest = json.loads((root / 'REPLACEMENT.json').read_text())
    mapping = manifest['episodes']
    base = Path(manifest['base']).resolve()
    added = {int(m['episode_index']) for m in mapping if Path(m['source']).resolve() != base}
    expected_frames = sum(int(m['length']) for m in mapping)
    data, eps = table(root / 'data'), table(root / 'meta/episodes')
    info = json.loads((root / 'meta/info.json').read_text())
    assert len(data) == info['total_frames'] == manifest['total_frames'] == expected_frames
    assert len(eps) == info['total_episodes'] == manifest['total_episodes'] == len(mapping)
    assert data['index'].tolist() == list(range(len(data)))
    assert eps.episode_index.tolist() == list(range(len(mapping)))
    windows = json.loads((root / 'meta/subtask_windows.json').read_text())['episodes']
    tasks = pd.read_parquet(root / 'meta/tasks.parquet').reset_index().set_index('task_index').iloc[:, 0]
    sources = {}
    manual_counts = {}
    depth_count = 0
    checked_tables = set()
    trace = np.load(root / 'meta/speed_hybrid_v1_trace.npz')
    trace_mask = np.zeros(len(data) - 1, dtype=bool)
    for m in mapping:
        source = Path(m['source'])
        if str(source) not in sources:
            sources[str(source)] = (table(source / 'data'), table(source / 'meta/episodes'))
        old, old_eps = sources[str(source)]
        sep, ep, length = int(m['source_episode']), int(m['episode_index']), int(m['length'])
        start, old_start = int(m['new_from_index']), int(m['old_from_index'])
        delta = start - old_start
        before = old[old.episode_index == sep].reset_index(drop=True)
        after = data[data.episode_index == ep].reset_index(drop=True)
        assert len(before) == len(after) == length
        assert after.frame_index.tolist() == list(range(length))
        assert after['index'].tolist() == list(range(start, start + length))
        numeric_keys = ['action', 'observation.state', 'timestamp', 'frame_index']
        pd.testing.assert_frame_equal(before[numeric_keys], after[numeric_keys], check_dtype=False)
        for key in ('action', 'observation.state'):
            values = np.stack(after[key])
            assert values.shape == (length, 7) and np.isfinite(values).all()
        stasks = pd.read_parquet(source / 'meta/tasks.parquet').reset_index().set_index('task_index').iloc[:, 0]
        assert before.task_index.map(stasks).tolist() == after.task_index.map(tasks).tolist()
        e = eps[eps.episode_index == ep].iloc[0]
        oe = old_eps[old_eps.episode_index == sep].iloc[0]
        assert list(e.tasks) == list(oe.tasks)
        assert (int(e.dataset_from_index), int(e.dataset_to_index)) == (start, start + length)
        sw = json.loads((source / 'meta/subtask_windows.json').read_text())['episodes'][str(sep)]
        expected_windows = [{**w, 'from_index': w['from_index'] + delta, 'to_index': w['to_index'] + delta} for w in sw]
        assert windows[str(ep)] == expected_windows, ('subtasks', ep)
        cover = np.zeros(length, dtype=np.int8)
        for w in windows[str(ep)]:
            assert start <= w['from_index'] < w['to_index'] <= start + length
            cover[w['from_index'] - start:w['to_index'] - start] += 1
        assert (cover == 1).all(), ('subtask coverage', ep)
        for f in sorted((source / 'meta').glob('*.parquet')):
            src = pd.read_parquet(f)
            if 'episode_index' not in src:
                continue
            src = src[src.episode_index == sep]
            expected = mapped(src, ep, delta)
            dst = pd.read_parquet(root / 'meta' / f.name)
            dst = dst[dst.episode_index == ep].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, dst[list(expected.columns)], check_dtype=False, check_index_type=False)
            checked_tables.add(f.name)
            for k in ('index', 'from_index', 'raw_from_index', 'commit_index'):
                if k in dst and len(dst):
                    v = dst[k].dropna()
                    assert ((v >= start) & (v < start + length)).all(), (ep, f.name, k)
            for k in ('to_index', 'raw_to_index'):
                if k in dst and len(dst):
                    v = dst[k].dropna()
                    assert ((v > start) & (v <= start + length)).all(), (ep, f.name, k)
            if 'frame_index' in dst:
                assert ((dst.frame_index >= 0) & (dst.frame_index < length)).all()
            if f.name == 'episode_metadata.parquet':
                manual_counts[str(ep)] = len(dst)
            if ep in added and f.name in ('episode_metadata.parquet', 'contact.parquet', 'precision.parquet', 'speed.parquet', 'speed_hybrid_v1.parquet'):
                cov = np.zeros(length, dtype=np.int8)
                for r in dst.itertuples():
                    cov[int(r.from_index) - start:int(r.to_index) - start] += 1
                assert (cov == 1).all(), ('new manual coverage', f.name)
        source_info = json.loads((source / 'meta/info.json').read_text())
        camera_map = {k: m['camera_map'].get(k, k) for k, v in source_info['features'].items() if v.get('dtype') == 'video'}
        for source_key, dest_key in camera_map.items():
            prefix, old_prefix = f'videos/{dest_key}', f'videos/{source_key}'
            video = root / prefix / f'chunk-{int(e[prefix + "/chunk_index"]):03d}' / f'file-{int(e[prefix + "/file_index"]):03d}.mp4'
            original = source / old_prefix / f'chunk-{int(oe[old_prefix + "/chunk_index"]):03d}' / f'file-{int(oe[old_prefix + "/file_index"]):03d}.mp4'
            assert os.path.samefile(video, original), ('video', ep, dest_key)
            assert e[prefix + '/from_timestamp'] == oe[old_prefix + '/from_timestamp']
        depth_name = 'wrist_0.depth' if (source / 'depth/wrist_0.depth').exists() else 'wrist.depth'
        old_depth = source / 'depth' / depth_name / f'episode-{sep:06d}'
        new_depth = root / 'depth/wrist_0.depth' / f'episode-{ep:06d}'
        old_pngs = sorted(old_depth.glob('*.png'))
        assert len(old_pngs) == len(list(new_depth.glob('*.png'))) > 0
        for p in old_pngs:
            assert os.path.samefile(p, new_depth / p.name)
        depth_count += len(old_pngs)
        source_trace = source / 'meta/speed_hybrid_v1_trace.npz'
        if source_trace.exists():
            ot = np.load(source_trace)
            # A historical append omitted its entire final episode from this trace.
            available = max(0, min(length - 1, len(ot['speed']) - old_start))
            if available:
                sl, osl = slice(start, start + available), slice(old_start, old_start + available)
                trace_mask[sl] = True
                for key in ot.files:
                    if ot[key].ndim:
                        np.testing.assert_equal(trace[key][sl], ot[key][osl])
        print(f'PASS preserved episode {ep} <- {source.name}:{sep}', flush=True)
    assert all(trace[k].shape == (len(data) - 1,) for k in trace.files if trace[k].ndim)
    assert not trace['valid'][~trace_mask].any()
    assert not trace['supervision_mask'][~trace_mask].any()
    assert (trace['speed'][~trace_mask] == 0).all()
    for key in trace.files:
        if trace[key].ndim and np.issubdtype(trace[key].dtype, np.floating):
            assert np.isnan(trace[key][~trace_mask]).all(), key
    missing_manual = sorted(int(ep) for ep, count in manual_counts.items() if count == 0)
    assert all(manual_counts.get(str(ep), 0) > 0 for ep in added)
    mistakes = pd.read_parquet(root / 'meta/mistakes.parquet')
    spans = pd.read_parquet(root / 'meta/quality_spans.parquet')
    for m in mistakes[mistakes.episode_index.isin(added)].itertuples():
        low = spans[(spans.episode_index == m.episode_index) & (spans.quality <= 2) & (spans.raw_from_index <= m.from_index) & (spans.raw_to_index >= m.to_index)]
        assert len(low), ('uncovered new mistake', m)
    config_data = yaml.safe_load(config.read_text())
    training_roots = [Path(s['root']).resolve() for s in config_data['dataset']['sources']]
    assert root not in training_roots
    for m in mapping:
        if int(m['episode_index']) not in added:
            continue
        source = Path(m['source']).resolve()
        assert source not in training_roots
        raw_provenance = json.loads((source / 'meta/provenance.json').read_text())
        assert all(Path(p['source']).resolve() not in training_roots for p in raw_provenance)

    # Exercise the same readers used by probes, not only parquet metadata.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.probes.utils import get_frame_data
    from lerobot.rl.offline_dataset_utils import _subtask_indices_from_windows, load_metadata_rows
    from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png
    from lerobot.utils.depth_gripper_events import load_depth_gripper_event_targets
    ds = LeRobotDataset('cijerezg/' + root.name, root=root, video_backend='pyav', download_videos=False)
    assert len(ds) == len(data)
    targets = load_depth_gripper_event_targets(ds)
    assert all(len(x) == len(data) and torch.isfinite(x).all() for x in targets.values())
    subtask_indices = _subtask_indices_from_windows(ds, len(data)).numpy().reshape(-1)
    subtask_ids = pd.read_parquet(root / 'meta/subtasks.parquet')['subtask_index']
    for ep_windows in windows.values():
        for w in ep_windows:
            assert (subtask_indices[w['from_index']:w['to_index']] == subtask_ids.loc[w['subtask']]).all()
    segments, _, _, _, _ = load_metadata_rows(root)
    quality_coverage = np.zeros(len(data), dtype=np.int8)
    for s in segments:
        assert 1 <= s['quality'] <= 5
        quality_coverage[s['from_index']:s['to_index']] += 1
    for e in eps.itertuples(index=False):
        start, end = int(e.dataset_from_index), int(e.dataset_to_index)
        if int(e.episode_index) in missing_manual:
            assert not quality_coverage[start:end].any()
        else:
            assert (quality_coverage[start:end] == 1).all()
    samples = []
    zero_depth_samples = []
    for e in eps.itertuples(index=False):
        ep = int(e.episode_index)
        entry = eps[eps.episode_index == ep].iloc[0]
        start, length = int(entry.dataset_from_index), int(entry.length)
        for local in (0, length // 2 // 3 * 3, length - 3):
            obs, actions, state, subtask, task, got_ep, frame = get_frame_data(ds, start + local, 30)
            assert int(got_ep) == ep and int(frame) == local
            assert actions.shape == (30, 7) and state.shape == (7,)
            assert task == list(entry.tasks)[0]
            expected = next(w['subtask'] for w in windows[str(ep)] if w['from_index'] <= start + local < w['to_index'])
            assert subtask == expected, (ep, local, subtask, expected)
            for key in ('observation.images.external_0', 'observation.images.wrist_0'):
                assert obs[key].shape == (1, 3, 480, 640) and torch.isfinite(obs[key]).all()
            depth = load_depth_png(root, 'wrist_0.depth', ep, local)
            assert depth.shape == (480, 640) and depth.dtype == np.uint16
            if not (depth > 0).any():
                assert ep not in added, 'New episode depth must be populated'
                zero_depth_samples.append({'episode': ep, 'frame': local})
            samples.append({'episode': ep, 'frame': local, 'subtask': subtask})
        print(f'PASS decoded RGB/depth/subtasks episode {ep}', flush=True)
    result = {'status': 'passed', 'root': str(root), 'episodes': len(eps), 'frames': len(data), 'retained_episodes': len(mapping) - len(added),
              'excluded_base_episodes': manifest.get('excluded_base_episodes', []), 'new_episodes': sorted(added),
              'new_manual_segments': sum(manual_counts[str(ep)] for ep in added),
              'new_mistakes': int(mistakes.episode_index.isin(added).sum()), 'checked_annotation_tables': sorted(checked_tables),
              'linked_depth_files': depth_count, 'decoded_samples': samples, 'preserved_zero_depth_samples': zero_depth_samples,
              'training_root_separation': True, 'speed_unknown_transitions_masked': int((~trace_mask).sum()),
              'limitations': [f'Retained episodes {missing_manual} have no manual quality, precision, contact, or speed annotations; missing labels remain missing.', f'Retained episodes {sorted({s["episode"] for s in zero_depth_samples})} have zero depth at sampled frames, identical to their original source.']}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config_rl.yaml'))
    args = parser.parse_args()
    check(args.root, args.report, args.config)
