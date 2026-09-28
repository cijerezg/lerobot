"""Survey checks: sample coverage, index integrity and honest aggregation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
from datasets import Dataset

from lerobot.probes import critic_text_swap as report
from lerobot.probes.utils import build_episode_index


def _row(value, next_value=None, *, source='train', split='training', episode=0):
    return dict(value=value, source=source, split=split, episode=episode, task='sort',
                seconds=0, frame_idx=0, subtask='grasp shirt', metadata={},
                swaps={'next': None if next_value is None else {'value': next_value}, 'unrelated': None})


def test_episode_averaging_does_not_weight_long_episodes_or_impute_missing_swaps():
    eps = [{'metrics': report.episode_metrics([_row(-1, -.9)] * 100)},
           {'metrics': report.episode_metrics([_row(-1, 0)])},
           {'metrics': report.episode_metrics([_row(-1), _row(None, -.5)])}]
    summary = report.aggregate(eps)
    assert summary['next_gap'] == pytest.approx(.55)
    assert summary['next_episodes'] == 2
    assert summary['next_pairs'] == 101
    assert summary['failed_values'] == 1
    assert summary['unrelated_gap'] is None


def test_bulk_episode_index_keeps_selected_row_order_without_reading_observations():
    hf = Dataset.from_dict({'episode_index': [0, 5, 5, 9], 'unused': ['a', 'b', 'c', 'd']})
    hf = hf.select([3, 1, 2])
    def forbidden(_):
        raise AssertionError('Observation transform should not run for metadata indexing')
    hf.set_transform(forbidden)
    assert build_episode_index(NS(hf_dataset=hf)) == {9: [0], 5: [1, 2]}


def test_round_robin_sampling_covers_groups_and_respects_episode_selection(tmp_path):
    path = Path(__file__).resolve().parents[3] / 'migration/critic_swap_survey.py'
    spec = importlib.util.spec_from_file_location('survey_test_module', path)
    module = importlib.util.module_from_spec(spec)
    # dataclass inspects its defining module during class creation.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sources = []
    for name in ('one', 'two'):
        root = tmp_path / name
        (root / 'meta/episodes').mkdir(parents=True)
        (root / 'meta/info.json').write_text('{"fps":30}')
        pd.DataFrame({'episode_index': list(range(5)), 'tasks': [['A'], ['A'], ['A'], ['B'], ['B']],
                      'length': [60]*5}).to_parquet(root / 'meta/episodes/part.parquet')
        sources.append(NS(name=name, root=root, weight=1, episodes=[0, 1, 3]))
    picked = module.select_episodes(sources, 5, 42)
    assert picked == module.select_episodes(sources, 5, 42)
    assert len(picked) == 5
    assert len({(r['source'], tuple(r['tasks'])) for r in picked[:4]}) == 4
    assert len({(r['source'], r['episode']) for r in picked}) == 5
    assert all(r['episode'] in (0, 1, 3) for r in picked)


def test_report_preserves_episode_identity_and_escapes_embedded_text(tmp_path):
    provenance = {'step': 2000, 'checkpoint': '/checkpoint/002000/pretrained_model',
                  'selected_episodes': [{'source': 'train', 'episode': 0}], 'validation_episodes': 1,
                  'sampling': 'test'}
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance))
    root = tmp_path / 'step_00002000/critic_text_swap'
    for source, split in [('train', 'training'), ('validation', 'validation')]:
        row = _row(-1, -.8, source=source, split=split)
        row['subtask'] = '</script><script>bad()</script>'
        folder = root / f'sources/{source}/episode_traces/ep0000'
        folder.mkdir(parents=True)
        (folder / 'critic_values.json').write_text(json.dumps({'records': [row], 'segment_end_seconds': [2]}))
    summary = report.render(tmp_path)
    assert summary['complete'] and summary['episodes'] == 2
    html = (root / 'episode_explorer.html').read_text()
    packed = html.split('<script type="application/json" id="probe-data">')[1].split('</script>')[0]
    assert '<script>bad' not in packed
    assert json.loads(packed)['episodes'][0]['records'][0]['subtask'] == row['subtask']
    manifest = json.loads((root / 'index.json').read_text())
    assert manifest['id'] == 'critic_text_swap'
    assert manifest['panels'][0]['primary']
    path = root / 'sources/train/episode_traces/ep0000/critic_values.json'
    bad = json.loads(path.read_text())
    bad['records'][0]['source'] = 'validation'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Mixed episode identities'):
        report.load_episodes(root)


def test_prepared_episode_camera_masks_override_stored_union_columns(monkeypatch, tmp_path):
    import torch
    from lerobot.probes import utils
    (tmp_path / 'meta').mkdir()
    contract = {'schema_version': 1, 'fps': 30, 'camera_roles': ['external_0', 'external_1', 'wrist_0'],
                'depth_key': None, 'episode_contracts': {'0': {'camera_roles': ['external_1']}}}
    (tmp_path / 'meta/cache_ready.json').write_text(json.dumps(contract))
    observations = {f'observation.images.{camera}': torch.ones(1, 3, 2, 2)
                    for camera in ('external_0', 'external_1', 'wrist_0')}
    monkeypatch.setattr(utils, 'get_frame_data', lambda *_: (observations.copy(), None, None, 'grasp', 'sort', 0, 0))
    monkeypatch.setattr(utils, 'canonical_camera_obs', lambda obs, cfg: obs)
    cfg = NS(policy=NS(pointmap_config=None, memory=None, depth_gripper_event_loss=None))
    frame = utils.probe_frame_inputs(NS(root=tmp_path), cfg, 0, 30, metadata={})
    obs, flags = utils.split_probe_complementary(frame['obs'])
    assert set(obs) == {'observation.images.external_1'}
    assert flags['camera_is_present.observation.images.external_1'].item()
    assert not flags['camera_is_present.observation.images.external_0'].item()
    assert not flags['camera_is_present.observation.images.wrist_0'].item()
    filled, inferred = utils.fill_absent_cameras(obs, list(observations))
    assert set(filled) == set(observations)
    assert not filled['observation.images.wrist_0'].any()
    assert not {**inferred, **flags}['camera_is_present.observation.images.wrist_0']
