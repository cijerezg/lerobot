"""MolmoAct review tooling: finalize coverage, and the round-2 top-up (exclusion, quota, round tags)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'examples/dataset/diverse_robot_dataset/prepare_molmoact.py'
spec = importlib.util.spec_from_file_location('prepare_molmoact_review_test', SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def summary(index, task, active_fraction=1.0, path=1.0, discontinuous=False, active_cells=4):
    return {'episode_index': index, 'task': task, 'annotated_task': f'{task} (annotated)', 'active_fraction': active_fraction,
            'xyz_path_length_m': path, 'discontinuous': discontinuous, 'active_cells': active_cells, 'duration_s': 5.0}


def review_paths(tmp_path, summaries, prior=None):
    scan = tmp_path / 'episode_scan.json'
    scan.write_text(json.dumps({'source': 'test/source', 'summaries': summaries}))
    if prior is not None:
        (tmp_path / 'candidates.json').write_text(json.dumps(prior))
    return SimpleNamespace(scan=scan, candidates=tmp_path / 'candidates.json', review_root=tmp_path, component='household')


@pytest.mark.parametrize('episodes,match', [
    ({'1': {'accept': True, 'quality': 5}}, 'missing='),
    ({'1': {'accept': False}, '2': {'accept': False}, '99': {'accept': False}}, 'extra='),
    ({'1': {'accept': False}, '2': {'accept': 'false'}}, 'explicit boolean'),
])
def test_invalid_coverage_leaves_existing_selection_untouched(tmp_path, episodes, match):
    candidates = tmp_path / 'candidates.json'
    candidates.write_text(json.dumps({'candidates': [{'episode_index': 1}, {'episode_index': 2}]}))
    verdicts = tmp_path / 'verdicts.json'
    verdicts.write_text(json.dumps({'episodes': episodes}))
    selection = tmp_path / 'selection.json'
    selection.write_text('{"previously_accepted": [7]}\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    # Metadata intentionally absent: coverage must be checked before metadata or writes.
    paths = SimpleNamespace(candidates=candidates, review_root=tmp_path, metadata_root=tmp_path/'missing_metadata')
    with pytest.raises(ValueError, match=match):
        prepare.finalize(paths, verdicts, 'review test')
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_union_finalize_requires_a_verdict_for_every_round(tmp_path):
    candidates = tmp_path / 'candidates.json'
    candidates.write_text(json.dumps({'candidates': [{'episode_index': 1, 'round': 'v2'}, {'episode_index': 2, 'round': 'v2'}, {'episode_index': 3, 'round': 'v3'}]}))
    verdicts = tmp_path / 'verdicts_v3.json'
    verdicts.write_text(json.dumps({'episodes': {'3': {'accept': True, 'quality': 5}}}))
    paths = SimpleNamespace(candidates=candidates, review_root=tmp_path, metadata_root=tmp_path/'missing_metadata')
    with pytest.raises(ValueError, match=r"missing=\['1', '2'\]"):
        prepare.finalize(paths, verdicts, 'review test')


def test_nominate_without_prior_is_unchanged(tmp_path):
    paths = review_paths(tmp_path, [summary(1, 'a', 0.5), summary(2, 'a', 0.9), summary(3, 'a', 0.7), summary(4, 'b')])
    value = prepare.nominate(paths, 2)
    # Task a is split into slices [1, 2] and [3]: the most active of each.
    assert [item['episode_index'] for item in value['candidates']] == [2, 3, 4]
    assert 'round' not in value['candidates'][0] and 'round_counts' not in value
    assert not (tmp_path / 'candidates_v2.json').exists()


def test_nominate_excludes_prior_nominees_and_fills_the_shortfall(tmp_path):
    prior_records = [summary(2, 'a', 0.9), summary(5, 'a', 0.9)]
    prior = tmp_path / 'prior' / 'candidates.json'
    prior.parent.mkdir()
    prior.write_text(json.dumps({'per_task': 2, 'candidates': prior_records}))
    pool = [summary(i, 'a', active_fraction=fraction) for i, fraction in zip(range(1, 7), (0.5, 1.0, 0.6, 0.4, 1.0, 0.8))]
    paths = review_paths(tmp_path, pool, prior={'per_task': 2, 'candidates': prior_records})
    value = prepare.nominate(paths, 4, prior)
    new = [item for item in value['candidates'] if item['round'] == 'v3']
    # Remaining recording order [1, 3, 4, 6] split in two slices; most active per slice.
    assert [item['episode_index'] for item in new] == [3, 6]
    assert not {2, 5} & {item['episode_index'] for item in new}
    old = [item for item in value['candidates'] if item['round'] == 'v2']
    assert [{k: v for k, v in item.items() if k != 'round'} for item in old] == prior_records
    assert value['candidates'][:2] == old
    assert value['round_counts'] == {'v2': 2, 'v3': 2}
    assert value['new_candidate_count'] == 2 and value['candidate_count'] == 4
    assert value['prior_candidates_path'] == str(prior)
    assert value['pool_size'] == 4
    assert json.loads((tmp_path / 'candidates_v2.json').read_text()) == {'per_task': 2, 'candidates': prior_records}
    assert json.loads(paths.candidates.read_text())['candidates'] == value['candidates']


def test_nominate_quota_is_per_task_minus_prior_and_never_negative(tmp_path):
    prior_records = [summary(10, 'two_prior'), summary(11, 'two_prior')] + [summary(20 + i, 'full') for i in range(4)] + [summary(30 + i, 'over') for i in range(5)]
    prior = tmp_path / 'prior.json'
    prior.write_text(json.dumps({'candidates': prior_records}))
    pool = prior_records + [summary(12 + i, 'two_prior') for i in range(5)] + [summary(24, 'full'), summary(35, 'over')] + [summary(40 + i, 'fresh') for i in range(3)]
    paths = review_paths(tmp_path, pool)
    value = prepare.nominate(paths, 4, prior)
    new = {}
    for item in value['candidates']:
        if item['round'] == 'v3':
            new.setdefault(item['task'], []).append(item['episode_index'])
    # Remaining pool [12..16] in two slices, ties resolve to the earliest episode.
    assert new == {'two_prior': [12, 15], 'fresh': [40, 41, 42]}
    assert value['round_counts'] == {'v2': 11, 'v3': 5}


def test_backup_is_not_overwritten(tmp_path):
    prior = tmp_path / 'prior.json'
    prior.write_text(json.dumps({'candidates': []}))
    paths = review_paths(tmp_path, [summary(1, 'a')], prior={'candidates': [summary(1, 'a')]})
    (tmp_path / 'candidates_v2.json').write_text('{"kept": true}\n')
    prepare.nominate(paths, 4, prior)
    assert (tmp_path / 'candidates_v2.json').read_text() == '{"kept": true}\n'


def test_batches_are_round_robin_over_sorted_indices():
    batches = prepare.batches_for(list(range(223, 0, -1)))
    assert len(batches) == 12 and [len(batch) for batch in batches] == [19] * 7 + [18] * 5
    assert batches[0][:3] == [1, 13, 25] and batches[1][:2] == [2, 14]
    assert sorted(index for batch in batches for index in batch) == list(range(1, 224))
    assert prepare.batches_for([5, 3]) == [[3, 5]]


def test_proxies_render_one_round_into_the_sheets_dir(tmp_path, monkeypatch):
    metadata = tmp_path / 'metadata'
    (metadata / 'meta').mkdir(parents=True)
    (metadata / 'meta/info.json').write_text(json.dumps({'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}))
    rows = [{'episode_index': index, **{f'videos/{camera}/{field}': 0 for camera in prepare.CAMERAS for field in ('chunk_index', 'file_index', 'from_timestamp')},
             **{f'videos/{camera}/to_timestamp': 5.0 for camera in prepare.CAMERAS}} for index in (1, 2, 3)]
    monkeypatch.setattr(prepare, 'episode_rows', lambda root: rows)
    monkeypatch.setattr(prepare, 'contact_sheet', lambda video, start, offsets, destination, header: destination.write_text(header))
    union = {'candidates': [{**summary(1, 'a'), 'round': 'v2'}, {**summary(2, 'a'), 'round': 'v3'}, {**summary(3, 'b'), 'round': 'v3'}]}
    paths = review_paths(tmp_path, [], prior=union)
    paths.metadata_root = metadata
    paths.source_root = tmp_path / 'staging'
    sheets = tmp_path / 'sheets'
    written = prepare.proxies(paths, 'v3', sheets)
    assert sorted(p.name for p in written) == sorted(f'episode_{i:06d}.{cam}.jpg' for i in (2, 3) for cam in ('primary', 'secondary', 'wrist'))
    assert all(p.parent == sheets for p in written)
    assert json.loads((sheets / 'batches.json').read_text()) == [[2, 3]]
    assert json.loads((sheets / 'candidates.json').read_text()) == union
    assert not list(tmp_path.glob('episode_*.jpg'))
    # Default call: every candidate, into the review dir, nothing else written.
    assert len(prepare.proxies(paths)) == 9 and (tmp_path / 'episode_000001.wrist.jpg').is_file()
    assert not (tmp_path / 'batches.json').exists()
