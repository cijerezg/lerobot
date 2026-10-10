"""Corrected actor views retain integrity checks without using the old v3 counts."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from lerobot.datasets.diverse_actor_selection import DiverseActorSelection, assert_selection_counts


@pytest.fixture
def corrected_selection(tmp_path):
    tables = []
    selected = []
    for store, source in [('corpus', 'droid'), ('fmb', 'fmb')]:
        rows = [
            dict(source=source, episode_id=source + '_train', anchor_s=0.2, retained=True),
            dict(source=source, episode_id=source + '_excluded', anchor_s=0.4, retained=False),
        ]
        if store == 'corpus':
            rows.append(dict(source=source, episode_id='held', anchor_s=0.0, retained=True))
        path = tmp_path / store / 'actor_anchors_5hz.jsonl'
        path.parent.mkdir()
        payload = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        path.write_bytes(payload)
        tables.append(dict(path=path.relative_to(tmp_path).as_posix(), bytes=len(payload),
                           sha256=hashlib.sha256(payload).hexdigest()))
        selected.append(rows[0])
    (tmp_path / 'corrected_store_manifest.json').write_text(json.dumps({'output_semantic_tables': tables}))
    (tmp_path / 'holdout_episodes.json').write_text(json.dumps({'episode_ids': ['held']}))
    corpus = SimpleNamespace(**{
        name: SimpleNamespace(root=tmp_path / store, _actor_view_name='actor_anchors_5hz.jsonl')
        for name, store in [('common', 'corpus'), ('fmb', 'fmb')]
    })
    return DiverseActorSelection(corpus=corpus, rows=selected, episode_records={}, held_out={'held': 1})


def test_corrected_counts_may_differ_from_legacy(corrected_selection):
    assert_selection_counts(corrected_selection)


def test_changed_table_with_same_counts_is_rejected(corrected_selection):
    path = corrected_selection.corpus.common.root / 'actor_anchors_5hz.jsonl'
    path.write_bytes(path.read_bytes().replace(b'0.2', b'0.3'))
    with pytest.raises(ValueError, match='does not match'):
        assert_selection_counts(corrected_selection)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_anchor', 'wrong_source', 'held_out'])
def test_selection_must_match_audited_anchors(corrected_selection, change):
    rows = corrected_selection.rows
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows.append(dict(rows[0]))
    elif change == 'wrong_anchor':
        rows[0]['anchor_s'] = 0.3
    elif change == 'wrong_source':
        rows[0]['source'] = 'yam'
    else:
        rows[0]['episode_id'] = 'held'
    with pytest.raises(ValueError, match='after holdout'):
        assert_selection_counts(corrected_selection)


def test_holdout_accounting_is_checked(corrected_selection):
    corrected_selection.held_out = {}
    with pytest.raises(ValueError, match='holdout accounting'):
        assert_selection_counts(corrected_selection)


def test_manifest_must_cover_both_stores(corrected_selection):
    path = corrected_selection.corpus.common.root.parent / 'corrected_store_manifest.json'
    data = json.loads(path.read_text())
    data['output_semantic_tables'].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='no actor table'):
        assert_selection_counts(corrected_selection)


def test_without_manifest_legacy_guard_still_applies(corrected_selection):
    (corrected_selection.corpus.common.root.parent / 'corrected_store_manifest.json').unlink()
    with pytest.raises(ValueError, match='expected 891 / 87306'):
        assert_selection_counts(corrected_selection)
