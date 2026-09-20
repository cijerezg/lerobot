"""Holdout changes must update cache identity without relabelling frame banks."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lerobot.rl.data_sources import diverse_actor_cache as cache
from lerobot.rl.data_sources.diverse_actor_buffer import DiverseSampleSpec


@pytest.fixture
def fixture(tmp_path):
    root=tmp_path/'corpus'
    for directory in ('corpus','fmb'):
        (root/directory).mkdir(parents=True)
        for name in ('episodes.jsonl','actor_anchors_5hz.jsonl'):
            (root/directory/name).write_text('{}\n')
    (root/'holdout_episodes.json').write_text('{"episode_ids":["b"]}')
    spec=DiverseSampleSpec(image_size=(2,2),depth_size=(2,2))
    source=tmp_path/'source';(source/'anchors').mkdir(parents=True);(source/'banks').mkdir()
    rows=[['a',0,0],['a',1,0],['b',0,1],['c',0,2]]
    for column in cache._anchor_columns(spec):
        array=np.zeros((4,*column.shape),dtype=column.dtype)
        for i in range(4):array[i]=i+1
        if column.name=='identity':array[:]=[[0,0,0,0,0],[0,0,1,0,0],[1,1,0,1,1],[2,2,0,2,2]]
        array.tofile(source/'anchors'/f'{column.name}.bin')
    (source/'banks'/'rgb_external_0.bin').write_bytes(b'unchanged video bytes')
    meta=dict(schema_version=cache.CACHE_SCHEMA_VERSION,partial=False,built_rows=4,
              corpus=cache.corpus_fingerprint(root),spec=spec.fingerprint(),selection=dict(anchors=4,episodes=3))
    (source/'metadata.json').write_text(json.dumps(meta))
    selection=SimpleNamespace(rows=[dict(episode_id=e,anchor_index=a,action_layout_id=l) for e,a,l in (rows[3],rows[1])],
                              episode_ids=['c','a'],held_out={'b':1})
    return root,spec,source,rows,selection


def test_subset_preserves_all_payloads_and_remaps_only_episode_position(fixture,tmp_path):
    root,spec,source,rows,selection=fixture
    target=cache.subset_cache(source,root,tmp_path/'cache',selection,rows,spec)
    assert cache.find_cache(root,target.parent,spec,anchors=2,episodes=2)==target
    for column in cache._anchor_columns(spec):
        old=np.fromfile(source/'anchors'/f'{column.name}.bin',dtype=column.dtype).reshape(4,*column.shape)
        expected=old[[3,1]].copy()
        if column.name=='identity':expected[:,1]=[0,1]
        actual=np.fromfile(target/'anchors'/f'{column.name}.bin',dtype=column.dtype).reshape(2,*column.shape)
        np.testing.assert_array_equal(actual,expected)
    oldbank=source/'banks'/'rgb_external_0.bin';newbank=target/'banks'/oldbank.name
    assert oldbank.stat().st_ino==newbank.stat().st_ino
    assert json.loads((source/'metadata.json').read_text())['selection']['anchors']==4


def test_holdout_membership_changes_fingerprint_even_at_same_counts(fixture):
    root,spec,*_=fixture
    first=cache.cache_fingerprint(root,spec,anchors=2,episodes=2)
    (root/'holdout_episodes.json').write_text('{"episode_ids":["a"]}')
    assert cache.cache_fingerprint(root,spec,anchors=2,episodes=2)!=first


def test_subset_rejects_wrong_manifest_or_new_anchors(fixture,tmp_path):
    root,spec,source,rows,selection=fixture
    wrong=[r.copy() for r in rows];wrong[0][1]=99
    with pytest.raises(ValueError,match='identities'):
        cache.subset_cache(source,root,tmp_path/'cache',selection,wrong,spec)
    selection.rows[0]['episode_id']='new'
    with pytest.raises(ValueError,match='not a subset'):
        cache.subset_cache(source,root,tmp_path/'cache',selection,rows,spec)
