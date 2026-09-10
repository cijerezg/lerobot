"""Atomic speed labels must retain child identity and reject stale parent labels."""
import json
from argparse import Namespace

import numpy as np
import pytest

from lerobot.data_processing.annotate import speed_annotate as speed


@pytest.fixture
def diverse(tmp_path):
    common = dict(episode_id='shared-name', source='droid', embodiment='Franka', split='train',
                  native_rate_hz=10, parent_interval_index=0, quality=4, confidence='confident', primitive=None)
    rows = [dict(common, atom_index=0, start_timestep=0, end_timestep_exclusive=20,
                 verb='move', subtask='move the mug to the table'),
            dict(common, atom_index=1, start_timestep=20, end_timestep_exclusive=30,
                 verb='release', subtask='release the mug on the table')]
    for sub in ('corpus', 'fmb'):
        root = tmp_path / sub
        ep = root / 'episodes' / 'shared-name'
        ep.mkdir(parents=True)
        q = np.zeros((30, 7))
        q[:, 0] = np.arange(30) / 10
        q[:, -1] = np.arange(30) * 100  # excluded gripper in corpus
        np.save(ep / ('state.npy' if sub == 'corpus' else 'q.npy'), q)
        data = rows if sub == 'corpus' else [dict(rows[0], source='fmb', primitive='move_up',
                                                   subtask='lift the object')]
        (root / 'subtask_atoms.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reversed(data)))
    return tmp_path


def test_atomic_children_keep_identity_timing_and_order(diverse):
    rows = speed.diverse_segments(diverse)
    corpus = [r for r in rows if r['source'] == 'droid']
    assert [r['atom_index'] for r in corpus] == [0, 1]
    assert [r['duration_s'] for r in corpus] == [2, 1]
    assert corpus[0]['net_displacement'] == pytest.approx(1.9)
    assert corpus[1]['class'] == 'release'
    assert len({(r['root'], *speed._diverse_key(r)) for r in rows}) == 3
    assert next(r for r in rows if r['source'] == 'fmb')['class'] == 'move_up'


def test_missing_atoms_never_falls_back_to_parent_intervals(diverse):
    (diverse/'corpus/subtask_atoms.jsonl').unlink()
    (diverse/'corpus/critic_intervals.jsonl').write_text('{}\n')
    with pytest.raises(FileNotFoundError, match='finish and validate'):
        speed.diverse_segments(diverse)


def test_duplicate_child_is_rejected(diverse):
    p=diverse/'corpus/subtask_atoms.jsonl'
    p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='Duplicate atom'):
        speed.diverse_segments(diverse)


def reference():
    return {'diverse_layer':'atoms','edges':list(speed.EDGES),'min_class_segments':4,
            'cells':{'droid/*':{'a':2.0,'b':0.0,'n':4},'fmb/*':{'a':2.0,'b':0.0,'n':4}}}


def test_release_inherits_carry_and_atomic_labels_use_separate_file(diverse,tmp_path):
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(reference()))
    parent=diverse/'corpus/speed.jsonl';parent.write_text('historical parent labels\n')
    speed.cmd_annotate_diverse(Namespace(reference=str(ref),diverse=str(diverse),diverse_layer='atoms',force=False))
    rows=[json.loads(l) for l in (diverse/'corpus/speed_atoms.jsonl').read_text().splitlines()]
    assert [r['speed'] for r in rows]==[3,3]
    assert rows[1]['ratio']==2 and rows[1]['speed_source']=='inherited'
    assert 'interval_index' not in rows[0]
    assert parent.read_text()=='historical parent labels\n'


def test_review_rejects_stale_child_boundaries_before_rendering(diverse,tmp_path):
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(reference()))
    speed.cmd_annotate_diverse(Namespace(reference=str(ref),diverse=str(diverse),diverse_layer='atoms',force=False))
    p=diverse/'corpus/subtask_atoms.jsonl'
    rows=[json.loads(l) for l in p.read_text().splitlines()]
    rows[0]['start_timestep']=21
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Stale speed annotation'):
        speed.cmd_review(Namespace(root=[],diverse=str(diverse),diverse_layer='atoms',out=str(tmp_path/'review'),
                                  seed=1,per_bucket=1,diverse_per_bucket=1))


def test_parent_reference_cannot_label_atoms(diverse,tmp_path):
    ref=tmp_path/'reference.json';data=reference();data.pop('diverse_layer');ref.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='layer differ'):
        speed.cmd_annotate_diverse(Namespace(reference=str(ref),diverse=str(diverse),diverse_layer='atoms',force=False))


def conservative_reference():
    ref = reference()
    ref['unclear_policy'] = 'atoms_default_3_v1'
    ref['cells']['droid/move'] = {'a': 4.0, 'b': 0.0, 'n': 10}
    return ref


def test_unclear_carry_defaults_and_release_inherits_default(diverse):
    rows = [r for r in speed.diverse_segments(diverse) if r['source'] == 'droid']
    rows[0]['confidence'] = 'unsure'
    result = speed.label(rows, conservative_reference())
    assert [r['raw_speed'] for r in result] == [5, 5]
    assert [r['speed'] for r in result] == [3, 3]
    assert all(r['speed_source'] == 'default_unclear' for r in result)
    assert result[1]['speed_default_reason'] == 'release inherits an uncertain predecessor'


def test_release_does_not_inherit_across_excluded_gap(diverse):
    rows = [r for r in speed.diverse_segments(diverse) if r['source'] == 'droid']
    rows[1]['start_timestep'] += 1
    result = speed.label(rows, conservative_reference())
    assert [r['speed'] for r in result] == [5, 3]
    assert result[1]['raw_speed'] == 5
    assert 'no contiguous' in result[1]['speed_default_reason']


def test_supported_carry_release_unchanged_and_rare_action_defaults(diverse):
    rows = [r for r in speed.diverse_segments(diverse) if r['source'] == 'droid']
    assert [r['speed'] for r in speed.label(rows, conservative_reference())] == [5, 5]
    rows[0]['class'] = 'stir'
    result = speed.label(rows, conservative_reference())
    assert result[0]['speed'] == 3
    assert 'too few training' in result[0]['speed_default_reason']
    assert result[1]['speed'] == 3


def test_parent_continuations_flag_both_sides_without_changing_speed(diverse):
    rows = [r for r in speed.diverse_segments(diverse) if r['source'] == 'droid']
    rows[1].update(parent_interval_index=1, atom_index=0, subtask=rows[0]['subtask'], **{'class':'move'})
    result = speed.label(rows, conservative_reference())
    assert [r['speed'] for r in result] == [5, 5]
    assert all('same action continues across a parent boundary' in r['speed_flags'] for r in result)
