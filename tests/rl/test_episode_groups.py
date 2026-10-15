"""Whole-episode task shares without duplicating or slicing cached datasets."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from lerobot.configs.default import DatasetConfig, OfflineDatasetSourceConfig
from lerobot.configs.diverse import DiverseCollectionConfig
from lerobot.rl.data_sources.diverse_integration import build_mixture_groups
from lerobot.rl.data_sources.diverse_mixture import make_hierarchical_offline_iterator
from lerobot.rl.data_sources.episode_groups import episode_group_views
from lerobot.rl.offline_dataset_utils import get_offline_dataset_sources


class FakeBuffer:
    def __init__(self, root, lengths, groups=None, group='external_rebot'):
        self.size = sum(lengths)
        self.capacity = self.size
        self.optimize_memory = True
        self.image_stride = 3
        self.storage_device = torch.device('cpu')
        self.actions = torch.zeros(1, 2).expand(self.size, 2)
        self.dones = torch.zeros(self.size, dtype=torch.bool)
        self.offline_source = SimpleNamespace(root=str(root), name=root.name, episodes=None,
                                              sampling_group=group, episode_groups=groups or {})
        offset = 0
        rows = []
        for ep, length in enumerate(lengths):
            rows.append(dict(episode_index=ep, dataset_from_index=offset, dataset_to_index=offset+length))
            self.dones[offset+length-1] = True
            offset += length
        path = root/'meta/episodes/chunk-000'
        path.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(path/'file-000.parquet')
        self.code = torch.zeros(self.size, dtype=torch.long)

    def sample(self, batch_size, action_chunk_size=30, *, indices=None):
        if indices is None:
            indices = torch.zeros(batch_size, dtype=torch.long)
        assert len(indices) == batch_size
        assert (indices % self.image_stride == 0).all()
        checks = indices[:, None] + torch.arange(action_chunk_size-1)
        assert not self.dones[checks].any()
        assert (indices + action_chunk_size <= self.size).all()
        state = {'observation.state': self.code[indices, None].float()}
        return dict(state=state, next_state=state, action=torch.zeros(batch_size, action_chunk_size, 2),
                    reward=torch.zeros(batch_size), done=torch.zeros(batch_size), truncated=torch.zeros(batch_size),
                    complementary_info={'sample_index': indices, 'group_code': self.code[indices]})


def test_partition_selects_only_owned_episodes_and_valid_stride_aligned_chunks(tmp_path):
    b = FakeBuffer(tmp_path/'data', [62, 45, 61], {'a':[0,2], 'b':[1]})
    groups = list(episode_group_views(b, b.offline_source.episode_groups, 30))
    assert [g[2] for g in groups] == [22, 5]
    assert groups[1][1].sampling_indices.tolist() == [63, 66, 69, 72, 75]
    for group, view, count in groups:
        batch = view.sample(2000)
        starts = batch['complementary_info']['sample_index']
        if group == 'a':
            assert ((starts < 62) | (starts >= 107)).all()
        else:
            assert ((starts >= 62) & (starts < 107)).all()
        assert view.buffer is b
        assert len(view) == count
        generated = next(view.get_iterator(batch_size=8, action_chunk_size=30))
        assert torch.isin(generated["complementary_info"]["sample_index"], view.sampling_indices).all()
        with pytest.raises(ValueError, match='chunk_size changes'):
            view.sample(1, action_chunk_size=3)
    with pytest.raises(ValueError, match='outside'):
        groups[0][1].sample(1, indices=torch.tensor([63]))


@pytest.mark.parametrize('partition', [{'a':[0]}, {'a':[0,1], 'b':[1]}, {'a':[0,2]}])
def test_invalid_partition_fails_closed(tmp_path, partition):
    b = FakeBuffer(tmp_path/'data', [60,60])
    with pytest.raises(ValueError, match='every episode exactly once'):
        list(episode_group_views(b, partition, 30))


def test_too_short_episode_is_not_silently_dropped(tmp_path):
    b = FakeBuffer(tmp_path/'data', [15,60])
    with pytest.raises(ValueError, match='no eligible'):
        list(episode_group_views(b, {'a':[0,1]}, 30))


def test_config_partition_survives_source_resolution():
    cfg = SimpleNamespace(dataset=DatasetConfig(repo_id='local/test', sources=[
        OfflineDatasetSourceConfig(root='/data/root', episode_groups={'a':[0], 'b':[1]})]))
    assert get_offline_dataset_sources(cfg)[0].episode_groups == {'a':[0], 'b':[1]}
    with pytest.raises(ValueError, match='distinct'):
        OfflineDatasetSourceConfig(root='/data/root', episode_groups={'a':[0], 'b':[0]})
    with pytest.raises(ValueError, match='complete'):
        OfflineDatasetSourceConfig(root='/data/root', episodes=[0], episode_groups={'a':[0]})


def test_real_iterator_matches_shares_and_weights_roots_by_eligible_starts(tmp_path):
    a = FakeBuffer(tmp_path/'a', [60, 90], {'clothing':[0], 'rigid':[1]})
    b = FakeBuffer(tmp_path/'b', [120], {'rigid':[0]})
    external = FakeBuffer(tmp_path/'ext', [90])
    diverse = FakeBuffer(tmp_path/'div', [90])
    a.code[:60] = 0
    a.code[60:] = 1
    b.code[:] = 2
    external.code[:] = 3
    diverse.code[:] = 4
    cfg = DiverseCollectionConfig(weight=.30, rebot_weight=.70,
        rebot_group_weights={'clothing':.0455, 'rigid':.3045, 'external_rebot':.35})
    groups = build_mixture_groups(cfg, [a,b,external], diverse, action_chunk_size=30)
    assert groups[1].inner_weights == [21,31]
    it = make_hierarchical_offline_iterator(groups, batch_size=64, proportional=True, seed=23, action_chunk_size=30)
    observed = Counter()
    for _ in range(1000):
        observed.update(next(it)['complementary_info']['group_code'].tolist())
    expected = {0:.0455, 1:.3045*21/52, 2:.3045*31/52, 3:.35, 4:.30}
    for key, probability in expected.items():
        assert observed[key]/64000 == pytest.approx(probability, abs=.008)


def test_zero_weight_group_is_excluded_from_sampling(tmp_path):
    external = FakeBuffer(tmp_path/'ext', [90, 60], {'external_rebot':[0], 'excluded':[1]})
    diverse = FakeBuffer(tmp_path/'div', [90])
    cfg = DiverseCollectionConfig(weight=.30, rebot_weight=.70, rebot_group_weights={'external_rebot':.70, 'excluded':0})
    groups = build_mixture_groups(cfg, [external], diverse, action_chunk_size=30)
    assert [g.name for g in groups] == ['external_rebot', 'diverse']
    assert int(groups[0].buffers[0].sampling_indices.max()) < 90
    with pytest.raises(ValueError):
        DiverseCollectionConfig(rebot_group_weights={'a': -1}).validate()


def test_checked_in_training_and_validation_splits_cover_every_internal_episode():
    import yaml
    from lerobot.rl.data_sources.episode_groups import read_episode_ranges

    workspace = Path(__file__).resolve().parents[3]
    configs = [yaml.safe_load((workspace/name).read_text()) for name in ('config_rl.yaml','config_rl_validate.yaml')]
    assert configs[0]['dataset']['sources'] == configs[1]['dataset']['sources']
    expected = dict(clothing=.13, bits_clippers=.18, rigid=.245, bottles_lids=.154,
                    pills_caps=.132, cases_kit_box_push=.159)
    cfg = configs[0]
    assert cfg['val_freq'] == 0 and not cfg['val_on_start']
    assert cfg['diverse']['weight'] == .25 and cfg['diverse']['rebot_start_cell'] == 5  # 2026-10-14 mix 50/25/25
    shares = cfg['diverse']['rebot_group_weights']
    assert shares['external_rebot'] == .25
    assert sum(shares.values()) == pytest.approx(.75)
    totals = Counter()
    for source in cfg['dataset']['sources']:
        if not source.get('episode_groups'):
            continue
        ranges = read_episode_ranges(str(workspace/source['root']))
        assigned = [ep for eps in source['episode_groups'].values() for ep in eps]
        assert sorted(assigned) == sorted(ranges)
        for group, eps in source['episode_groups'].items():
            totals[group] += len(eps)
    assert sum(totals.values()) == 189 + 241  # internal 174 (10-14 task review) + 2 oldval_train + 13 oct14
    assert totals['external_rebot'] == 231 and totals['excluded'] == 13 and shares['excluded'] == 0  # 10 shake-hand + 3 clipper (oct07 3,4; oct08 9)
    for group, share in expected.items():
        assert totals[group] > 0
        assert shares[group] == pytest.approx(.50*share)
