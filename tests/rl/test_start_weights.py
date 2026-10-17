"""Per-start draw weights around commit moments (2026-10-16)."""
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.configs.diverse import DiverseCollectionConfig, StartWeightsConfig
from lerobot.rl.data_sources.diverse_integration import build_mixture_groups
from lerobot.rl.data_sources.diverse_mixture import make_hierarchical_offline_iterator
from lerobot.rl.data_sources.episode_groups import episode_group_views
from lerobot.rl.data_sources.start_weights import frame_weights
from tests.rl.test_episode_groups import FakeBuffer


def write_meta(root, segments, events, fps=30):
    """segments: (episode, from, to, subtask, precision); events: (type, global index)."""
    meta = root / 'meta'
    json.dump({'fps': fps}, open(meta / 'info.json', 'w'))
    rows = [dict(episode_index=e, segment_index=i, from_index=a, to_index=b, subtask=s, precision=p)
            for i, (e, a, b, s, p) in enumerate(segments)]
    pd.DataFrame(rows).to_parquet(meta / 'episode_metadata.parquet')
    pd.DataFrame(rows).to_parquet(meta / 'precision.parquet')
    pd.DataFrame([dict(event_type=t, index=i, episode_index=0, frame_index=i) for t, i in events],
                 columns=['event_type', 'index', 'episode_index', 'frame_index']).to_parquet(meta / 'depth_gripper_events.parquet')


CFG = StartWeightsConfig(enabled=True, peak_ratio=5, precision_scale=(0.3, 0.4, 0.6, 0.8, 1.0), skill_ratio=3,
                         grasp_window=(2, 1), release_window=(1, 0.5), smooth=False)


def test_frame_weights_follow_the_rule(tmp_path):
    b = FakeBuffer(tmp_path / 'r', [300, 150])
    # ep0: grasp p3 (close at 120) | move | release p5 (open at 270); ep1: push p2 whole segment
    write_meta(tmp_path / 'r', [(0, 0, 180, 'grasp the cup', 3), (0, 180, 240, 'move the cup to the bin', 4),
                               (0, 240, 300, 'release the cup in the bin', 5), (1, 300, 450, 'push the book aside', 2)],
               [('close', 120), ('open', 270)])
    w = frame_weights(tmp_path / 'r', b.size, CFG)
    assert w[0] == 1 and w[59] == 1                       # transit before the 2 s window = base
    assert w[60] == 3 and w[120] == 3 and w[150] == 3     # p3 peak = 5 * .6, window [-2 s, +1 s]
    assert w[151] == 1 and w[200] == 1                    # after the decay, and the move
    assert w[240] == 5 and w[255] == 5 and w[285] == 5    # release p5 window [-1 s, +.5 s]
    assert w[239] == 1 and w[286] == 1
    assert (w[300:450] == 3).all()                        # skill, hard edges
    smooth = frame_weights(tmp_path / 'r', b.size, StartWeightsConfig(enabled=True, smooth=True))
    assert smooth[120] == pytest.approx(3) and 1 < smooth[60] < 3 and smooth[0] < smooth[30] < smooth[90]


def test_views_draw_by_weight_and_groups_keep_config_shares(tmp_path):
    a = FakeBuffer(tmp_path / 'a', [300, 150], {'rigid': [0], 'cases': [1]})
    write_meta(tmp_path / 'a', [(0, 0, 180, 'grasp the cup', 3), (0, 180, 300, 'move the cup to the bin', 4),
                               (1, 300, 450, 'push the book aside', 2)], [('close', 120)])
    ext = FakeBuffer(tmp_path / 'ext', [150], {'external_rebot': [0]})
    write_meta(tmp_path / 'ext', [(0, 0, 150, 'move the cup to the bin', 1)], [])
    diverse = FakeBuffer(tmp_path / 'div', [90])
    a.code[:300] = 0; a.code[300:] = 1; ext.code[:] = 2; diverse.code[:] = 3
    cfg = DiverseCollectionConfig(weight=.25, rebot_weight=.75, rebot_start_weights=CFG,
                                  rebot_group_weights={'rigid': .4, 'cases': .1, 'external_rebot': .25})
    cfg.validate()
    views = list(episode_group_views(a, a.offline_source.episode_groups, 30, 1, CFG))
    rigid = views[0][1]
    assert rigid.weights.tolist() == [3.0 if 60 <= i <= 150 else 1.0 for i in rigid.sampling_indices.tolist()]
    mass = {name: m for name, _, m in views}
    assert mass['rigid'] == pytest.approx(float(rigid.weights.sum()))
    groups = build_mixture_groups(cfg, [a, ext], diverse, action_chunk_size=30)
    assert [g.weight for g in groups] == [.4, .1, .25, .25]                      # config shares, not mass
    assert groups[0].inner_weights == [pytest.approx(mass['rigid'])]               # mass splits the views inside a group
    it = make_hierarchical_offline_iterator(groups, batch_size=64, proportional=True, seed=5, action_chunk_size=30)
    observed = Counter(); peak = 0
    for _ in range(600):
        batch = next(it)
        observed.update(batch['complementary_info']['group_code'].tolist())
        idx = batch['complementary_info']['sample_index']; code = batch['complementary_info']['group_code']
        peak += int(((code == 0) & (idx >= 60) & (idx <= 150)).sum())
    n = 64 * 600
    assert observed[0] / n == pytest.approx(.4, abs=.01)
    assert observed[1] / n == pytest.approx(.1, abs=.01)
    # inside rigid, the 31 peak starts carry weight 3 each against 1 for the other 60 (91 starts: 0..270 step 3)
    assert peak / observed[0] == pytest.approx(93 / 153, abs=.02)


def test_cell_and_weights_are_exclusive():
    with pytest.raises(ValueError, match='exclusive'):
        DiverseCollectionConfig(rebot_start_cell=5, rebot_start_weights=StartWeightsConfig(enabled=True)).validate()
