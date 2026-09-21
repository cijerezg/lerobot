"""Cache-ready external ReBot regressions: roles, calibration, and bucket ratios."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.rl.data_sources.diverse_mixture import MixtureGroup, draw_proportional_quotas
from lerobot.rl.data_sources.rebot_role_adapter import RoleAlignedBuffer, rebot_role_renames, probe_rebot_cache
from lerobot.rl.offline_dataset_utils import buffer_state_keys


def test_canonical_keys_do_not_change_existing_legacy_renames():
    assert rebot_role_renames(['observation.images.top', 'observation.images.wrist'], ['wrist.depth']) == {
        'observation.images.top': 'observation.images.external_0',
        'observation.images.wrist': 'observation.images.wrist_0',
        'depth.wrist.depth': 'depth.wrist_0.depth',
    }
    assert rebot_role_renames(['observation.images.external_1'], ['wrist_0.depth']) == {
        'observation.images.external_1': 'observation.images.external_1',
        'depth.wrist_0.depth': 'depth.wrist_0.depth',
    }


def contract(root, depth=True):
    root.mkdir(exist_ok=True)
    (root/'meta').mkdir(exist_ok=True)
    d={'schema_version':1,'fps':30,'camera_roles':['external_0','wrist_0'],
       'image_valid_boxes':{'external_0':[1,0,2,4],'wrist_0':[0,0,4,4]},
       'depth_key':'wrist_0.depth' if depth else None,'depth_units_mm_per_level':.1 if depth else None,
       'depth_intrinsics':{'fx':390.,'fy':391.,'cx':315.,'cy':246.} if depth else None}
    (root/'meta/cache_ready.json').write_text(json.dumps(d))
    return d


def test_source_calibration_letterbox_and_missing_view_survive_adapter(tmp_path):
    contract(tmp_path)
    states={k:torch.zeros(2,3,4,4,dtype=torch.uint8) for k in ('observation.images.external_0','observation.images.wrist_0')}
    buffer=SimpleNamespace(dataset=SimpleNamespace(root=tmp_path),states=states)
    adapter=RoleAlignedBuffer(buffer,action_layout_id=6,image_size=(4,4),depth_intrinsics=(1,1,1,1),align=False)
    batch={'action':torch.zeros(2,30,7),'reward':torch.zeros(2),'state':dict(states),'next_state':dict(states),'complementary_info':{'depth.wrist_0.depth':torch.ones(2,4,4,dtype=torch.uint16)}}
    result=adapter.decorate(batch)['complementary_info']
    assert result['camera_is_present'].tolist()==[[True,False,True]]*2
    assert result['image_valid_box'][:,0].tolist()==[[1,0,2,4]]*2
    assert result['depth.wrist_0.intrinsics'].tolist()==[[390.,391.,315.,246.]]*2


def test_rgb_only_prepared_cache_requires_only_its_recorded_views(tmp_path):
    root=tmp_path/'root';contract(root,depth=False)
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'metadata.json').write_text(json.dumps({'dataset_root':str(root),'num_transitions':30,'image_stride':3,'image_keys':['observation.images.external_0','observation.images.wrist_0'],'depth_keys':[], 'fingerprint':'example','image_storage_dtype':'uint8','image_storage_size':None}))
    report=probe_rebot_cache(cache,history_offsets_frames=[],depth_role='wrist',measure_reach=False)
    assert report.usable,report.problems
    cfg=SimpleNamespace(policy=SimpleNamespace(input_features={'observation.images.external_0':None,'observation.images.external_1':None,'observation.images.wrist_0':None,'observation.state':None}))
    dataset=SimpleNamespace(root=root,meta=SimpleNamespace(features={'observation.state':{},'observation.images.external_0':{},'observation.images.wrist_0':{},'action':{}}))
    assert set(buffer_state_keys(cfg,dataset))=={'observation.state','observation.images.external_0','observation.images.wrist_0'}


def test_three_buckets_keep_402040_with_more_sources_than_slots():
    groups=[MixtureGroup('main',[object()]*7,weight=.4,inner_weights=[10,1,1,1,1,1,1]),MixtureGroup('external',[object()]*34,weight=.2),MixtureGroup('diverse',[object()],weight=.4)]
    rng=np.random.default_rng(12);total=np.zeros(3);tiny=0
    for _ in range(2000):
        q=draw_proportional_quotas(32,groups,rng)
        assert sum(map(sum,q))==32
        assert len(q[1])==34
        total+=list(map(sum,q));tiny+=q[0][1]
    assert np.allclose(total/total.sum(),[.4,.2,.4],atol=.008)
    # No artificial floor of one per source per batch.
    assert tiny<2000


def test_consolidated_source_routes_calibration_and_missing_views_per_episode(tmp_path):
    c=contract(tmp_path)
    c['episode_contracts']={'0':{**c,'camera_roles':['external_0']},'1':{**c,'depth_intrinsics':{'fx':101.,'fy':102.,'cx':103.,'cy':104.}}}
    c['episode_contracts']['0']['depth_key']=None
    c['episode_ranges']=[[0,1],[1,2]]
    (tmp_path/'meta/cache_ready.json').write_text(json.dumps(c))
    states={k:torch.zeros(2,3,4,4,dtype=torch.uint8) for k in ('observation.images.external_0','observation.images.wrist_0')}
    buffer=SimpleNamespace(dataset=SimpleNamespace(root=tmp_path),states=states,size=2,storage_device='cpu',complementary_info={},complementary_info_keys=[])
    adapter=RoleAlignedBuffer(buffer,action_layout_id=6,image_size=(4,4),align=False)
    assert buffer.complementary_info['prepared_episode_index'].tolist()==[0,1]
    batch={'action':torch.zeros(2,30,7),'reward':torch.zeros(2),'state':dict(states),'next_state':dict(states),'complementary_info':{'prepared_episode_index':torch.tensor([1,0]),'depth.wrist_0.depth':torch.zeros(2,4,4,dtype=torch.uint16)}}
    info=adapter.decorate(batch)['complementary_info']
    assert info['camera_is_present'].tolist()==[[True,False,True],[True,False,False]]
    assert info['depth.wrist_0.depth_is_present'].tolist()==[True,False]
    assert info['depth.wrist_0.intrinsics'][0].tolist()==[101.,102.,103.,104.]
    assert 'prepared_episode_index' not in info
