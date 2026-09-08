import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from lerobot.probes.future_visual_cases import capture_case, latent_colors, latent_error, render_report, select_cases
from tests.policies.test_molmoact2_future_visual import objective


def annotation_root(tmp_path):
    meta=tmp_path/'meta'
    meta.mkdir()
    (meta/'info.json').write_text(json.dumps({'fps':10}))
    pd.DataFrame([
        {'episode_index':2,'from_index':0,'to_index':200,'subtask':'grasp sock','quality':5},
        {'episode_index':3,'from_index':200,'to_index':300,'subtask':'move sock','quality':5},
        {'episode_index':3,'from_index':300,'to_index':400,'subtask':'grasp sock','quality':1},
    ]).to_parquet(meta/'episode_metadata.parquet')
    pd.DataFrame([
        {'episode_index':3,'from_index':330,'to_index':360,'mistake':True,'mistake_type':'close'},
        {'episode_index':3,'from_index':370,'to_index':400,'mistake':True,'mistake_type':'close'},
    ]).to_parquet(meta/'mistakes.parquet')
    return tmp_path


def test_selection_is_fixed_and_does_not_invent_future_or_recovery(tmp_path):
    root=annotation_root(tmp_path)
    m=select_cases(root,episode=3,stride=2,history_seconds=(2.,1.))
    assert m == select_cases(root,episode=3,stride=2,history_seconds=(2.,1.))
    assert m['omitted'] and any(c['future_index'] is None for c in m['cases'])
    for c in m['cases']:
        assert c['frame_index'] % 2 == 0
        if c['future_index'] is not None:
            assert c['future_index'] < (400 if c['episode']==3 else 200)
        if c['label'].startswith('clean'):
            assert not c['mistake']
    assert any('same subtask' in c['label'] for c in m['cases'])
    assert next(c for c in m['cases'] if c['label']=='close: after span')['mistake']


def test_annotation_provenance_and_configuration_validation(tmp_path):
    root=annotation_root(tmp_path)
    before=select_cases(root,stride=2)['annotation_sha256']
    p=root/'meta/mistakes.parquet'
    m=pd.read_parquet(p)
    m['note']='new annotation'
    m.to_parquet(p)
    assert select_cases(root,stride=2)['annotation_sha256'] != before
    with pytest.raises(ValueError,match='absent'):
        select_cases(root,episode=99,stride=2)
    with pytest.raises(ValueError,match='horizon'):
        select_cases(root,stride=3)


def tiny_policy():
    backbone,aux=objective()
    policy=nn.Module()
    policy.backbone=backbone
    policy.future_visual=aux
    policy._backbone=lambda:backbone
    current=torch.randn(1,2,4,6)
    future=torch.randn_like(current)
    cameras=torch.ones(1,2,dtype=torch.bool)
    aux.prepare(backbone.vision_backbone.image_vit,current,future,torch.ones(1,dtype=torch.bool),cameras)
    aux.optimizer_step(backbone.vision_backbone.image_vit)
    batch={'pixel_values':current.flatten(0,1),'history_images':torch.randn(1,2,3,4,6),
           'history_image_times':torch.tensor([6.,4.,2.]),'history_images_mask':cameras}
    return policy,batch,future.flatten(0,1)


def test_capture_matches_loss_and_preserves_state():
    from lerobot.policies.molmoact2.modeling_molmoact2 import _MEM_TEMPORAL_CAPTURE
    policy,batch,future=tiny_policy()
    state=copy.deepcopy(policy.future_visual.state_dict())
    policy.train()
    saved=dict(_MEM_TEMPORAL_CAPTURE)
    existing_record={'sentinel':True}
    _MEM_TEMPORAL_CAPTURE.update(enabled=False,records=[existing_record])
    try:
        out=capture_case(policy,batch,cameras=2,future_pixels=future)
        assert out['prediction_status']=='ready' and out['prediction'].shape==(2,4,4)
        np.testing.assert_allclose(out['attention'].sum(-1),1,atol=1e-6)
        np.testing.assert_allclose(out['patch_attention'].sum(-1),1,atol=1e-6)
        expected=latent_error(torch.from_numpy(out['prediction']),torch.from_numpy(out['target']))
        np.testing.assert_allclose(out['error'],expected.numpy(),atol=1e-6)
        assert policy.training and _MEM_TEMPORAL_CAPTURE['records']==[existing_record]
        assert not _MEM_TEMPORAL_CAPTURE['enabled']
        for key,value in policy.future_visual.state_dict().items():
            torch.testing.assert_close(value,state[key])
        assert all(p.grad is None for p in policy.parameters())
    finally:
        _MEM_TEMPORAL_CAPTURE.clear()
        _MEM_TEMPORAL_CAPTURE.update(saved)


def test_missing_target_history_and_head_do_not_produce_fake_values():
    policy,batch,_=tiny_policy()
    batch={k:v for k,v in batch.items() if not k.startswith('history')}
    out=capture_case(policy,batch,cameras=2)
    assert 'prediction' in out and 'target' not in out and 'error' not in out
    assert out['layers']==[] and 'attention' not in out
    policy.future_visual=None
    out=capture_case(policy,batch,cameras=2)
    assert out['prediction_status']=='auxiliary absent in checkpoint' and 'prediction' not in out


def test_fresh_head_is_not_presented_as_trained_prediction():
    policy,batch,future=tiny_policy()
    policy.future_visual.optimizer_steps.zero_()
    out=capture_case(policy,batch,cameras=2,future_pixels=future)
    assert 'untrained' in out['prediction_status'] and 'prediction' not in out


def test_latent_colors_are_fixed_and_error_uses_full_vector():
    x=np.random.default_rng(3).normal(size=(20,128)).astype(np.float32)
    np.testing.assert_array_equal(latent_colors(x[:5]),latent_colors(x)[:5])
    assert latent_error(torch.tensor(x),torch.tensor(x)).max()==0


def test_report_embeds_data_and_escapes_annotation_text(tmp_path):
    render_report(tmp_path,{'cases':[],'note':'</script><script>bad()</script>'},[],mode='dataset preview')
    html=(tmp_path/'cases.html').read_text()
    assert '__CASE_DATA__' not in html and '</script><script>bad()' not in html
    assert '\\u003c/script>' in html
    assert json.loads((tmp_path/'index.json').read_text())['panels'][0]['file']=='cases.html'


def test_probe_registration():
    from lerobot.configs.train import ProbeConfig
    from lerobot.scripts.rl_offline import _validate_probe_registry
    _validate_probe_registry(ProbeConfig(enable_future_visual_cases=True))


def test_complete_model_report_writes_raw_arrays(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import lerobot.probes.future_visual_cases as module
    (tmp_path/'dataset').mkdir()
    root=annotation_root(tmp_path/'dataset')
    policy,batch,_=tiny_policy()
    keys=['observation.images.top','observation.images.wrist']
    obs={key:torch.zeros(1,3,8,8,dtype=torch.uint8) for key in keys}
    monkeypatch.setattr(module,'probe_frame_inputs',lambda *args,**kwargs:{
        'obs':obs,'task':'move','subtask':'grasp','metadata':{'mistake':False},
    })
    cfg=SimpleNamespace(
        policy=SimpleNamespace(memory=SimpleNamespace(history_times_seconds=lambda:[2.,1.]),
                               future_visual_loss=SimpleNamespace(horizon_seconds=4.),image_stride=2,
                               image_keys=keys,pretrained_path='synthetic-test'),
        probe_parameters=SimpleNamespace(future_visual_case_episode=3),
    )
    adapter=SimpleNamespace(policy=policy,_preprocessor=None,_configured_image_keys=lambda:keys,
                            _make_batch=lambda *args,**kwargs:batch)
    output=tmp_path/'report'
    module.run(adapter,SimpleNamespace(root=root),cfg,output)
    manifest=json.loads((output/'cases.json').read_text())
    for case in manifest['cases']:
        with np.load(output/f"{case['id']}.npz") as arrays:
            assert 'prediction' in arrays and 'attention' in arrays
            assert ('error' in arrays)==(case['future_index'] is not None)
