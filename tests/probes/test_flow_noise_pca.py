from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.probes import flow_noise_pca as probe


def test_shared_pca_preserves_relative_group_positions_and_scale():
    a=np.array([[-1.,0,0],[1,0,0],[0,-1,0],[0,1,0]])
    b=10*a+np.array([20,4,0])
    z=np.concatenate([a,b])
    pca=probe.shared_pca(z)
    np.testing.assert_allclose(pca['mean'],z.mean(0))
    np.testing.assert_allclose(pca['scores']@pca['components']+pca['mean'],z,atol=1e-12)
    # Group centering/standardization would erase these differences.
    scores=pca['scores']
    np.testing.assert_allclose(np.linalg.norm(scores[:4].mean(0)-scores[4:].mean(0)),np.linalg.norm(a.mean(0)-b.mean(0)))
    assert np.linalg.norm(scores[4]-scores[5])/np.linalg.norm(scores[0]-scores[1])==pytest.approx(10)
    assert pca['explained_variance_ratio'].sum()==pytest.approx(1)


def test_balanced_sampling_returns_exact_count_and_interleaves_episodes(monkeypatch):
    from lerobot.probes import utils
    candidates=[(ep,i*3,ep*10000+i*3) for ep in range(7) for i in range(200)]
    monkeypatch.setattr(utils,'sample_episodes_evenly',lambda *args:candidates)
    samples=probe.balanced_contexts(None,150,42,3)
    assert len(samples)==len(set(samples))==150
    counts=[sum(s[0]==ep for s in samples) for ep in range(7)]
    assert max(counts)-min(counts)==1
    assert len({ep for ep,_,_ in samples[:7]})==7
    assert samples==probe.balanced_contexts(None,150,42,3)
    assert all(fr%3==0 for _,fr,_ in samples)


def test_capture_pairs_five_targets_and_checks_each_with_production(tmp_path,monkeypatch):
    import json
    from lerobot.probes import utils

    monkeypatch.setattr(probe,'normalization_contract',lambda p:dict(action_layout_id=6))
    monkeypatch.setattr(probe,'balanced_contexts',lambda *args:[(2,3,42)])
    monkeypatch.setattr(utils,'probe_image_stride',lambda cfg:3)
    monkeypatch.setattr(utils,'probe_frame_inputs',lambda *args,**kwargs:{})
    monkeypatch.setattr(probe,'render',lambda out:None)
    calls=[]
    @contextmanager
    def prepare(adapter,frame):
        valid=torch.tensor([True]*7+[False])
        target=torch.zeros(1,30,8)
        v=torch.zeros_like(target);v[...,valid]=2
        def velocity(x,t):return v.expand_as(x)
        velocity.action_layout_id=6
        def deployed(noise,n):
            assert noise.shape[0]==1  # actual sampler checks are separate
            calls.append(noise.clone())
            return noise+v
        yield target,valid,velocity,torch.float32,deployed
    own=SimpleNamespace(inv_out=str(tmp_path),inv_pca_points=1,inv_seed=42,inv_val_max_episodes=None,inv_num_steps=20,inv_refine=4)
    probe.run(SimpleNamespace(chunk_size=30),None,None,own,prepare,dict(checkpoint='test'))
    d=json.loads((tmp_path/'noise_capture.json').read_text())
    assert len(calls)==5
    assert {r['kind'] for r in d['records']}==set(probe.KINDS)
    assert max(r['mse'] for r in d['records'])<1e-10
    assert d['header']['complete']
    with np.load(tmp_path/'noise_0000.npz') as a:
        assert a['noise_input'].shape==(5,30,8)
        assert np.all(a['noise_input'][...,7]==0)
