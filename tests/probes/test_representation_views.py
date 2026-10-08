"""Native capture must preserve model tensors and distinguish old pooled caches."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lerobot.probes.representation_views import NativeRepresentationCapture, resolve_view_cache, VIEW_METADATA
from lerobot.probes import conditions_matrix as cm


class Scale(torch.nn.Module):
    def forward(self,x):return x*torch.tensor([2.,.5])

class Cross(torch.nn.Module):
    def forward(self,x,*,kv_k,kv_v):return x


def test_capture_native_values_before_pooling_and_do_not_renormalize():
    norm=Scale();cross=Cross()
    transformer=SimpleNamespace(blocks=[SimpleNamespace(attn_norm=norm)])
    expert=SimpleNamespace(blocks=[SimpleNamespace(cross_attn=cross)])
    cap=NativeRepresentationCapture(transformer,expert,{'subtask':[0,1],'missing':[]},torch.device('cpu'))
    x=torch.tensor([[[10.,1.],[2.,8.]]]);q=torch.tensor([[[3.,7.],[9.,5.]]])
    k=torch.tensor([[[[1.,2.]],[[3.,4.]]]]);v=k*7
    normalized=norm(x);out=cross(q,kv_k=k,kv_v=v)
    values=cap.result();cap.close()
    torch.testing.assert_close(values['attention_input']['encoder']['subtask'][0],normalized[0].mean(0),rtol=0,atol=0)
    torch.testing.assert_close(values['attention_input']['action_expert']['action'][0],q[0].mean(0),rtol=0,atol=0)
    torch.testing.assert_close(values['expert_key']['encoder']['subtask'][0],k[0].reshape(2,-1).mean(0),rtol=0,atol=0)
    torch.testing.assert_close(values['expert_value']['encoder']['subtask'][0],v[0].reshape(2,-1).mean(0),rtol=0,atol=0)
    assert values['expert_value']['action_expert']=={}
    assert values['attention_input']['encoder']['missing'] is None
    assert torch.equal(out,q)
    assert values['attention_input']['encoder']['subtask'].dtype==torch.float32
    assert not norm._forward_hooks and not cross._forward_pre_hooks


def test_raw_cache_is_never_silently_treated_as_native(tmp_path):
    (tmp_path/'meta.json').write_text(json.dumps({'rows':[]}))
    assert resolve_view_cache(tmp_path,'block_output')==str(tmp_path)
    with pytest.raises(ValueError,match='cannot be recovered'):
        resolve_view_cache(tmp_path,'attention_input')
    native=tmp_path/'native/attention_input';native.mkdir(parents=True)
    (native/'meta.json').write_text(json.dumps({'rows':[],'representation_view':'attention_input','capture_id':'different'}))
    with pytest.raises(ValueError,match='different capture'):
        resolve_view_cache(tmp_path,'attention_input')


def test_shared_collect_stores_native_views_with_masks_dtype_and_provenance(tmp_path,monkeypatch):
    class Adapter:
        chunk_size=30
        calls=0
        def capture_layer_representations(self,*args,**kwargs):
            self.calls+=1
            assert kwargs['capture_native_views'] is True
            raw={'encoder':{'subtask':torch.ones(2,4)},'action_expert':{'action':torch.ones(2,3)}}
            native={view:{'encoder':{'subtask':torch.full((2,4),3.,dtype=torch.float32)},'action_expert':{}}
                    for view in VIEW_METADATA if view!='block_output'}
            native['attention_input']['action_expert']={'action':torch.full((2,3),5.)}
            return {**raw,'native_views':native}
    monkeypatch.setattr(cm,'_rebot_inputs',lambda *a:dict(obs={},task='move',metadata={},extra={}))
    monkeypatch.setattr(cm,'_save_thumbs',lambda *a:None)
    adapter=Adapter();cfg=SimpleNamespace(probe_parameters=SimpleNamespace(random_seed=0,enable_subspace_spans=True))
    sample=dict(kind='rebot',source_key='r',robot='rebot',episode='e',frame=0,subtask='grasp',holdout=False,object_class='cup',phase='grasp')
    cm.collect(adapter,cfg,[sample],{'r':None},{},str(tmp_path))
    assert adapter.calls==2 # real and neutral, not one forward per view
    root=json.loads((tmp_path/'meta.json').read_text())
    for view in ('attention_input','expert_key','expert_value'):
        path=resolve_view_cache(tmp_path,view);rows,arrays,present=cm._load_cache(path)
        meta=json.loads((__import__('pathlib').Path(path)/'meta.json').read_text())
        assert len(rows)==2 and meta['capture_id']==root['capture_id']
        assert arrays['subtask'].dtype==np.float32 and present['subtask'].all()
        assert ('action' in arrays)==(view=='attention_input')
        assert meta['representation_view']==view
    assert cm._load_cache(str(tmp_path))[1]['subtask'].dtype==np.float16
