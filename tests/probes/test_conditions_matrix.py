"""Behavioral checks for the bounded, label-neutral conditions probe."""
import json
from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lerobot.probes import conditions_matrix as cm


def test_budget_deduplicates_instances_and_balances_episodes():
    samples = [dict(holdout=False, robot='rebot', episode=str(e), object_class=c,
                    phase=p, position=.3 + .1*i, instance=str(i), index=e*100+i)
               for e in range(20) for c in ('cup','cloth') for p in cm.PHASES for i in range(3)]
    held = [dict(holdout=True, robot='droid', episode='held', object_class='cup',
                 phase=p, position=.5, instance='cup', index=i) for i,p in enumerate(cm.PHASES)]
    chosen=cm._bound_samples(samples+held,39,10,42)
    assert chosen==cm._bound_samples(samples+held,39,10,42)
    assert len(chosen)==39 and sum(s['holdout'] for s in chosen)==3
    keys=[(s['episode'],s['object_class'],s['phase']) for s in chosen]
    assert len(keys)==len(set(keys))
    assert set(Counter((s['object_class'],s['phase']) for s in chosen if not s['holdout']).values())=={6}
    assert all(s['position']==.5 for s in chosen)
    assert cm._even_picks(11,1).tolist()==[5]


def test_constant_prompt_and_cache_protocol(tmp_path, monkeypatch):
    calls=[]
    class Adapter:
        chunk_size=30
        def capture_layer_representations(self, obs, task, **kwargs):
            calls.append((task,kwargs['subtask']))
            return {'encoder':{'subtask':torch.ones(2,4)},'action_expert':{'action':torch.ones(2,4)}}
    monkeypatch.setattr(cm,'_rebot_inputs',lambda *a:dict(obs={},task='put the cup away',subtask='',metadata={},extra={}))
    monkeypatch.setattr(cm,'_save_thumbs',lambda *a:None)
    samples=[dict(kind='rebot',source_key='rebot',robot='rebot',episode='ep',frame=i,
                  object_class='cup',phase=p,subtask=f'{p} the cup',holdout=False,index=i)
             for i,p in enumerate(cm.PHASES)]
    cfg=SimpleNamespace(probe_parameters=SimpleNamespace(random_seed=42))
    cm.collect(Adapter(),cfg,samples,{'rebot':None},{},str(tmp_path))
    assert calls[1::2]==[(cm.NEUTRAL_TASK,cm.NEUTRAL_SUBTASK)]*3
    assert len(set(calls[::2]))==3
    assert len(cm._load_cache(str(tmp_path))[0])==6
    meta=json.loads((tmp_path/'meta.json').read_text());meta.pop('protocol')
    (tmp_path/'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='older prompt'):
        cm._load_cache(str(tmp_path))


def test_headline_is_raw_rho_at_fixed_layer(tmp_path, monkeypatch):
    rng=np.random.default_rng(42)
    rows=[];vectors=[]
    prototypes=rng.normal(size=(6,16))
    for robot in ('rebot','droid'):
        for ep in range(4):
            for c in range(6):
                for text in cm.TEXT_CONDITIONS:
                    rows.append(dict(row=len(rows),robot=robot,source=robot,episode=f'{robot}/{ep}',
                                     object_class=('cup','cloth')[c//3],instance=('cup','cloth')[c//3],
                                     phase=cm.PHASES[c%3],holdout=ep==3,text=text))
                    vectors.append(np.stack([prototypes[c]+rng.normal(size=16)*.4 for _ in range(2)]))
    monkeypatch.setattr(cm,'N_NULL',32);monkeypatch.setattr(cm,'N_SPLITS',4)
    monkeypatch.setattr(cm,'plot_cell_frames',lambda *a:False)
    cfg=SimpleNamespace(probe_parameters=SimpleNamespace(random_seed=42,conditions_layers='',conditions_headline_layer=1))
    summary=cm.analyze(rows,{'action':np.array(vectors)}, {'action':np.ones(len(rows),bool)},cfg,str(tmp_path))
    assert summary['headline_layer']==1
    assert summary['headline']['neutral.rebot_droid']==summary['fixed_layer_action']['neutral']['rebot|droid']['rho']
    assert -1<=summary['headline']['neutral.rebot_droid']<=1
    assert (tmp_path/'index.json').is_file()
