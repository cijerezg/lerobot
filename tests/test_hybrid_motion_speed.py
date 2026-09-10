import numpy as np
import pytest
from lerobot.data_processing.annotate import hybrid_motion_speed as hybrid


def atoms(*spans,duration=3,use=True):
    return [dict(start_timestep=a,end_timestep_exclusive=b,duration_speed=duration,use_duration=use) for a,b in spans]


def test_duration_adjustment_is_bounded_and_fast_failure_stays_fast():
    scores=np.linspace(1,5,100)
    for prior in (1,2,3,4,5):
        assert np.max(np.abs(hybrid.blend_scores(scores,prior)-scores))<=.35+1e-12
    assert hybrid.quantize(hybrid.blend_scores(3.9,1))==4
    assert hybrid.quantize(hybrid.blend_scores(5,1))==5


def test_duration_can_resolve_borderline_speed_and_defaults_have_no_influence():
    assert hybrid.quantize(hybrid.blend_scores(3.45,5))==4
    assert hybrid.quantize(hybrid.blend_scores(3.55,1))==3
    assert hybrid.blend_scores(3.45,1,use_duration=False)==3.45


def test_fast_back_and_forth_is_not_penalized_for_zero_net_progress():
    q=np.tile([[0.],[.1]],(30,1))
    trace=hybrid.hybrid_trace(q,10,atoms((0,len(q)),duration=1),np.linspace(0,.5,1000))
    assert trace['joint_speed_rad_s']==pytest.approx(np.ones(len(q)-1))
    assert (hybrid.quantize(trace['speed_score'])==5).all()


def test_smoothing_reduces_fluctuations():
    v=np.tile([.1,1.],100);q=np.r_[0,np.cumsum(v/30)][:,None]
    ref=np.linspace(0,1,1000)
    trace=hybrid.hybrid_trace(q,30,atoms((0,len(q))),ref)
    raw_score=hybrid.motion_scores(v,ref)
    assert np.abs(np.diff(trace['speed_score'])).sum()<.1*np.abs(np.diff(raw_score)).sum()


def test_excluded_gaps_do_not_contaminate_adjacent_motion():
    q=np.r_[np.arange(20)/10,np.arange(20)*100,np.arange(20)/10][:,None]
    trace=hybrid.hybrid_trace(q,10,atoms((0,20),(40,60)),np.linspace(0,2,1000))
    assert trace['joint_speed_rad_s'][:19]==pytest.approx(np.ones(19))
    assert trace['joint_speed_rad_s'][40:]==pytest.approx(np.ones(19))
    assert not trace['supervision_mask'][19:40].any()


def test_invalid_states_remain_unscorable_and_overlaps_rejected():
    q=np.zeros((20,6));q[5]=np.nan
    trace=hybrid.hybrid_trace(q,10,atoms((0,20)),np.linspace(0,2,1000))
    assert not trace['valid'].any()
    with pytest.raises(ValueError,match='overlapping'):
        hybrid.hybrid_trace(q,10,atoms((0,10),(9,20)),np.linspace(0,2,1000))
