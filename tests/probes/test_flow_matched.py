import numpy as np

from lerobot.probes.flow_matched import normalized_candidates,select_shared_pairs


def test_pair_selection_matches_all_anchors_not_just_first():
    # First candidate exactly matches one anchor but misses the other two badly.
    recorded=np.array([[1,20,30],[1.05,2.05,3.05]])
    generated=np.array([[1,2,3]])
    assert select_shared_pairs(recorded,generated,[0,100],count=1)==[(1,0)]


def test_shared_pairs_use_distinct_nonoverlapping_sources():
    recorded=np.array([[1,2,3],[1.01,2.01,3.01],[4,5,6],[8,9,10]])
    generated=np.array([[1,2,3],[1.01,2.01,3.01],[4,5,6],[8,9,10]])
    pairs=select_shared_pairs(recorded,generated,[0,3,60,120])
    assert len({g for _,g in pairs})==3
    indices=np.array([0,3,60,120])[[r for r,_ in pairs]]
    assert np.min(np.diff(np.sort(indices)))>=30


def test_displacement_endpoints_are_independent_of_evaluation_anchor():
    pool=dict(ids=np.array([0,1]),state=np.array([[1]*7,[3]*7],dtype=float),
              raw=np.array([[[1.5]*7]*30,[[3.5]*7]*30]),lo=np.zeros((30,7)),span=np.ones((30,7))*2,eps=0)
    a,_=normalized_candidates(pool,0,'displacement')
    b,_=normalized_candidates(pool,1,'displacement')
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[0],a[1])
    np.testing.assert_allclose(a,-.5)


def test_distribution_selection_covers_families_and_all_anchors():
    from lerobot.probes.flow_interpolation import KINDS
    from lerobot.probes.flow_matched import select_distribution_pairs
    # The first row matches anchor 0 only; it must not win over row 1.
    recorded=np.array([[1,20,30],[1.05,2.05,3.05],[4,5,6],[7,8,9],[10,11,12]])
    fake=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    pairs=select_distribution_pairs(recorded,fake,np.arange(5)*100,KINDS)
    assert pairs==[(1,0),(2,1),(3,2),(4,3)]


def test_hardest_family_reserves_scarce_recorded_chunk():
    from lerobot.probes.flow_interpolation import KINDS
    from lerobot.probes.flow_matched import select_distribution_pairs
    recorded=np.repeat(np.array([1,.9,.89,.7,.8])[:,None],3,axis=1)
    fake=np.repeat(np.array([.9,.7,2,.8])[:,None],3,axis=1)
    pairs=select_distribution_pairs(recorded,fake,[0,3,100,200,300],KINDS)
    assert pairs==[(2,0),(3,1),(0,2),(4,3)]
