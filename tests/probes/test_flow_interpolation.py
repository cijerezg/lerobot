import numpy as np
import torch

from lerobot.probes.flow_interpolation import ALPHAS,KINDS,action_paths,path_geometry


def test_interpolations_have_shared_endpoints_and_exact_alpha_grid():
    demo=torch.zeros(1,30,8)
    endpoints={k:torch.full_like(demo,float(i+1)) for i,k in enumerate(KINDS)}
    for x in endpoints.values():x[...,7]=0
    target,index=action_paths(demo,endpoints)
    assert target.shape==(50,30,8)
    assert index.shape==(5,11)
    assert np.all(index[:4,0]==0)
    assert index[4,0]==index[0,-1]
    assert index[4,-1]==index[2,-1]
    for j,k in enumerate(KINDS):
        for i,alpha in enumerate(ALPHAS):
            torch.testing.assert_close(target[index[j,i]],((1-alpha)*demo+alpha*endpoints[k])[0])
    for i,alpha in enumerate(ALPHAS):
        torch.testing.assert_close(target[index[4,i]],((1-alpha)*endpoints['uniform']+alpha*endpoints['rademacher'])[0])
    assert target[...,7].eq(0).all()
    assert target.max()>1  # no post-blend clipping


def test_geometry_detects_linear_and_curved_inverse_paths():
    a=ALPHAS[:,None,None]
    straight=path_geometry(a,2*a,np.arange(11)[None])[0]
    np.testing.assert_allclose(straight['straight_line_deviation'],0,atol=1e-15)
    np.testing.assert_allclose(straight['step_stretch'],2)
    curved=path_geometry(a,a**2,np.arange(11)[None])[0]
    assert curved['straight_line_deviation'][5]==.25
    assert curved['straight_line_deviation'][0]==curved['straight_line_deviation'][-1]==0


def test_three_generated_endpoints_have_31_targets_without_bridge():
    demo=torch.arange(24,dtype=torch.float32).reshape(1,3,8)/10
    demo[...,7]=0
    endpoints={f'sample_{i}':demo+i for i in range(1,4)}
    for x in endpoints.values():x[...,7]=0
    target,indices=action_paths(demo,endpoints,bridge=False)
    assert target.shape==(31,3,8)
    assert indices.shape==(3,11)
    assert np.all(indices[:,0]==0)
    assert len(np.unique(indices))==31
    for j,endpoint in enumerate(endpoints.values()):
        torch.testing.assert_close(target[indices[j,-1]],endpoint[0],rtol=0,atol=0)
        for k,alpha in enumerate(ALPHAS):
            torch.testing.assert_close(target[indices[j,k]],((1-alpha)*demo+alpha*endpoint)[0])
    assert target[...,7].eq(0).all()
    assert target.max()>1
