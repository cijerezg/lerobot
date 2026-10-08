"""Match recorded displacement patterns to direct fake action samples across anchors."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.probes.flow_interpolation import ALPHAS, KINDS, action_paths
from lerobot.probes.flow_noise_pca import _write_json
from lerobot.probes.flow_roundtrip import inverse_path, reconstruction_metrics
from lerobot.probes.flow_synthetic import DISTRIBUTIONS, normalization_contract, synthetic_target


def candidate_pool(dataset_root,checkpoint,stride=9):
    import pyarrow.parquet as pq
    from safetensors.numpy import load_file

    root=Path(dataset_root)
    rows=json.loads((root/'meta/provenance.json').read_text())
    own=[int(r['episode_index']) for r in rows if 'external_validation' not in json.dumps(r)]
    import pyarrow as pa
    table=pa.concat_tables([pq.read_table(p) for p in sorted(root.glob('data/**/*.parquet'))]).to_pandas().sort_values('index')
    actions=np.stack(table['action']).astype(np.float32);state=np.stack(table['observation.state']).astype(np.float32)
    ep=table['episode_index'].to_numpy();fr=table['frame_index'].to_numpy();gid=table['index'].to_numpy()
    np.testing.assert_array_equal(gid,np.arange(len(gid)))
    ids=np.flatnonzero(fr%stride==0)
    ids=np.array([g for g in ids if g+29<len(ep) and ep[g+29]==ep[g]])
    chunks=np.stack([actions[g:g+30,:7] for g in ids])
    config=json.loads((Path(checkpoint)/'policy_preprocessor.json').read_text())
    normalizer=next(s for s in config['steps'] if s['registry_name']=='molmoact2_masked_normalizer')
    stats=load_file(str(Path(checkpoint)/normalizer['state_file']))
    layout=normalizer['config']['embodiment_names'].index('rebot_b601_joint7_commanded')
    lo=stats['action.q01'][layout,:,:7];span=stats['action.q99'][layout,:,:7]-lo
    return dict(ids=ids,raw=chunks,state=state,ep=ep,fr=fr,lo=lo,span=span,
                eps=normalizer['config']['eps'],own_episodes=own,pool_episodes=sorted(set(ep[ids].tolist())),source=rows)


def normalized_candidates(pool,anchor,mode):
    state=pool['state'][pool['ids']] if mode=='displacement' else np.broadcast_to(pool['state'][anchor],(len(pool['ids']),pool['state'].shape[-1]))
    values=2*(pool['raw']-state[:,None,:7]-pool['lo'])/(pool['span']+pool['eps'])-1
    # Preserve training target coordinates for displacement patterns; do not clip
    # absolute transfers, which would silently change the requested joint positions.
    return (np.clip(values,-1,1) if mode=='displacement' else values),np.mean(np.abs(values)>1,axis=(1,2))


def select_shared_pairs(recorded_distance,generated_distance,candidate_ids,*,count=3):
    """Minimize worst relative distance mismatch across all anchors; no outcome selection."""
    rd=np.asarray(recorded_distance,dtype=float);gd=np.asarray(generated_distance,dtype=float)
    error=(np.abs(rd[:,None,:]-gd[None,:,:])/np.maximum(gd[None,:,:],1e-12)).max(axis=2)
    selected=[];available=error.copy()
    for _ in range(count):
        r,g=np.unravel_index(np.argmin(available),available.shape)
        if not np.isfinite(available[r,g]):raise ValueError('Not enough distinct shared matches')
        selected.append((int(r),int(g)))
        available[:,g]=np.inf
        available[np.abs(np.asarray(candidate_ids)-candidate_ids[r])<30,:]=np.inf
    return sorted(selected,key=lambda p:np.mean(gd[p[1]]))


def select_distribution_pairs(recorded_distance, fake_distance, candidate_ids, fake_kinds):
    """One pair per distribution, minimizing worst mismatch over all anchors.

    Greedy with the hardest-to-match distribution first; chunks do not overlap.
    The selection uses action distances only, never inverse outcomes.
    """
    rd=np.asarray(recorded_distance,dtype=float)
    fd=np.asarray(fake_distance,dtype=float)
    available=np.ones(len(rd),dtype=bool);pairs=[]
    def best_error(kind):
        ids=np.flatnonzero(np.asarray(fake_kinds)==kind)
        return float((np.abs(rd[:,None,:]-fd[None,ids,:])/np.maximum(fd[None,ids,:],1e-12)).max(axis=2).min())
    order=sorted(KINDS,key=best_error,reverse=True)
    for kind in order:
        ids=np.flatnonzero(np.asarray(fake_kinds)==kind)
        error=(np.abs(rd[:,None,:]-fd[None,ids,:])/np.maximum(fd[None,ids,:],1e-12)).max(axis=2)
        error[~available]=np.inf
        r,j=np.unravel_index(np.argmin(error),error.shape)
        if not np.isfinite(error[r,j]):raise ValueError(f'No eligible match for {kind}')
        pairs.append((int(r),int(ids[j])))
        available[np.abs(np.asarray(candidate_ids)-candidate_ids[r])<30]=False
    return sorted(pairs,key=lambda pair:KINDS.index(fake_kinds[pair[1]]))


@torch.no_grad()
def run(adapter,dataset,cfg,own,prepare_flow,provenance):
    from PIL import Image
    from lerobot.probes.utils import as_image,probe_frame_inputs

    out=Path(own.inv_out);out.mkdir(parents=True,exist_ok=True)
    if (out/'matched.json').exists():raise FileExistsError('Choose a new output directory; matched.json exists')
    if own.inv_match_coordinates!='displacement':raise ValueError('Fixed endpoint tensors across anchors require displacement coordinates')
    started=time.monotonic()
    contract=normalization_contract(provenance['checkpoint'])
    pool=candidate_pool(dataset.root,provenance['checkpoint'],stride=3)
    anchor_ids=[1137,4290,14418]
    contexts=[probe_frame_inputs(dataset,cfg,gid,adapter.chunk_size,with_gripper_event_targets=False) for gid in anchor_ids]
    assert [(f['episode_idx'],f['frame_idx']) for f in contexts]==[(0,1137),(1,1446),(6,813)]
    demos=[]
    for frame in contexts:
        with prepare_flow(adapter,frame) as (demo,mask,velocity,dtype,deployed):
            assert demo.shape[1]==30 and int(mask.sum())==7 and velocity.action_layout_id==contract['action_layout_id']
            if demos:np.testing.assert_array_equal(mask.cpu().numpy(),valid)
            valid=mask.cpu().numpy();demos.append(demo.cpu().numpy().copy())
    anchors=np.concatenate(demos)[...,valid]
    candidates,clip_fraction=normalized_candidates(pool,anchor_ids[0],'displacement')
    # Modest deterministic CPU pool; all samples are directly normalized actions.
    fake=[];fake_kinds=[];seeds=[]
    for kind in KINDS:
        for j in range(64):
            seed=own.inv_seed+j
            fake.append(synthetic_target(kind,torch.from_numpy(demos[0]),torch.from_numpy(valid),seed)[0].numpy())
            fake_kinds.append(kind);seeds.append(seed)
    fake=np.stack(fake)
    rd=((candidates[:,None]-anchors[None])**2).mean((2,3))
    fd=((fake[...,valid][:,None]-anchors[None])**2).mean((2,3))
    eligible=(rd.min(axis=1)>=.1) # exclude tiny perturbations of any anchor
    for f in contexts:eligible&=(pool['ep'][pool['ids']]!=f['episode_idx'])|(np.abs(pool['ids']-f['global_idx'])>=60)
    selection_rd=rd.copy();selection_rd[~eligible]=np.inf
    pairs=select_distribution_pairs(selection_rd,fd,pool['ids'],fake_kinds)
    worst=max(float(np.max(abs(rd[r]-fd[g])/fd[g])) for r,g in pairs)
    header=dict(provenance,normalization_contract=contract,coordinates='displacement',shared_endpoints=True,
                comparison='recorded_vs_direct_fake',pool_size=len(pool['ids']),pool_own_episodes=pool['own_episodes'],pool_episodes=pool['pool_episodes'],candidate_stride=3,
                fake_pool_size=len(fake),fake_candidates_per_distribution=64,kinds=KINDS,
                minimum_recorded_endpoint_mse=.1,eligible_recorded_count=int(eligible.sum()),complete=False,completed_frames=0,
                selection='One pair per fake distribution, hardest-to-match distribution first, greedily minimizing worst relative endpoint-MSE mismatch across all three anchors; same eight endpoints everywhere; no rescaling or inverse-outcome selection',
                alphas=ALPHAS.tolist(),inverse_batch_size=own.inv_interpolation_batch_size,worst_shared_mismatch=worst)
    print(f'Shared selection: {eligible.sum()} eligible recorded / {len(fake)} direct fake samples; worst mismatch {worst:.2%}',flush=True)
    def photos(context,prefix):
        result=[]
        for key,x in context['obs'].items():
            if key.startswith('observation.images.'):
                camera=key.rsplit('.',1)[-1];filename=f'{prefix}_{camera}.jpg'
                Image.fromarray(as_image(x)).save(out/filename,quality=90)
                result.append(dict(file=filename,camera=camera))
        return result
    shared=[];fixed_endpoints=[]
    for p,(r,g) in enumerate(pairs):
        source_gid=int(pool['ids'][r]);source=probe_frame_inputs(dataset,cfg,source_gid,adapter.chunk_size,with_gripper_event_targets=False)
        # Verify the manually pooled representation against actual preprocessing,
        # including the saved mask, per-step quantiles, anchor encoding and clamp.
        with prepare_flow(adapter,source) as (source_demo,source_mask,_,_,_):
            np.testing.assert_array_equal(source_mask.cpu().numpy(),valid)
            np.testing.assert_allclose(source_demo[0,:,source_mask].cpu().numpy(),candidates[r],rtol=1e-5,atol=1e-6)
        real=np.zeros_like(fake[g]);real[:,valid]=candidates[r]
        fixed_endpoints.extend([real,fake[g]])
        shared.append(dict(pair=p+1,recorded_global_idx=source_gid,recorded_episode_idx=int(pool['ep'][source_gid]),
            recorded_frame_idx=int(pool['fr'][source_gid]),fake_candidate_index=g,distribution=fake_kinds[g],
            distribution_name=DISTRIBUTIONS[fake_kinds[g]][0],seed=seeds[g],sampling_device='cpu',
            recorded_distances=rd[r].tolist(),fake_distances=fd[g].tolist(),
            recorded_clipped_fraction=float(clip_fraction[r]),fake_fraction_outside_unit=float(np.mean(abs(fake[g][:,valid])>1)),
            fake_max_abs=float(abs(fake[g][:,valid]).max()),task=source['task'],subtask=source['subtask'],
            photographs=photos(source,f'shared_pair_{p+1}'),
            provenance=next(s for s in pool['source'] if int(s['episode_index'])==int(pool['ep'][source_gid]))))
        print(f'  {fake_kinds[g]}: recorded frame {source_gid}, distances {rd[r].round(4)} vs {fd[g].round(4)}',flush=True)
    fixed_endpoints=np.stack(fixed_endpoints)
    np.savez_compressed(out/'shared_endpoints.npz',endpoints=fixed_endpoints,valid=valid,anchors=np.concatenate(demos),
        selected_raw_recorded=np.stack([pool['raw'][r] for r,_ in pairs]),
        recorded_pool_distances=rd,fake_pool_distances=fd,recorded_pool_ids=pool['ids'],recorded_eligible=eligible,
        fake_pool=fake,fake_kinds=np.array(fake_kinds),fake_seeds=np.array(seeds),selected_pairs=np.array(pairs))
    frames=[]
    for i,frame in enumerate(contexts):
        gid=frame['global_idx'];frame_started=time.monotonic()
        with prepare_flow(adapter,frame) as (demo,valid_mask,velocity,dtype,deployed):
            np.testing.assert_array_equal(demo.cpu().numpy(),demos[i])
            np.testing.assert_array_equal(valid_mask.cpu().numpy(),valid)
            endpoints={f'endpoint_{j}':torch.from_numpy(x[None]).to(demo.device) for j,x in enumerate(fixed_endpoints)}
            target,indices=action_paths(demo,endpoints,bridge=False)
            np.testing.assert_array_equal(target[indices[:,-1]].cpu().numpy(),fixed_endpoints)
            recovered=torch.cat([inverse_path(target[lo:lo+own.inv_interpolation_batch_size],velocity,own.inv_num_steps,own.inv_refine)[0]
                                 for lo in range(0,len(target),own.inv_interpolation_batch_size)])
            noise=recovered.to(dtype)
            recon=torch.cat([deployed(z[None],own.inv_num_steps).float() for z in noise])
            if not torch.isfinite(recon).all() or not torch.isfinite(noise).all():raise ValueError('Nonfinite inverse or reconstruction')
            metrics=[reconstruction_metrics(recon[j],target[j],valid_mask) for j in range(len(target))]
            np.savez_compressed(out/f'matched_{i:02d}.npz',target=target.cpu().numpy(),noise_input=noise.float().cpu().numpy(),
                recovered_float32=recovered.cpu().numpy(),reconstruction=recon.cpu().numpy(),valid=valid,path_indices=indices,alphas=ALPHAS,
                recorded_pool_distances=rd[eligible,i],fake_pool_distances=fd[:,i])
        pair_records=[dict(p,recorded_endpoint_mse=p['recorded_distances'][i],fake_endpoint_mse=p['fake_distances'][i],
                           relative_mismatch=abs(p['recorded_distances'][i]-p['fake_distances'][i])/p['fake_distances'][i]) for p in shared]
        frames.append(dict(episode_idx=frame['episode_idx'],frame_idx=frame['frame_idx'],global_idx=gid,task=frame['task'],subtask=frame['subtask'],
            photographs=photos(frame,f'frame_{i}_anchor'),pairs=pair_records,metrics=metrics,elapsed_seconds=time.monotonic()-frame_started))
        header.update(completed_frames=len(frames),complete=len(frames)==3,elapsed_seconds=time.monotonic()-started,forward_dtype=str(dtype))
        _write_json(out/'matched.json',dict(header=header,shared_pairs=shared,frames=frames))
        print(f'Matched interpolation {i+1}/3 done; elapsed {header["elapsed_seconds"]:.1f}s',flush=True)
        from lerobot.probes.flow_matched_report import render
        render(out)
