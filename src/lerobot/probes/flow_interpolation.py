"""Fixed-observation action interpolations and their recovered input paths."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.probes.flow_noise_pca import _write_json
from lerobot.probes.flow_roundtrip import inverse_path, reconstruction_metrics
from lerobot.probes.flow_synthetic import normalization_contract, synthetic_target

KINDS=['uniform','gaussian','rademacher','student_t3']
NAMES=['Recorded → uniform','Recorded → Gaussian','Recorded → random ±1','Recorded → heavy tails','Uniform → random ±1']
COLORS=['#e57b25','#28976c','#a261bb','#dc5263','#586c7c']
ALPHAS=np.linspace(0,1,11)


def action_paths(demo,endpoints,*,bridge=True):
    """50 unique targets: shared original, four 10-point rays, nine bridge interiors."""
    targets=[demo]
    indices=[]
    for kind in endpoints:
        path=[0]
        for alpha in ALPHAS[1:]:
            path.append(len(targets))
            targets.append((1-float(alpha))*demo+float(alpha)*endpoints[kind])
        indices.append(path)
    if not bridge:
        return torch.cat(targets,dim=0),np.array(indices,dtype=np.int64)
    bridge=[indices[0][-1]]
    for alpha in ALPHAS[1:-1]:
        bridge.append(len(targets))
        targets.append((1-float(alpha))*endpoints['uniform']+float(alpha)*endpoints['rademacher'])
    bridge.append(indices[2][-1])
    indices.append(bridge)
    return torch.cat(targets,dim=0),np.array(indices,dtype=np.int64)


def path_geometry(targets,noise,path_indices):
    """Full-dimensional distances; PCA is used for display only."""
    target=np.asarray(targets,dtype=float).reshape(len(targets),-1)
    z=np.asarray(noise,dtype=float).reshape(len(noise),-1)
    curves=[]
    for index in path_indices:
        p=z[index];a=target[index]
        straight=(1-ALPHAS[:,None])*p[0]+ALPHAS[:,None]*p[-1]
        action_steps=np.sqrt(np.mean(np.diff(a,axis=0)**2,axis=1))
        noise_steps=np.sqrt(np.mean(np.diff(p,axis=0)**2,axis=1))
        curves.append(dict(distance_from_original=np.sqrt(np.mean((p-z[0])**2,axis=1)),
                           straight_line_deviation=np.sqrt(np.mean((p-straight)**2,axis=1)),
                           step_stretch=np.divide(noise_steps,action_steps,out=np.zeros_like(noise_steps),where=action_steps>1e-12)))
    return curves


@torch.no_grad()
def run(adapter,dataset,cfg,own,prepare_flow,provenance):
    from PIL import Image
    from lerobot.probes.utils import as_image,build_episode_index,probe_frame_inputs,probe_image_stride

    out=Path(own.inv_out);out.mkdir(parents=True,exist_ok=True)
    if (out/'interpolation.json').exists():raise FileExistsError('Choose a new inv_out; interpolation.json exists')
    contract=normalization_contract(provenance['checkpoint'])
    source_rows=json.loads((Path(dataset.root)/'meta/provenance.json').read_text())
    sources={int(r['episode_index']):r for r in source_rows}
    episodes=[int(e) for e in own.inv_interpolation_episodes.split(',')]
    index=build_episode_index(dataset);stride=probe_image_stride(cfg)
    header=dict(provenance,num_steps=own.inv_num_steps,refine=own.inv_refine,alphas=ALPHAS.tolist(),
                names=NAMES,kinds=KINDS,normalization_contract=contract,
                selection='40% through each specified own-ReBot episode, snapped down to image stride',
                inverse_batch_size=own.inv_interpolation_batch_size,completed_frames=0,complete=False)
    generated=getattr(own,"inv_generated_actions",False)
    header["generated_actions"]=generated
    if generated:
        header["kinds"]=["sample_1","sample_2","sample_3"]
        header["names"]=[f"Recorded → generated action {i+1}" for i in range(3)]
        header["noise_distribution"]="Standard Gaussian N(0,1), sampled directly in production dtype; padding masked"
    records=[];started=time.monotonic()
    for i,ep in enumerate(episodes):
        if ep not in index:raise ValueError(f'Missing episode {ep}')
        source=sources[ep]
        if 'external_validation' in json.dumps(source):raise ValueError(f'Episode {ep} is external data, not own ReBot')
        position=int(.4*(len(index[ep])-1));position-=position%stride
        global_idx=index[ep][position]
        frame=probe_frame_inputs(dataset,cfg,global_idx,adapter.chunk_size,with_gripper_event_targets=False)
        photographs=[]
        for key,tensor in frame['obs'].items():
            if key.startswith('observation.images.'):
                camera=key.rsplit('.',1)[-1];name=f'frame_{i}_{camera}.jpg'
                Image.fromarray(as_image(tensor)).save(out/name,quality=92)
                photographs.append(dict(camera=camera,file=name))
        frame_started=time.monotonic()
        with prepare_flow(adapter,frame) as (demo,valid,velocity,dtype,deployed):
            if int(valid.sum())!=7 or velocity.action_layout_id!=contract['action_layout_id']:
                raise ValueError('Unexpected action normalization layout')
            extra_arrays={}
            if generated:
                seeds=[own.inv_seed+global_idx+100003*(j+1) for j in range(3)]
                sampled=[];endpoints={}
                for kind,seed in zip(header['kinds'],seeds):
                    generator=torch.Generator(device=demo.device).manual_seed(seed)
                    z=torch.randn(demo.shape,device=demo.device,dtype=dtype,generator=generator)
                    z[...,~valid]=0
                    sampled.append(z.float())
                    endpoints[kind]=deployed(z,own.inv_num_steps).float()
                extra_arrays=dict(sampled_noise=torch.cat(sampled).cpu().numpy(),
                                  generated_actions=torch.cat(list(endpoints.values())).cpu().numpy(),
                                  noise_seeds=np.array(seeds,dtype=np.int64))
            else:
                endpoints={k:synthetic_target(k,demo,valid,own.inv_seed+global_idx) for k in KINDS}
            target,paths=action_paths(demo,endpoints,bridge=not generated)
            recovered=[]
            bs=own.inv_interpolation_batch_size
            for lo in range(0,len(target),bs):
                recovered.append(inverse_path(target[lo:lo+bs],velocity,own.inv_num_steps,own.inv_refine)[0])
            recovered=torch.cat(recovered)
            noise=recovered.to(dtype)
            # Use the actual production sampler individually, preserving the exact
            # per-observation conditioning, as in the previous round-trip experiment.
            recon=torch.cat([deployed(noise[j:j+1],own.inv_num_steps).float() for j in range(len(target))])
            metrics=[reconstruction_metrics(recon[j],target[j],valid) for j in range(len(target))]
            np.savez_compressed(out/f'paths_{i:02d}.npz',target=target.cpu().numpy(),
                                recovered_float32=recovered.cpu().numpy(),noise_input=noise.float().cpu().numpy(),
                                reconstruction=recon.cpu().numpy(),valid=valid.cpu().numpy(),
                                path_indices=paths,alphas=ALPHAS,**extra_arrays)
            header['forward_dtype']=str(dtype)
        record=dict(episode_idx=ep,frame_idx=int(frame['frame_idx']),global_idx=global_idx,
                    task=frame['task'],subtask=frame['subtask'],metadata=frame['metadata'],
                    source=source,photographs=photographs,metrics=metrics,
                    elapsed_seconds=time.monotonic()-frame_started)
        records.append(record);header['completed_frames']=len(records);header['complete']=len(records)==len(episodes)
        header['elapsed_seconds']=time.monotonic()-started
        _write_json(out/'interpolation.json',dict(header=header,frames=records))
        print(f'Interpolation: {i+1}/{len(episodes)} frames, {len(target)} unique targets/frame; '
              f'{record["elapsed_seconds"]:.1f}s this frame, {header["elapsed_seconds"]:.1f}s total',flush=True)
    render(out)


def render(out_dir):
    from lerobot.probes.flow_interpolation_report import render as render_report
    render_report(out_dir)
