"""Paired recovered-noise collection and a common PCA for five action families."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.probes.flow_roundtrip import inverse_path, reconstruction_metrics
from lerobot.probes.flow_synthetic import normalization_contract, synthetic_target

KINDS = ['demo', 'uniform', 'gaussian', 'rademacher', 'student_t3']
LABELS = ['Original recorded actions', 'Uniform actions [−1,1]', 'Gaussian actions (σ=1/√3)',
          'Random ±1 actions', 'Heavy-tailed actions (t₃/3)']
COLORS = ['#1e77bd', '#e57b25', '#28976c', '#a261bb', '#dc5263']


def balanced_contexts(dataset, count, seed, stride, max_episodes=None):
    from lerobot.probes.utils import sample_episodes_evenly

    candidates = sample_episodes_evenly(dataset, count, max_episodes, seed, stride)
    by_episode = defaultdict(list)
    for sample in candidates:
        by_episode[sample[0]].append(sample)
    if len(candidates) < count:
        raise ValueError(f'Only {len(candidates)} unique frames available; requested {count}')
    episodes = list(by_episode)
    np.random.default_rng(seed).shuffle(episodes)
    quotas = {ep: 0 for ep in episodes}
    remaining = count
    while remaining:
        for ep in episodes:
            if remaining and quotas[ep] < len(by_episode[ep]):
                quotas[ep] += 1
                remaining -= 1
    selected = {ep: [by_episode[ep][i] for i in np.linspace(0,len(by_episode[ep])-1,quotas[ep],dtype=int)]
                for ep in episodes}
    # Interleave episodes so partial captures already span the validation split.
    return [selected[ep][i] for i in range(max(quotas.values())) for ep in episodes if i < quotas[ep]]


def shared_pca(vectors):
    """One globally centered, unscaled PCA; no independent per-group transformations."""
    z = np.asarray(vectors, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 3 or not np.isfinite(z).all():
        raise ValueError('PCA requires a finite matrix with at least three samples and coordinates')
    mean = z.mean(axis=0)
    centered = z-mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    # Stable signs for reproducible exports (PCA sign is otherwise arbitrary).
    for row in vt:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    components = vt[:3]
    variance = singular**2
    ratio = variance/variance.sum() if variance.sum() else np.zeros_like(variance)
    return dict(mean=mean, components=components, scores=centered@components.T,
                explained_variance_ratio=ratio, singular_values=singular)


def _write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


@torch.no_grad()
def run(adapter, dataset, cfg, own, prepare_flow, provenance):
    from lerobot.probes.utils import probe_frame_inputs, probe_image_stride

    out = Path(own.inv_out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'noise_capture.json').exists():
        raise FileExistsError(f'Choose a new inv_out; {out}/noise_capture.json exists')
    contract = normalization_contract(provenance['checkpoint'])
    samples = balanced_contexts(dataset, own.inv_pca_points, own.inv_seed,
                                probe_image_stride(cfg), own.inv_val_max_episodes)
    n, refine = own.inv_num_steps, own.inv_refine
    header = dict(provenance, num_steps=n, refine=refine, kinds=KINDS, labels=LABELS,
                  points_per_group=own.inv_pca_points, normalization_contract=contract,
                  sampler='production _generate_actions_from_inputs_with_rtc; no previous chunk',
                  inverse='batched five independent target trajectories under the same context',
                  pca_input='actual sampler input after dtype cast, flattened valid joints only',
                  sampling='balanced episode quotas, evenly spaced stride-snapped anchors, interleaved episodes',
                  samples=[dict(episode_idx=e, frame_idx=f, global_idx=g) for e,f,g in samples],
                  completed_contexts=0, complete=False)
    _write_json(out/'noise_capture.json', dict(header=header, records=[]))
    records=[]
    started=time.monotonic()
    for i,(ep,fr,idx) in enumerate(samples):
        frame=probe_frame_inputs(dataset,cfg,idx,adapter.chunk_size,with_gripper_event_targets=False)
        with prepare_flow(adapter,frame) as (demo,valid,velocity,dtype,deployed):
            if int(valid.sum()) != 7 or velocity.action_layout_id != contract['action_layout_id']:
                raise ValueError('Frame does not match the audited seven-joint ReBot normalization')
            target=torch.cat([demo]+[synthetic_target(k,demo,valid,own.inv_seed+idx) for k in KINDS[1:]],dim=0)
            # Context K/V and conditioning have batch size one and broadcast over
            # independent candidate trajectories; no attention crosses batch rows.
            recovered=inverse_path(target,velocity,n,refine)[0]
            input_noise=recovered.to(dtype)
            recon=torch.cat([deployed(input_noise[j:j+1],n).float() for j in range(len(KINDS))],dim=0)
            if not torch.isfinite(recovered).all() or not torch.isfinite(recon).all():
                raise ValueError(f'Nonfinite inversion/reconstruction at global frame {idx}')
            for j,kind in enumerate(KINDS):
                metric=reconstruction_metrics(recon[j],target[j],valid)
                records.append(dict(context_index=i,episode_idx=ep,frame_idx=fr,global_idx=idx,kind=kind,
                                    **metric,noise_rms=input_noise[j,...,valid].float().square().mean().sqrt().item(),
                                    target_max_abs=target[j,...,valid].abs().max().item(),
                                    target_fraction_outside_training_range=(target[j,...,valid].abs()>1).float().mean().item()))
            np.savez_compressed(out/f'noise_{i:04d}.npz', kinds=np.array(KINDS), valid=valid.cpu().numpy(),
                                target=target.cpu().numpy(), recovered_float32=recovered.cpu().numpy(),
                                noise_input=input_noise.float().cpu().numpy(), reconstruction=recon.cpu().numpy())
            header['forward_dtype']=str(dtype)
        header['completed_contexts']=i+1
        header['complete']=i+1==len(samples)
        _write_json(out/'noise_capture.json', dict(header=header,records=records))
        seconds=(time.monotonic()-started)/(i+1)
        print(f'Noise PCA: {i+1}/{len(samples)} contexts, {len(records)} points, '
              f'{seconds:.1f} s/context, ETA {seconds*(len(samples)-i-1)/60:.1f} min',flush=True)
        if i in (4,14,49,99) or header['complete']:
            render(out)


def render(out_dir):
    from lerobot.probes.flow_noise_pca_report import render as render_report
    render_report(out_dir)
