"""Numerical round-trip checks shared by the standalone flow inversion experiment.

Flow time runs from noise (0) to action (1). All paths include both endpoints.
The inverse stays float32; forward continuation can use the deployment dtype.
"""

from __future__ import annotations

import torch


def inverse_path(target, velocity, num_steps: int, refine: int = 0):
    """Reverse Euler with optional fixed-point inversion of each forward step.

Refinement solves y + dt*v(y, t_prev) = x; finite iteration is not guaranteed
to converge. The forward reconstruction measures whether it actually worked.
"""
    if num_steps < 1 or refine < 0:
        raise ValueError("num_steps must be positive and refine nonnegative")
    x = target.float().clone()
    path = [x]
    for k in range(num_steps, 0, -1):
        y = x - velocity(x, k / num_steps).float() / num_steps
        for _ in range(refine):
            y = x - velocity(y, (k - 1) / num_steps).float() / num_steps
        x = y
        path.append(x)
    return torch.stack(path[::-1])


def forward_path(start, velocity, num_steps: int, start_step: int = 0):
    """Continue on the original k/N grid, retaining start's accumulation dtype."""
    if num_steps < 1 or not 0 <= start_step <= num_steps:
        raise ValueError("require num_steps > 0 and 0 <= start_step <= num_steps")
    x = start.clone()
    path = [x]
    for k in range(start_step, num_steps):
        x = x + (1.0 / num_steps) * velocity(x, k / num_steps).to(x.dtype)
        path.append(x)
    return torch.stack(path)


def reconstruction_metrics(recon, target, valid):
    """Score valid action dimensions only, in processed action coordinates."""
    error = (recon.float() - target.float())[..., valid]
    if not error.numel():
        raise ValueError("no valid action coordinates")
    if not torch.isfinite(error).all():
        raise ValueError("non-finite round-trip reconstruction")
    return {
        "mse": error.square().mean().item(),
        "rms": error.square().mean().sqrt().item(),
        "mae": error.abs().mean().item(),
        "max_abs": error.abs().max().item(),
    }


def experiment_grid(own):
    steps = sorted(set(int(v) for v in own.inv_roundtrip_steps.split(',')))
    refinements = sorted(set(int(v) for v in own.inv_roundtrip_refinements.split(',')))
    times = sorted(set(float(v) for v in own.inv_roundtrip_times.split(',')))
    if not steps or min(steps) < 1 or not refinements or min(refinements) < 0:
        raise ValueError('step counts must be positive and refinements nonnegative')
    if not times or any(not 0 <= t <= 1 for t in times) or 0 not in times:
        raise ValueError('restart times must be in [0,1] and include 0')
    for n in steps:
        if any(abs(t * n - round(t * n)) > 1e-7 for t in times):
            raise ValueError(f'restart times must lie on the k/{n} grid')
    return steps, refinements, times


@torch.no_grad()
def run_experiment(adapter, dataset, cfg, own, prepare_flow, header):
    """Capture paired demonstrations / generated controls, without touching training."""
    import json
    from pathlib import Path

    import numpy as np

    from lerobot.probes.utils import probe_frame_inputs, probe_image_stride, sample_episodes_evenly

    from lerobot.probes.flow_synthetic import (
        DISTRIBUTIONS, noise_statistics, normalization_contract, selected_distributions, synthetic_target,
    )

    steps, refinements, times = experiment_grid(own)
    distributions = selected_distributions(own)
    contract = normalization_contract(header['checkpoint']) if distributions else None
    out = Path(own.inv_out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'roundtrip.json').exists():
        raise FileExistsError(f'Choose a new inv_out; {out}/roundtrip.json already exists')
    samples = sample_episodes_evenly(
        dataset, own.inv_val_frames_per_episode, own.inv_val_max_episodes,
        own.inv_seed, probe_image_stride(cfg),
    )
    if not samples:
        raise ValueError('no validation frames selected')
    header = dict(header, step_counts=steps, refinements=refinements, restart_times=times,
                  frames_per_episode=own.inv_val_frames_per_episode,
                  max_episodes=own.inv_val_max_episodes, n_frames=len(samples),
                  split='val', inverse_dtype='float32', units='processed action coordinates',
                  rtc='no prior chunk; production RTC applies no guidance',
                  generated_control='native forward of seeded standard-normal noise',
                  padding='joint padding excluded; episode-end repeated targets retained',
                  torch_version=torch.__version__)
    if distributions:
        header['synthetic_distributions'] = {k: dict(label=DISTRIBUTIONS[k][0], description=DISTRIBUTIONS[k][1]) for k in distributions}
        header['normalization_contract'] = contract
        header['synthetic_sampling'] = 'one independently seeded chunk per frame/distribution, held fixed across solver settings'
    records = []
    header['completed_frames'] = 0
    header['complete'] = False
    for i, (ep, frame_idx, global_idx) in enumerate(samples):
        frame = probe_frame_inputs(dataset, cfg, global_idx, adapter.chunk_size,
                                   with_gripper_event_targets=False)
        arrays = {}
        with prepare_flow(adapter, frame) as (demo, valid, velocity, dtype, deployed):
            header['forward_dtype'] = str(dtype)
            native_velocity = lambda x, t: velocity(x, t, native=True)
            generator = torch.Generator(device=demo.device).manual_seed(own.inv_seed + global_idx)
            noise = torch.randn(demo.shape, dtype=dtype, device=demo.device, generator=generator)
            if adapter._policy.config.mask_action_dim_padding:
                noise[..., ~valid] = 0
            arrays['valid'] = valid.cpu().numpy()
            arrays['demo'] = demo.cpu().numpy()
            arrays['control_noise'] = noise.float().cpu().numpy()
            synthetic = []
            if distributions:
                if int(valid.sum()) != contract['valid_joint_count']:
                    raise ValueError('Actual valid joint mask does not match the audited ReBot normalization')
                if getattr(velocity, 'action_layout_id', None) != contract['action_layout_id']:
                    raise ValueError('Actual frame action_layout_id does not match the audited normalization row')
                for kind in distributions:
                    target = synthetic_target(kind, demo, valid, own.inv_seed + global_idx)
                    synthetic.append((kind, target))
                    arrays[f'target_{kind}'] = target.cpu().numpy()
            for n in steps:
                generated = forward_path(noise, native_velocity, n)[-1].float()
                target_cases = synthetic if distributions else [('demo', demo), ('generated', generated)]
                for kind, target in target_cases:
                    for refine in refinements:
                        key = f'{kind}_n{n}_r{refine}'
                        inverse = inverse_path(target, velocity, n, refine)
                        arrays[key + '_inverse'] = inverse.cpu().numpy()
                        fp32 = forward_path(inverse[0], velocity, n)
                        arrays[key + '_forward_fp32'] = fp32.cpu().numpy()
                        deployed_recon = deployed(inverse[0], n).float()
                        arrays[key + '_deployed'] = deployed_recon.cpu().numpy()
                        print(f'Frame {i + 1}/{len(samples)}: {kind}, {n} steps, {refine} corrections', flush=True)
                        for t in times:
                            k = round(t * n)
                            path = forward_path(inverse[k].to(dtype), native_velocity, n, k).float()
                            arrays[f'{key}_forward_k{k}'] = path.cpu().numpy()
                            metrics = reconstruction_metrics(path[-1], target, valid)
                            row = dict(
                                episode_idx=int(ep), frame_idx=int(frame_idx), global_idx=int(global_idx),
                                target_kind=kind, num_steps=n, refine=refine, start_step=k, start_time=t,
                                **metrics,
                                **noise_statistics(inverse[0], valid),
                                prior_reference_sq_norm_over_d=noise[..., valid].float().square().mean().item(),
                                target_fraction_outside_training_range=(target[..., valid].abs()>1).float().mean().item(),
                                target_max_abs=target[..., valid].abs().max().item(),
                                cast_only_mse=reconstruction_metrics(target.to(dtype), target, valid)['mse'],
                                fp32_rms=reconstruction_metrics(fp32[-1], target, valid)['rms'],
                                random_baseline_rms=reconstruction_metrics(generated, target, valid)['rms'],
                                target_rms=target[..., valid].square().mean().sqrt().item(),
                                noise_rms=inverse[0][..., valid].square().mean().sqrt().item(),
                                target_error_by_step=[reconstruction_metrics(x, target, valid)['rms'] for x in path],
                                retrace_error_by_step=[reconstruction_metrics(x, y, valid)['rms']
                                                       for x, y in zip(path, inverse[k:])],
                            )
                            if kind == 'generated':
                                row['noise_recovery_rms'] = reconstruction_metrics(inverse[0], noise, valid)['rms']
                            if k == 0:
                                row['deployment_rms'] = reconstruction_metrics(deployed_recon, target, valid)['rms']
                                row['sampler_parity_max_abs'] = reconstruction_metrics(path[-1], deployed_recon, valid)['max_abs']
                                # Partial-time continuations share this loop. Verify its full-time
                                # endpoint against production before interpreting any of them.
                                torch.testing.assert_close(path[-1], deployed_recon, rtol=0, atol=0)
                            records.append(row)
        np.savez_compressed(out / f'trajectories_{i:04d}.npz', **arrays)
        header['completed_frames'] = i + 1
        header['complete'] = i + 1 == len(samples)
        (out / 'roundtrip.json').write_text(json.dumps({'header': header, 'records': records}, indent=2, allow_nan=False))
        print(f'Round trip: {i + 1}/{len(samples)} frames saved', flush=True)
    render(out)


def render(out_dir):
    """Rebuild the MSE-first report from saved arrays; no model evaluation."""
    import json
    from pathlib import Path

    data = json.loads((Path(out_dir) / 'roundtrip.json').read_text())
    if data['header'].get('synthetic_distributions'):
        from lerobot.probes.flow_reachability_report import render as render_report
    else:
        from lerobot.probes.flow_roundtrip_report import render as render_report

    render_report(out_dir)
