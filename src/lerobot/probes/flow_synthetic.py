"""Synthetic action targets in the action expert's already-normalized coordinates."""
from __future__ import annotations

import json
from pathlib import Path

import torch

DISTRIBUTIONS = {
    'uniform': ('Uniform [−1, 1]', 'Independent U(−1,1) at every future step and valid joint; within the training clamp range.'),
    'gaussian': ('Gaussian, σ=1/√3', 'Independent normal coordinates with variance 1/3, matching uniform [−1,1]. Unbounded and not clipped.'),
    'rademacher': ('Random ±1 extremes', 'Independent equally likely −1 or +1 at every coordinate; sits at the training clamp boundaries.'),
    'constant_uniform': ('Constant-in-time uniform', 'One U(−1,1) value per valid joint, repeated across the chunk. Same marginal range, perfect temporal dependence.'),
    'student_t3': ('Heavy tails, t₃ / 3', 'Independent Student t with 3 degrees of freedom divided by 3. Variance 1/3, matching uniform and Gaussian; unbounded and not clipped.'),
    'wide_uniform': ('Wide uniform [−3, 3]', 'Independent U(−3,3). Deliberately extends beyond the training clamp range; never re-clamped.'),
}


def selected_distributions(own):
    names = [v.strip() for v in getattr(own, 'inv_roundtrip_distributions', '').split(',') if v.strip()]
    if names == ['all']:
        return list(DISTRIBUTIONS)
    unknown = set(names) - set(DISTRIBUTIONS)
    if unknown:
        raise ValueError(f'Unknown synthetic distributions: {sorted(unknown)}')
    return list(dict.fromkeys(names))


def normalization_contract(checkpoint):
    """Audit the saved ReBot coordinate system before injecting normalized targets."""
    path = Path(checkpoint) / 'policy_preprocessor.json'
    steps = json.loads(path.read_text())['steps']
    normalizer = next(s['config'] for s in steps if s['registry_name']=='molmoact2_masked_normalizer')
    clamp = next(s['config'] for s in steps if s['registry_name']=='molmoact2_clamp_normalized')
    anchor = next(s['config'] for s in steps if s['registry_name']=='anchor_encode')
    if normalizer['norm_map']['ACTION'] != 'QUANTILES' or anchor['encoding'] != 'anchor':
        raise ValueError('Synthetic test expects the checkpoint’s quantile-normalized anchor action coordinates')
    names = normalizer['embodiment_names']
    row = names.index('rebot_b601_joint7_commanded')
    mask = clamp['action_masks'][row]
    if mask[:7] != [True]*7 or any(mask[7:]):
        raise ValueError('Expected seven normalized ReBot joints followed by padding')
    return dict(source=str(path), action_encoding='anchor', action_normalization='QUANTILES',
                stats_index_key=normalizer['stats_index_key'], action_layout_id=row,
                action_layout=names[row], valid_joint_count=7, training_clamp=[-1,1],
                generation_space='after anchor encoding, quantile normalization and training clamp',
                synthetic_clipping=False, reconstruction_clipping=False,
                units='squared normalized action units; not raw joint units or executable commands')


def synthetic_target(kind, template, valid, seed):
    """Separate deterministic stream per distribution, invariant to menu/solver order."""
    if kind not in DISTRIBUTIONS:
        raise ValueError(f'Unknown distribution {kind}')
    gen = torch.Generator(device=template.device).manual_seed(int(seed) + 100003*(list(DISTRIBUTIONS).index(kind)+1))
    shape = (*template.shape[:-1], int(valid.sum()))
    kwargs = dict(device=template.device, dtype=torch.float32, generator=gen)
    if kind in ('uniform', 'wide_uniform'):
        values = 2*torch.rand(shape, **kwargs)-1
        if kind == 'wide_uniform': values *= 3
    elif kind == 'gaussian':
        values = torch.randn(shape, **kwargs) / (3**.5)
    elif kind == 'rademacher':
        values = (torch.rand(shape, **kwargs) >= .5).float()*2-1
    elif kind == 'constant_uniform':
        values = (2*torch.rand((*shape[:-2], 1, shape[-1]), **kwargs)-1).expand(shape)
    else:
        numerator = torch.randn(shape, **kwargs)
        chi2 = torch.randn((3, *shape), **kwargs).square().sum(0)
        values = numerator / (chi2/3).sqrt() / 3
    target = torch.zeros_like(template, dtype=torch.float32)
    target[..., valid] = values
    return target


def noise_statistics(noise, valid):
    z = noise[..., valid].float()
    return dict(noise_sq_norm_over_d=z.square().mean().item(), noise_mean=z.mean().item(),
                noise_std=z.std(unbiased=False).item(), noise_max_abs=z.abs().max().item(),
                noise_fraction_abs_gt3=(z.abs()>3).float().mean().item())
