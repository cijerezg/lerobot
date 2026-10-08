import torch
import pytest

from lerobot.probes.flow_synthetic import DISTRIBUTIONS, synthetic_target


@pytest.mark.parametrize('kind', DISTRIBUTIONS)
def test_synthetic_targets_are_seeded_and_never_fill_padding(kind):
    template = torch.zeros(1, 30, 8)
    valid = torch.tensor([True]*7+[False])
    a = synthetic_target(kind, template, valid, 42)
    assert a.shape == template.shape
    assert torch.equal(a, synthetic_target(kind, template, valid, 42))
    assert not torch.equal(a, synthetic_target(kind, template, valid, 43))
    assert torch.isfinite(a).all()
    assert a[..., ~valid].eq(0).all()
    if kind in ('uniform', 'constant_uniform', 'rademacher'):
        assert a.abs().max() <= 1
    if kind == 'rademacher':
        assert a[..., valid].abs().eq(1).all()
    if kind == 'constant_uniform':
        assert torch.equal(a[:, 0], a[:, -1])
    if kind == 'wide_uniform':
        assert a.abs().max() > 1  # no silent clipping of the stress target


def test_uniform_gaussian_and_heavy_tail_have_matching_population_variance():
    template = torch.zeros(1, 100000, 1)
    valid = torch.tensor([True])
    for kind in ('uniform', 'gaussian', 'student_t3'):
        values = synthetic_target(kind, template, valid, 7)
        assert values.var().item() == pytest.approx(1/3, rel=.08)
