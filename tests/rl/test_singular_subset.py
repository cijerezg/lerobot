import torch
import torch.nn as nn

from lerobot.rl.molmoact2.singular_subset import SingularSubset


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.attn_out = nn.Linear(12, 10, bias=False)
        self.mlp = nn.Module()
        self.mlp.ff_proj = nn.Linear(10, 32, bias=False)  # two fused 16 x 10 matrices


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.blocks = nn.ModuleList([_Block()])

    def forward(self, x):
        block = self.model.transformer.blocks[0]
        a, b = block.mlp.ff_proj(block.self_attn.attn_out(x)).chunk(2, dim=-1)
        return (a * torch.tanh(b)).sum()


def _setup():
    torch.manual_seed(0)
    policy = _Policy()
    base = {name: p.detach().clone() for name, p in policy.named_parameters()}
    singular = SingularSubset(policy, fraction=0.3, clamp=4.0, seed=0)
    return policy, base, singular


def test_start_is_the_loaded_model():
    policy, base, singular = _setup()
    assert len(singular.parts) == 3
    assert [len(part.s) for part in singular.parts] == [3, 3, 3]
    for name, p in policy.named_parameters():
        assert torch.allclose(p, base[name], atol=1e-6)


def test_projected_gradient_matches_autograd():
    policy, base, singular = _setup()
    x = torch.randn(5, 12)
    policy(x).backward()
    singular.project_gradients()

    part = singular.parts[2]  # second half of ff_proj
    m = part.m.detach().clone().requires_grad_(True)
    weights = {name: p.clone() for name, p in base.items()}
    name = "model.transformer.blocks.0.mlp.ff_proj.weight"
    matrix = m * (part.mass.sqrt() / m.norm())
    delta = part.u @ (matrix - torch.diag(part.s)) @ part.v.T
    weights[name] = torch.cat([weights[name][:16], weights[name][16:] + delta])
    torch.func.functional_call(policy, weights, (x,)).backward()
    assert torch.allclose(part.m.grad, m.grad, atol=1e-5)
    assert all(p.grad is None for p in policy.parameters())


def test_step_keeps_frozen_values_and_mass():
    policy, base, singular = _setup()
    optimizer = torch.optim.AdamW(singular.optimizer_params(list(policy.parameters())), lr=0.05)
    for _ in range(20):
        policy(torch.randn(5, 12)).backward()
        singular.project_gradients()
        optimizer.step()
        optimizer.zero_grad()
        metrics = singular.after_step()
    assert metrics["singular/off_diagonal_lm"] > 0

    part = singular.parts[0]
    name = "model.transformer.blocks.0.self_attn.attn_out.weight"
    before = torch.linalg.svdvals(base[name].double())
    after = torch.linalg.svdvals(dict(policy.named_parameters())[name].detach().double())
    assert not torch.allclose(before, after, atol=1e-3)
    assert torch.allclose(before.pow(2).sum(), after.pow(2).sum(), rtol=1e-5)
    frozen = torch.ones(len(before), dtype=torch.bool)
    frozen[part.indices] = False
    # every frozen value is still a singular value of the trained weight
    assert (before[frozen][:, None] - after[None, :]).abs().min(dim=1).values.max() < 1e-5
    assert after.max() <= 4.0 * part.s.max() * 1.05
