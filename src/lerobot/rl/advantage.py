"""Advantage-weighted regression weights, shared by the actor trainer and the critic probe
so the probe reports the weights the actor is actually trained with."""

from __future__ import annotations

import torch


def advantage_weights(advantage: torch.Tensor, beta: float, clip: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(w, Â) for one pooled set of advantages; statistics are taken over the rows given.

    Â = clip((A − mean A) / std A, ±clip)
    w = exp(Â / beta) / mean w          mean 1 keeps the LR of plain BC
    w = (1 − lam) + lam · w             BC floor; lam 1 = pure AWR, 0 = BC
    """
    a_hat = ((advantage - advantage.mean()) / advantage.std(correction=0).clamp_min(1e-6)).clamp(-clip, clip)
    w = torch.exp(a_hat / beta)
    w = w / w.mean()
    return (1.0 - lam) + lam * w, a_hat


def subtask_stats(advantage: torch.Tensor, groups: list) -> dict:
    """{subtask: (n, mean, var)} of A; ``groups[i]`` is row i's subtask key."""
    a = advantage.double()
    per = {}
    for g in dict.fromkeys(groups):
        ag = a[[i for i, x in enumerate(groups) if x == g]]
        per[g] = (ag.numel(), ag.mean().item(), ag.var(correction=0).item())
    return per


def subtask_table(per: dict, prior: float, floor: float) -> tuple[dict, dict]:
    """({subtask: (center, scale)}, pooled): the per-subtask standardization, estimated once.

    With n_k, μ_k, σ²_k per subtask, N = Σ n_k over K subtasks:
        μ    = Σ n_k μ_k / N                               pooled mean
        σ²_w = Σ n_k σ²_k / (N − K)                        pooled within-subtask variance
        τ²   = max(0, mean_k[(μ_k − μ)² − σ²_w / n_k])     spread of the true subtask means
        μ̃_k  = μ + n_k τ² / (n_k τ² + σ²_w) · (μ_k − μ)    empirical-Bayes mean: a subtask keeps
                                                            its own mean in proportion to how far
                                                            it rises above its sampling noise
        σ̃²_k = (n_k σ²_k + prior · σ²_w) / (n_k + prior)   variance shrunk by ``prior`` pseudo-frames
        s_k  = max(σ̃_k, floor · σ_w)
    pooled = {n, mean μ, std_within σ_w, tau τ}; a subtask missing from the table gets (μ, σ_w).
    """
    n = sum(k for k, _, _ in per.values())
    mu = sum(k * m for k, m, _ in per.values()) / n
    var_w = sum(k * v for k, _, v in per.values()) / (n - len(per))
    tau2 = max(0.0, sum((m - mu) ** 2 - var_w / k for k, m, _ in per.values()) / len(per))
    table = {
        g: (
            mu + k * tau2 / (k * tau2 + var_w) * (m - mu),
            max(((k * v + prior * var_w) / (k + prior)) ** 0.5, floor * var_w**0.5),
        )
        for g, (k, m, v) in per.items()
    }
    return table, {"n": n, "mean": mu, "std_within": var_w**0.5, "tau": tau2**0.5}


def subtask_normalize(advantage: torch.Tensor, groups: list, table: dict, pooled: dict) -> torch.Tensor:
    """z = (A − center_k) / scale_k per row, from subtask_table; advantage_weights then runs on z."""
    default = (pooled["mean"], pooled["std_within"])
    center = torch.tensor([table.get(g, default)[0] for g in groups], dtype=advantage.dtype, device=advantage.device)
    scale = torch.tensor([table.get(g, default)[1] for g in groups], dtype=advantage.dtype, device=advantage.device)
    return (advantage - center) / scale
