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
