"""Damped least-squares inverse kinematics on a :class:`SerialChain`, Jacobian by autograd.

Batched over frames. Seeded at a joint vector (corpus units) and iterated to a target pose
``(p*, R*)`` with the residual ``e = [p* - p, log(R* R^T)]`` and the step
``dq = -J^T (J J^T + damping^2 I)^{-1} e`` (``J = de/dq``). The pseudo-inverse step is minimum-norm, so a 6-DOF
arm stays on the branch of its seed and the 7-DOF Panda converges to the point on the
null-space line nearest the seed. Not used in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lerobot.model.fk.chain import SerialChain, rotation_log


@dataclass
class IKResult:
    q: torch.Tensor  # (B, dof) corpus units
    pos_err_m: torch.Tensor  # (B,)
    rot_err_rad: torch.Tensor  # (B,)
    iters: torch.Tensor  # (B,) iteration at which the frame converged, or max_iters
    converged: torch.Tensor  # (B,) bool


def _residual(chain: SerialChain, q_rad: torch.Tensor, p_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
    p, r = chain.pose(chain.from_rad(q_rad[None]))
    return torch.cat([p_t - p[0], rotation_log(r_t[None] @ r.transpose(1, 2))[0]])


def solve_ik(
    chain: SerialChain,
    p_target: torch.Tensor,
    r_target: torch.Tensor,
    q_seed: torch.Tensor,
    *,
    damping: float = 1e-3,
    max_iters: int = 100,
    tol_pos_m: float = 1e-9,
    tol_rot_rad: float = 1e-9,
) -> IKResult:
    p_t = p_target.to(torch.float64)
    r_t = r_target.to(torch.float64)
    q = chain.to_rad(torch.atleast_2d(q_seed)[:, : chain.dof].to(torch.float64)).clone()
    n = q.shape[0]
    iters = torch.full((n,), max_iters, dtype=torch.long)
    converged = torch.zeros(n, dtype=torch.bool)
    jac = torch.func.vmap(torch.func.jacrev(_residual, argnums=1), in_dims=(None, 0, 0, 0))
    res = torch.func.vmap(_residual, in_dims=(None, 0, 0, 0))
    eye = damping**2 * torch.eye(6, dtype=torch.float64)
    for it in range(max_iters):
        e = res(chain, q, p_t, r_t)
        pos_err = e[:, :3].norm(dim=1)
        rot_err = e[:, 3:].norm(dim=1)
        done = (pos_err < tol_pos_m) & (rot_err < tol_rot_rad)
        iters[done & ~converged] = it
        converged |= done
        if bool(converged.all()):
            break
        j = jac(chain, q, p_t, r_t)  # (B, 6, dof)
        dq = -(j.transpose(1, 2) @ torch.linalg.solve(j @ j.transpose(1, 2) + eye, e[:, :, None]))[:, :, 0]
        q = torch.where(converged[:, None], q, q + dq)
    e = res(chain, q, p_t, r_t)
    return IKResult(chain.from_rad(q), e[:, :3].norm(dim=1), e[:, 3:].norm(dim=1), iters, converged)
