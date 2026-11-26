"""Singular value subset training (docs/notes/singular_subset.md).

Each attention / MLP matrix W of the selected blocks is split by its SVD, W = U S V^T, into a
random subset T of singular triplets (a fraction of the count) and the rest F:

    W = U_T M V_T^T + U_F D_F V_F^T,        M = D_T at the start.

Only the k x k matrix M is trained. U, V and D_F never change, so the triplets in F stay
singular triplets of W. The mass ||M||_F^2 = sum of the chosen s_i^2 is held fixed, and no
singular value of M may exceed `clamp` times the largest chosen s_i.

The model keeps its ordinary weights: after every optimizer step W is rewritten as
W0 + U_T (M - D_T) V_T^T, so the forward pass, checkpoints, probes and inference see merged
weights and need no change. The gradient of M is read off the gradient of W:
dL/dM = U_T^T (dL/dW) V_T. M is float32; a bfloat16 W is written with stochastic rounding
(as StochasticRoundingAdamW does), because a change of M moves most entries of W by less
than their bfloat16 spacing and rounding to nearest would drop it.

M is stored as M~ with M = sqrt(mass) M~ / ||M~||_F and ||M~||_F = k * rms(W entries), so an
entry of M~ has the scale of an entry of W and the policy learning rate carries over: an Adam
step is the same fraction of ||M|| as it is of ||W|| when W itself is trained.
"""

import logging

import torch
import torch.nn as nn

from lerobot.rl.stochastic_rounding_adamw import stochastic_round_bf16_bits_

# Weight name inside a block -> number of equal row parts (fused matrices are split into the
# matrices they hold). None: the parts are the module's fused_dims (q, k, v).
_TARGETS = {
    # language block
    "self_attn.att_proj.weight": None,
    "self_attn.attn_out.weight": 1,
    "mlp.ff_proj.weight": 2,
    "mlp.ff_out.weight": 1,
    # action expert block
    "self_attn.qkv.weight": 3,
    "self_attn.out_proj.weight": 1,
    "cross_attn.q_proj.weight": 1,
    "cross_attn.out_proj.weight": 1,
    "mlp.gate_proj.weight": 1,
    "mlp.up_proj.weight": 1,
    "mlp.down_proj.weight": 1,
}
_CLAMP_EVERY = 10


def is_singular_target(name: str) -> bool:
    in_block = ".transformer.blocks." in name or ".action_expert.blocks." in name
    return in_block and any(name.endswith(suffix) for suffix in _TARGETS)


def _row_parts(policy: nn.Module, name: str, rows: int) -> list[slice]:
    suffix = next(suffix for suffix in _TARGETS if name.endswith(suffix))
    parts = _TARGETS[suffix]
    if parts is None:
        sizes = list(policy.get_submodule(name.rsplit(".", 2)[0]).fused_dims)
    else:
        sizes = [rows // parts] * parts
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    return [slice(start, start + size) for start, size in zip(starts, sizes, strict=True)]


class _Part:
    """One matrix: a row range of a weight parameter."""

    def __init__(self, name: str, rows: slice, weight: torch.Tensor, fraction: float, clamp: float, generator):
        u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
        k = round(fraction * len(s))
        self.indices = torch.randperm(len(s), generator=generator)[:k].sort().values.to(s.device)
        self.name = name
        self.rows = rows
        self.u = u[:, self.indices].float()
        self.v = vh[self.indices].T.float()
        self.s = s[self.indices].float()
        self.mass = self.s.pow(2).sum()
        self.mass_fraction = (self.mass.double() / s.pow(2).sum()).item()
        self.cap = clamp * self.s.max()
        self.norm = k * weight.double().pow(2).mean().sqrt().float()
        self.m = nn.Parameter(torch.diag(self.s) * (self.norm / self.mass.sqrt()))

    def matrix(self) -> torch.Tensor:
        return self.m * (self.mass.sqrt() / self.m.norm())


class SingularSubset:
    def __init__(self, policy: nn.Module, fraction: float, clamp: float, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weights: list[tuple[nn.Parameter, torch.Tensor, list[_Part]]] = []
        self.steps = 0
        self.top_ratio = 1.0
        for name, param in policy.named_parameters():
            if not (param.requires_grad and is_singular_target(name)):
                continue
            parts = [
                _Part(name, rows, param.detach()[rows], fraction, clamp, generator)
                for rows in _row_parts(policy, name, param.shape[0])
            ]
            self.weights.append((param, param.detach().clone(), parts))
            for part in parts:
                logging.info(
                    f"[singular] {name}[{part.rows.start}:{part.rows.stop}] "
                    f"{tuple(part.u.shape[:1] + part.v.shape[:1])} k={len(part.s)} "
                    f"mass fraction {part.mass_fraction:.3f}"
                )
        self.parts = [part for _, _, parts in self.weights for part in parts]
        self.rounding = torch.Generator(device=self.parts[0].m.device).manual_seed(seed)
        # No write here: until the first step the weights are the loaded ones, bit for bit.
        fractions = torch.tensor([part.mass_fraction for part in self.parts])
        logging.info(
            f"[singular] {len(self.parts)} matrices, {sum(part.m.numel() for part in self.parts):,} trained "
            f"numbers; mass fraction min {fractions.min():.3f} median {fractions.median():.3f} "
            f"max {fractions.max():.3f}"
        )

    def optimizer_params(self, trainable: list[nn.Parameter]) -> list[nn.Parameter]:
        """The weights leave the optimizer (they are rewritten from M), the M~ matrices enter."""
        weight_ids = {id(param) for param, _, _ in self.weights}
        return [p for p in trainable if id(p) not in weight_ids] + [part.m for part in self.parts]

    @torch.no_grad()
    def write(self) -> None:
        for param, base, parts in self.weights:
            new = base.to(torch.float32, copy=True)
            for part in parts:
                new[part.rows] += part.u @ (part.matrix() - torch.diag(part.s)) @ part.v.T
            if param.dtype == torch.bfloat16:
                stochastic_round_bf16_bits_(new, self.rounding)
            param.copy_(new)

    @torch.no_grad()
    def project_gradients(self) -> None:
        """dL/dW -> dL/dM~, then drop the weight gradients (no optimizer owns them)."""
        for param, _, parts in self.weights:
            grad = param.grad.float()
            for part in parts:
                g = part.u.T @ grad[part.rows] @ part.v
                # M depends on M~ only through its direction: the component along M~ is zero.
                g -= (g * part.m).sum() / part.m.pow(2).sum() * part.m
                part.m.grad = g * (part.mass.sqrt() / part.m.norm())
            param.grad = None

    @torch.no_grad()
    def after_step(self) -> dict[str, float]:
        """Fixed mass, clamp, rewrite the weights. Returns the step's metrics."""
        self.steps += 1
        clamp_now = self.steps % _CLAMP_EVERY == 0
        if clamp_now:
            self.top_ratio = 0.0
        off_diagonal = {"lm": [], "expert": []}
        for part in self.parts:
            part.m.mul_(part.norm / part.m.norm())
            if clamp_now:
                matrix = part.matrix().double()
                top = torch.linalg.matrix_norm(matrix, 2)
                if top > part.cap:
                    p, sigma, qh = torch.linalg.svd(matrix)
                    matrix = (p * sigma.clamp(max=part.cap)) @ qh
                    part.m.copy_(matrix * (part.norm / matrix.norm()))
                    # Restoring the mass lifts the clamped values again by this factor.
                    top = part.cap * part.mass.sqrt() / matrix.norm()
                self.top_ratio = max(self.top_ratio, (top / part.s.max()).item())
            group = "expert" if ".action_expert." in part.name else "lm"
            off_diagonal[group].append(1 - part.m.diagonal().pow(2).sum() / part.norm**2)
        self.write()
        metrics = {"singular/top_ratio": self.top_ratio}
        for group, values in off_diagonal.items():
            if values:
                metrics[f"singular/off_diagonal_{group}"] = torch.stack(values).mean().item()
        return metrics

    def state(self) -> dict[str, dict[str, torch.Tensor | float]]:
        return {
            f"{part.name}[{part.rows.start}:{part.rows.stop}]": {
                "indices": part.indices.cpu(),
                "s": part.s.cpu(),
                "mass_fraction": part.mass_fraction,
                "m": part.matrix().detach().cpu(),
            }
            for part in self.parts
        }
