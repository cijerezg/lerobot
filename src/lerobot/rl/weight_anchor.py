"""Periodic convex pull of trainable weights toward their init snapshot.

Every `every_n_steps` optimization steps the live weights are interpolated
toward the captured init weights with mixing factor `alpha`:

    W_new = (1 - alpha) * W_current + alpha * W_init

After the merge the optimizer's per-parameter state (`exp_avg`, `exp_avg_sq`,
`step`) is cleared so subsequent gradient steps explore directions free from
pre-merge momentum/variance estimates.

The anchor snapshot lives in pinned CPU memory in the same dtype as the live
parameter. No extra GPU allocation persists between merges; the transient
CPU→GPU tensor created inside `lerp_` is freed when the call returns.
"""

from __future__ import annotations

import logging

import torch
from torch.optim import Optimizer


class WeightAnchor:
    def __init__(
        self,
        name: str,
        optimizer: Optimizer,
        alpha: float,
        every_n_steps: int,
    ):
        self.name = name
        self.alpha = float(alpha)
        self.every_n_steps = int(every_n_steps)
        self._init_params: list[torch.Tensor] = []
        self._live_params: list[torch.Tensor] = []

        if not self.enabled:
            return

        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    cpu_copy = p.detach().to("cpu", copy=True)
                    try:
                        cpu_copy = cpu_copy.pin_memory()
                    except RuntimeError:
                        pass
                    self._init_params.append(cpu_copy)
                    self._live_params.append(p)

        logging.info(
            f"[WeightAnchor:{self.name}] snapshotted {len(self._init_params)} tensors "
            f"(alpha={self.alpha}, every_n_steps={self.every_n_steps})"
        )

    @property
    def enabled(self) -> bool:
        return self.alpha > 0.0 and self.every_n_steps > 0

    def should_merge(self, optimization_step: int) -> bool:
        if not self.enabled:
            return False
        if optimization_step <= 0:
            return False
        return optimization_step % self.every_n_steps == 0

    @torch.no_grad()
    def merge(self, optimizer: Optimizer) -> None:
        for live_p, init_cpu in zip(self._live_params, self._init_params):
            init_gpu = init_cpu.to(live_p.device, non_blocking=True)
            live_p.data.lerp_(init_gpu, self.alpha)
        optimizer.state.clear()

    def maybe_merge(self, optimization_step: int, optimizer: Optimizer) -> bool:
        if not self.should_merge(optimization_step):
            return False
        self.merge(optimizer)
        logging.info(
            f"[WeightAnchor:{self.name}] merged at step {optimization_step} "
            f"(alpha={self.alpha}); optimizer state cleared"
        )
        return True


def build_weight_anchors(
    optimizers: dict[str, Optimizer],
    alpha: float,
    every_n_steps: int,
    targets: list[str],
) -> dict[str, WeightAnchor]:
    """Build one WeightAnchor per requested optimizer key. Missing keys are skipped."""
    anchors: dict[str, WeightAnchor] = {}
    if alpha <= 0.0 or every_n_steps <= 0 or not targets:
        return anchors
    for key in targets:
        if key not in optimizers:
            logging.warning(f"[WeightAnchor] target '{key}' not in optimizers; skipping")
            continue
        anchors[key] = WeightAnchor(
            name=key,
            optimizer=optimizers[key],
            alpha=alpha,
            every_n_steps=every_n_steps,
        )
    return anchors


def apply_weight_anchors(
    anchors: dict[str, WeightAnchor],
    optimizers: dict[str, Optimizer],
    optimization_step: int,
) -> None:
    """Run a merge on every anchor whose period divides `optimization_step`."""
    for key, anchor in anchors.items():
        anchor.maybe_merge(optimization_step, optimizers[key])
