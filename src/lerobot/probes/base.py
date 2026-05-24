"""
Abstract interface that policies must implement to be probable.

Each policy ships an adapter under ``lerobot.probes.adapters.<policy>`` implementing
:class:`ProbablePolicy`. Probes call the adapter only — never policy internals
directly — so the same probe code works across policies.

Dispatch mirrors :meth:`lerobot.rl.rl_trainer.Trainer.for_config`: call
``ProbablePolicy.for_config(cfg, device)`` and you get the right adapter.

The interface grows as probes need more from policies. Today: enough for the
action and offline_inference probes. Future probes will add methods (e.g.
``embed_sequence``, ``forward_with_attention_capture``, ``get_value``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class AttentionCaptureResult:
    """Output of :meth:`ProbablePolicy.capture_attention`.

    Two attention kinds per layer:
      - ``cross_attn_by_layer[i]``: action queries -> encoder keys.
        Shape ``[B, n_heads, n_action_tokens, encoder_seq_len]``.
      - ``self_attn_by_layer[i]``:  action queries -> action keys.
        Shape ``[B, n_heads, n_action_tokens, n_action_tokens]``.

    Plus the metadata the probe needs to label and overlay heatmaps.
    ``encoder_segments`` is a list of ``(name, start, end)`` tuples covering
    the encoder axis (e.g. ``("img1", 0, 256), ("language", 256, 280), ...``).
    ``encoder_pad_masks`` is ``[B, encoder_seq_len]`` bool, or None if the
    policy has no padding on the encoder side.
    """

    cross_attn_by_layer: dict[int, Tensor]
    self_attn_by_layer:  dict[int, Tensor]
    encoder_segments:    list[tuple[str, int, int]]
    encoder_pad_masks:   Tensor | None
    image_tensors:       list[Tensor]
    patches_per_cam:     int
    task_tokens:         Tensor | None
    subtask_tokens:      Tensor | None
    tokenizer:           Any
    extras:              dict = field(default_factory=dict)


class ProbablePolicy(ABC):
    """Adapter over a trained policy, exposing the surface probes need."""

    def __init__(self, policy, preprocessor, postprocessor, device: torch.device, cfg):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._cfg = cfg

    # ── Construction ─────────────────────────────────────────────────────────

    @staticmethod
    def for_config(cfg, device: torch.device, dataset=None) -> "ProbablePolicy":
        """
        Instantiate the right adapter for ``cfg.policy.type``.

        Loads dataset (if not given), policy, and pre/post-processors via the
        existing Trainer + policy factories, then wraps everything in the
        matching adapter subclass.
        """
        from lerobot.datasets.factory import make_dataset
        from lerobot.policies.factory import make_policy
        from lerobot.rl.rl_trainer import Trainer

        if dataset is None:
            dataset = make_dataset(cfg)
            dataset.delta_timestamps = None
            dataset.delta_indices = None

        trainer = Trainer.for_config(cfg)
        preprocessor, postprocessor = trainer.make_processors(
            cfg, dataset=dataset, is_main_process=True
        )

        original_device = cfg.policy.device
        cfg.policy.device = device.type
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
        cfg.policy.device = original_device

        policy.eval()
        policy.to(device)

        adapter_cls = _adapter_for_type(getattr(cfg.policy, "type", None))
        return adapter_cls(policy, preprocessor, postprocessor, device, cfg)

    # ── Action prediction ────────────────────────────────────────────────────

    @abstractmethod
    def predict_action_chunk(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,
        advantage: float = 1.0,
    ) -> tuple[Tensor, Tensor, str | None]:
        """
        Run one forward pass and return predicted actions.

        Args:
            obs: observation tensors, each with batch dim 1.
            task_str: high-level task language string.
            state: current joint state, shape ``[state_dim]`` (used for
                anchor/delta action encoding by policies that support it).
            advantage: scalar advantage hint for policies trained with it
                (pi05 RL); ignored by policies that don't use it.

        Returns:
            ``(pred_unnorm, pred_norm, pred_subtask)`` where
                - ``pred_unnorm``: ``[chunk_size, action_dim]`` float32 CPU,
                  in dataset units (after anchor/delta reconstruction).
                - ``pred_norm``:   ``[chunk_size, action_dim]`` float32 CPU,
                  in normalised model space.
                - ``pred_subtask``: decoded subtask string, or ``None`` for
                  policies that don't generate subtasks.
        """

    # ── Representations ──────────────────────────────────────────────────────

    @abstractmethod
    def capture_representations(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,
        timestep: float = 1.0,
        gt_actions: Tensor | None = None,
        gt_subtask: str | None = None,
    ) -> dict[str, Tensor]:
        """Run a probe forward and return mean-pooled hidden states per site.

        Args:
            gt_actions: dataset GT action chunk ``[chunk_size, action_dim]``, used
                by policies that build action-token streams from GT (pi05). May be
                ignored by policies that don't (molmoact2).
            gt_subtask: dataset subtask string, used by pi05 to populate
                subtask_tokens in the preprocessor. Ignored by molmoact2.

        Returns ``dict[site_name -> Tensor[hidden_dim]]``. Site names are
        policy-specific:
          - pi05:      ``"prefix"`` (VLM, ~2048d), ``"suffix"`` (action expert, ~1024d)
          - molmoact2: ``"encoder"``, ``"action_expert"``

        The probe iterates over whatever sites are returned, so adapters can
        expose more sites later without touching the probe code.
        """

    # ── Critic / value head ──────────────────────────────────────────────────

    @abstractmethod
    def predict_value(self, obs: dict[str, Tensor], task_str: str) -> float:
        """Scalar V(s) for one observation."""

    @abstractmethod
    def predict_value_and_probs(
        self, obs: dict[str, Tensor], task_str: str,
    ) -> tuple[float, "np.ndarray", "np.ndarray"]:
        """Scalar V(s) plus the predicted distribution over the value support.

        Returns ``(v, probs, bin_centers)`` where ``probs`` is ``[n_bins]`` and
        ``bin_centers`` is ``[n_bins]`` (the policy's distributional support).
        """

    def value_gradient_magnitude(
        self, obs: dict[str, Tensor], task_str: str,
    ) -> float:
        """L2 norm of ∂V/∂(vision_features) for sensitivity diagnostics.

        Default: raises ``NotImplementedError``. Adapters implement when the
        critic exposes the right intermediate tensor to ``requires_grad_``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement value_gradient_magnitude. "
            f"The critic probe will skip gradient-based plots for this policy."
        )

    # ── Attention capture ────────────────────────────────────────────────────

    @abstractmethod
    def capture_attention(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,
        timestep: float = 0.5,
        layers: list[int] | None = None,
        requires_grad: bool = False,
        gt_actions: Tensor | None = None,
    ) -> AttentionCaptureResult:
        """
        Run a probe-time forward and return per-layer attention plus the
        sequence metadata the probe needs to label / overlay.

        Args:
            obs: observation tensors, batch dim 1.
            task_str: high-level task language string.
            state: ``[state_dim]`` current joint state, if available.
            timestep: diffusion / flow-matching timestep for the suffix noise
                (matches the single-timestep convention from pi05's probe).
            layers: which layer indices to capture; ``None`` means all layers.
            requires_grad: keep captured attention in the autograd graph (for
                future jacobian probes). Adapters may raise ``NotImplementedError``
                until they implement this path.
            gt_actions: optional raw GT action chunk, used by adapters whose
                gradient probe is defined relative to a flow/action target.
        """

    # ── GT normalisation (for plotting in normalised space) ──────────────────

    @abstractmethod
    def normalize_gt_actions(self, gt_actions: Tensor, state: Tensor | None) -> Tensor:
        """
        Return GT actions in the same normalised space as ``pred_norm``.

        Args:
            gt_actions: ``[chunk_size, action_dim]`` raw GT actions (dataset units).
            state: ``[state_dim]`` current joint state, used for anchor/delta
                action encodings.

        Returns:
            ``[chunk_size, action_dim]`` float32 CPU, in normalised model space.
        """

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Action chunk length the policy predicts per forward pass."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of a single action (after slicing model output)."""

    @property
    def policy(self) -> Any:
        return self._policy

    @property
    def preprocessor(self) -> Any:
        return self._preprocessor

    @property
    def postprocessor(self) -> Any:
        return self._postprocessor

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def cfg(self) -> Any:
        return self._cfg

    # ── Misc ─────────────────────────────────────────────────────────────────

    def suppress_logs(self, enabled: bool) -> None:
        """Silence policy debug logging during probing. Default: no-op."""


def _adapter_for_type(policy_type: str | None) -> type[ProbablePolicy]:
    if policy_type in ("pi05_rl", "pi05_full"):
        from lerobot.probes.adapters.pi05 import Pi05Adapter
        return Pi05Adapter
    if policy_type in ("molmoact2_rl", "molmoact2"):
        from lerobot.probes.adapters.molmoact2 import MolmoAct2Adapter
        return MolmoAct2Adapter
    raise ValueError(
        f"No probe adapter registered for policy type {policy_type!r}. "
        f"Add one in lerobot/probes/adapters/ and register it in _adapter_for_type()."
    )
