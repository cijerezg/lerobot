"""
MolmoAct2 adapter for :class:`lerobot.probes.base.ProbablePolicy`.

Wraps a loaded molmoact2 policy + processors so probes can call a uniform API.

Unlike pi05, the molmoact2 policy exposes a top-level
``MolmoAct2Policy.predict_action_chunk(batch)`` that internally handles
autocast, action-mode dispatch, slicing to the configured action dim, and
float32 conversion. The adapter just builds the preprocessor input, calls
predict, and unnormalises the result.

MolmoAct2 does not generate subtasks, so ``pred_subtask`` is always ``None``.
It also only supports absolute action encoding, so ``state`` is unused.
"""

from __future__ import annotations

import torch
from torch import Tensor

import numpy as np

from lerobot.policies.molmoact2.modeling_molmoact2 import (
    _MOLMOACT2_PROBING_CAPTURE,
    register_action_attention_probing,
)
from lerobot.probes.base import AttentionCaptureResult, ProbablePolicy
from lerobot.probes.utils import find_normalizer_step
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


class MolmoAct2Adapter(ProbablePolicy):

    @property
    def chunk_size(self) -> int:
        return int(self._cfg.policy.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self._cfg.policy.output_features[ACTION].shape[0])

    def _make_batch(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        advantage: float = 1.0,
        gt_actions: Tensor | None = None,
    ) -> dict:
        """Build the preprocessor input for molmoact2 probe forwards.

        Passes ``advantage`` via ``TransitionKey.COMPLEMENTARY_DATA`` so the
        :class:`MolmoAct2PackInputsProcessorStep` picks it up and binds the
        "negative"/"positive" advantage clause into the prompt — same path the
        trainer uses.
        """
        device = self._device
        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        flat = {
            **obs_on_device,
            "task": task_str,
            TransitionKey.COMPLEMENTARY_DATA: {
                "advantage": torch.tensor([[advantage]], device=device, dtype=torch.float32),
            },
        }
        if gt_actions is not None:
            flat[ACTION] = gt_actions
        batch = self._preprocessor(flat)
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    @torch.no_grad()
    def normalize_gt_actions(self, gt_actions: Tensor, state: Tensor | None) -> Tensor:
        # Molmoact2 is absolute-only; no anchor/delta adjustment needed.
        # The masked normalizer uses an internal mask (from _tensor_stats), so it
        # can be invoked in isolation without action_dim_is_pad.
        norm_step = find_normalizer_step(self._preprocessor)
        batch = {TransitionKey.ACTION: gt_actions.unsqueeze(0).to(self._device)}
        out = norm_step(batch)
        return out[TransitionKey.ACTION].squeeze(0).float().cpu()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,  # unused: molmoact2 is absolute-only
        advantage: float = 1.0,
    ) -> tuple[Tensor, Tensor, str | None]:
        batch = self._make_batch(obs, task_str, advantage=advantage)
        # MolmoAct2Policy.predict_action_chunk returns [B, n_action_steps, action_dim],
        # already sliced and float32. See modeling_molmoact2.py:2004.
        norm_actions = self._policy.predict_action_chunk(batch)
        pred_norm = norm_actions.squeeze(0).float().cpu()
        unnorm = self._postprocessor(norm_actions.float())
        pred_unnorm = unnorm.squeeze(0).float().cpu()
        return pred_unnorm, pred_norm, None

    def capture_attention(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,    # noqa: ARG002 — molmoact2 absolute-only
        timestep: float = 0.5,
        layers: list[int] | None = None,
        requires_grad: bool = False,
    ) -> AttentionCaptureResult:
        # Register the action-expert hooks once per adapter. The hooks stay
        # installed (no-ops when the global flag is off), so this is safe.
        if not getattr(self, "_attn_hooks_registered", False):
            register_action_attention_probing(self._policy._action_expert())
            self._attn_hooks_registered = True

        if requires_grad:
            return self._capture_attention_jacobian(obs, task_str, timestep, layers)
        return self._capture_attention_viz(obs, task_str, layers)

    @torch.no_grad()
    def _capture_attention_viz(self, obs, task_str, layers):
        obs_on_device = {k: v.to(self._device) for k, v in obs.items()}
        batch = self._make_batch(obs, task_str, advantage=1.0)

        _MOLMOACT2_PROBING_CAPTURE.clear()
        _MOLMOACT2_PROBING_CAPTURE["enabled"] = True
        try:
            # NOTE: predict_action_chunk runs the full flow-matching loop.
            # Captured attention reflects the LAST step's activations.
            # Single-timestep capture (matching pi05) is a future improvement;
            # the `timestep` arg is accepted but currently ignored.
            self._policy.predict_action_chunk(batch)
        finally:
            _MOLMOACT2_PROBING_CAPTURE["enabled"] = False

        cross_raw = _MOLMOACT2_PROBING_CAPTURE.get("cross_attn_by_layer", {})
        self_raw  = _MOLMOACT2_PROBING_CAPTURE.get("self_attn_by_layer", {})

        wanted = set(layers) if layers is not None else set(cross_raw.keys()) | set(self_raw.keys())
        cross_attn = {k: v for k, v in cross_raw.items() if k in wanted}
        self_attn  = {k: v for k, v in self_raw.items()  if k in wanted}
        return self._pack_molmoact2_result(cross_attn, self_attn, batch, obs_on_device)

    def _capture_attention_jacobian(self, obs, task_str, timestep, layers):
        """Per-layer forward+backward through the training flow path, returns
        causal maps ``A * |dA|`` packed into ``cross_attn_by_layer`` /
        ``self_attn_by_layer``.

        Uses ``MolmoAct2Policy._compute_flow_matching_loss_joint_per_layer``
        (grad-enabled) instead of ``predict_action_chunk`` (no_grad). The flow
        loss is L2 on (pred_velocity - target_velocity); backprop populates
        ``.grad`` on captured weights. Per-layer iteration prevents OOM by only
        routing the target layer through the grad-aware patched _attention; all
        other layers go through SDPA.
        """
        device = self._device
        chunk_size = self.chunk_size
        action_dim = self.action_dim

        # Dummy GT actions — the flow loss requires them but their values don't
        # matter for Jacobian shape; grads still flow through captured weights.
        dummy_actions = torch.zeros(1, chunk_size, action_dim, device=device)

        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        batch = self._make_batch(obs, task_str, advantage=1.0, gt_actions=dummy_actions)
        model_inputs = self._policy._model_inputs(batch)

        num_t = max(1, int(getattr(self._policy.config, "num_flow_timesteps", 1)))
        action_dtype = next(self._policy._action_expert().parameters()).dtype
        timesteps_tensor = torch.full(
            (1, num_t), float(timestep), device=device, dtype=action_dtype,
        )

        action_expert = self._policy._action_expert()
        n_layers = len(action_expert.blocks)
        target_layers = list(layers) if layers else list(range(n_layers))

        # Build empty result containers; fill per-layer.
        causal_cross: dict[int, Tensor] = {}
        causal_self:  dict[int, Tensor] = {}

        for layer_idx in target_layers:
            _MOLMOACT2_PROBING_CAPTURE.clear()
            _MOLMOACT2_PROBING_CAPTURE["enabled"] = True
            _MOLMOACT2_PROBING_CAPTURE["requires_grad"] = True
            _MOLMOACT2_PROBING_CAPTURE["target_layer"] = layer_idx

            try:
                with torch.set_grad_enabled(True):
                    flow_loss, _ = self._policy._compute_flow_matching_loss_joint_per_layer(
                        batch=batch,
                        model_inputs=model_inputs,
                        timesteps=timesteps_tensor,
                        reduction="mean",
                    )
                    flow_loss.backward()

                bucket_cross = _MOLMOACT2_PROBING_CAPTURE.get("cross_attn_by_layer", {})
                bucket_self  = _MOLMOACT2_PROBING_CAPTURE.get("self_attn_by_layer", {})

                cw = bucket_cross.get(layer_idx)
                sw = bucket_self.get(layer_idx)
                if cw is not None and cw.grad is not None:
                    causal_cross[layer_idx] = (cw.detach() * cw.grad.abs()).float().cpu()
                elif cw is not None:
                    causal_cross[layer_idx] = cw.detach().float().cpu()
                if sw is not None and sw.grad is not None:
                    causal_self[layer_idx] = (sw.detach() * sw.grad.abs()).float().cpu()
                elif sw is not None:
                    causal_self[layer_idx] = sw.detach().float().cpu()
            finally:
                _MOLMOACT2_PROBING_CAPTURE.clear()
                self._policy.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        return self._pack_molmoact2_result(causal_cross, causal_self, batch, obs_on_device)

    def _pack_molmoact2_result(self, cross_attn, self_attn, batch, obs_on_device):
        """Wrap captured attention dicts in an AttentionCaptureResult."""
        encoder_seq_len = 0
        if cross_attn:
            encoder_seq_len = int(next(iter(cross_attn.values())).shape[-1])
        # Encoder-side segmentation: v1 emits a single ("encoder", 0, S) span.
        # Refining into (img1, img2, ..., language, state) blocks requires
        # mapping input_ids → hidden-state positions through the HF processor.
        encoder_segments: list[tuple[str, int, int]] = [("encoder", 0, encoder_seq_len)]
        image_tensors = [v for k, v in obs_on_device.items() if "images" in k]

        return AttentionCaptureResult(
            cross_attn_by_layer=cross_attn,
            self_attn_by_layer=self_attn,
            encoder_segments=encoder_segments,
            encoder_pad_masks=batch.get("attention_mask"),
            image_tensors=image_tensors,
            patches_per_cam=0,
            task_tokens=batch.get("input_ids"),
            subtask_tokens=None,
            tokenizer=getattr(self._policy, "tokenizer", None) or getattr(
                self._preprocessor, "tokenizer", None
            ),
            extras={"_capture_caveat": "viz path: last flow-matching step"},
        )

    # ── Critic / value head ──────────────────────────────────────────────────

    def _critic_batch(self, obs: dict[str, Tensor], task_str: str) -> dict:
        return self._make_batch(obs, task_str, advantage=1.0)

    @torch.no_grad()
    def predict_value(self, obs: dict[str, Tensor], task_str: str) -> float:
        out = self._policy.forward_critic(self._critic_batch(obs, task_str))
        return float(out["value"].mean().item())

    @torch.no_grad()
    def predict_value_and_probs(
        self, obs: dict[str, Tensor], task_str: str,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        out = self._policy.forward_critic(self._critic_batch(obs, task_str))
        v = float(out["value"].mean().item())
        probs = out["probs"].squeeze(0).float().cpu().numpy()
        bin_centers = self._policy.critic.bin_centers.detach().float().cpu().numpy()
        return v, probs, bin_centers

    # NOTE: value_gradient_magnitude is deferred. The pi05 version puts
    # requires_grad_ on vision_features extracted via policy.critic.embed_image(),
    # which has no equivalent in molmoact2's forward_critic path (the encoder
    # forward is opaque from outside). Implementing it requires plumbing
    # requires_grad through MolmoAct2RLPolicy._forward_critic_impl onto
    # inputs_embeds. The base class raises NotImplementedError, and the probe
    # skips gradient-based plots when this isn't supported.

    # ── Representations ──────────────────────────────────────────────────────

    @torch.no_grad()
    def capture_representations(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,        # noqa: ARG002 — absolute-only
        timestep: float = 1.0,               # noqa: ARG002 — captured from last flow step
        gt_actions: Tensor | None = None,    # noqa: ARG002 — molmoact2 doesn't need GT actions
        gt_subtask: str | None = None,       # noqa: ARG002 — no subtask path
    ) -> dict[str, Tensor]:
        backbone = self._policy._backbone()
        transformer = backbone.transformer
        action_expert = backbone._require_action_expert()

        captured: dict[str, Tensor] = {}

        def _hook(site_name: str):
            def fn(_module, _inputs, output):
                # decoder_block returns a tuple (hidden, ...); action_block returns a tensor.
                hidden = output[0] if isinstance(output, tuple) else output
                captured[site_name] = hidden.detach().float()
            return fn

        h_enc = transformer.blocks[-1].register_forward_hook(_hook("encoder"))
        h_act = action_expert.blocks[-1].register_forward_hook(_hook("action_expert"))

        try:
            batch = self._make_batch(obs, task_str, advantage=1.0)
            # NOTE: predict_action_chunk runs the full flow-matching loop.
            # Captured hidden states reflect the LAST step. Same caveat as
            # capture_attention.
            self._policy.predict_action_chunk(batch)
        finally:
            h_enc.remove()
            h_act.remove()

        out: dict[str, Tensor] = {}
        for site, tensor in captured.items():
            # tensor: [B, seq, hidden_dim] — mean over seq, squeeze batch.
            out[site] = tensor.mean(dim=1).squeeze(0).cpu()
        return out
