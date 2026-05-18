"""
Pi05 adapter for :class:`lerobot.probes.base.ProbablePolicy`.

Wraps a loaded pi05 policy + processors so probes can call a uniform API.
Mirrors the inference logic in
``lerobot.probes.offline_inference_pi05.run_inference`` (kept as a reference
implementation).
"""

from __future__ import annotations

import torch
from torch import Tensor

import logging
from contextlib import contextmanager

import numpy as np

from lerobot.policies.pi05_full.modeling_pi05 import _PROBING_CAPTURE, make_att_2d_masks, pad_vector
from lerobot.probes.base import AttentionCaptureResult, ProbablePolicy
from lerobot.probes.utils import find_normalizer_step
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)


@contextmanager
def _capture_pi05_repr(module):
    """Monkey-patch ``paligemma_with_expert.forward`` to capture its
    ``(prefix_out, suffix_out)`` outputs.

    register_forward_hook is not used: the call chain invokes ``.forward()``
    directly (not via ``__call__``), bypassing PyTorch hooks.
    """
    captured: dict[str, Tensor] = {}
    original_forward = module.forward

    def patched(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if result[0][0] is not None:
            captured["prefix_out"] = result[0][0].detach().float()
        if result[0][1] is not None:
            captured["suffix_out"] = result[0][1].detach().float()
        return result

    module.forward = patched
    try:
        yield captured
    finally:
        module.forward = original_forward


class Pi05Adapter(ProbablePolicy):

    @property
    def chunk_size(self) -> int:
        return int(self._cfg.policy.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self._cfg.policy.output_features[ACTION].shape[0])

    def suppress_logs(self, enabled: bool) -> None:
        self._policy.model.suppress_debug_log = enabled

    @torch.no_grad()
    def normalize_gt_actions(self, gt_actions: Tensor, state: Tensor | None) -> Tensor:
        action_encoding = getattr(self._cfg.policy, "action_encoding", "absolute")
        processed = gt_actions
        if state is not None and action_encoding in ("anchor", "delta"):
            anchor_val = state[:gt_actions.shape[-1]].unsqueeze(0).cpu()
            if action_encoding == "anchor":
                processed = gt_actions - anchor_val
            elif action_encoding == "delta":
                d_0 = gt_actions[0:1] - anchor_val
                if gt_actions.shape[0] > 1:
                    d_rest = torch.diff(gt_actions, dim=0)
                    processed = torch.cat([d_0, d_rest], dim=0)
                else:
                    processed = d_0

        norm_step = find_normalizer_step(self._preprocessor)
        batch = {TransitionKey.ACTION: processed.unsqueeze(0).to(self._device)}
        out = norm_step(batch)
        return out[TransitionKey.ACTION].squeeze(0).float().cpu()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,
        advantage: float = 1.0,
    ) -> tuple[Tensor, Tensor, str | None]:
        device = self._device
        action_dim = self.action_dim

        # Dummy action is only consumed by the preprocessor for FAST tokenisation —
        # not used for inference. Keep its shape (B=1, 1, 6) as in the reference.
        dummy_action = torch.zeros(1, 1, 6, device=device)
        complementary_data = {
            "task": [task_str],
            "subtask": [""],
            "advantage": torch.tensor([[advantage]], device=device),
        }

        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        batch_for_proc = {
            TransitionKey.ACTION: dummy_action,
            **obs_on_device,
            TransitionKey.COMPLEMENTARY_DATA: complementary_data,
        }
        processed = self._preprocessor(batch_for_proc)

        images, img_masks = self._policy._preprocess_images(obs_on_device)
        task_tokens = processed[OBS_LANGUAGE_TOKENS]
        task_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]

        subtask_tokens, subtask_masks = self._policy.model.generate_subtask_tokens(
            images, img_masks, task_tokens, task_masks
        )
        tokenizer = self._policy.model._paligemma_tokenizer
        valid = subtask_tokens[0][subtask_masks[0]]
        pred_subtask = tokenizer.decode(valid, skip_special_tokens=True).strip()

        pred_actions = self._policy.model.sample_actions(
            images, img_masks, task_tokens, task_masks, subtask_tokens, subtask_masks
        )
        pred_actions = pred_actions[:, :, :action_dim].squeeze(0)  # [chunk_size, action_dim]
        pred_norm = pred_actions.float().cpu()

        unnorm = self._postprocessor(pred_actions.unsqueeze(0).float())
        pred_unnorm = unnorm.squeeze(0).float().cpu()

        action_encoding = getattr(self._cfg.policy, "action_encoding", "absolute")
        if state is not None and action_encoding in ("anchor", "delta"):
            anchor_val = state[:action_dim].unsqueeze(0).cpu()
            if action_encoding == "anchor":
                pred_unnorm = pred_unnorm + anchor_val
            elif action_encoding == "delta":
                pred_unnorm = torch.cumsum(pred_unnorm, dim=0) + anchor_val

        return pred_unnorm, pred_norm, pred_subtask

    def capture_attention(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,
        timestep: float = 0.5,
        layers: list[int] | None = None,
        requires_grad: bool = False,
    ) -> AttentionCaptureResult:
        # Top-level dispatch — visualization vs Jacobian (causal map) capture.
        if requires_grad:
            return self._capture_attention_jacobian(obs, task_str, timestep, layers)
        return self._capture_attention_viz(obs, task_str, timestep, layers)

    def _build_pi05_probe_inputs(self, obs, task_str, timestep):
        """Build prefix + suffix embeddings and segments for a probe forward.

        Shared by both visualisation and Jacobian capture paths.
        """
        device = self._device
        model = self._policy.model

        dummy_action = torch.zeros(1, 1, 6, device=device)
        complementary_data = {
            "task": [task_str], "subtask": [""],
            "advantage": torch.tensor([[1.0]], device=device),
        }
        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        processed = self._preprocessor({
            TransitionKey.ACTION: dummy_action,
            **obs_on_device,
            TransitionKey.COMPLEMENTARY_DATA: complementary_data,
        })

        images, img_masks = self._policy._preprocess_images(obs_on_device)
        task_tokens = processed[OBS_LANGUAGE_TOKENS]
        task_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK]

        subtask_tokens, subtask_masks = model.generate_subtask_tokens(
            images, img_masks, task_tokens, task_masks,
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks, image_len = model.embed_prefix(
            images, img_masks,
            task_tokens, subtask_tokens,
            task_masks, subtask_masks,
            fast_action_tokens=None, fast_action_masks=None,
        )
        w_dtype = (model.paligemma_with_expert.paligemma
                   .language_model.layers[0].self_attn.q_proj.weight.dtype)
        if w_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)

        n_cameras = len(images)
        patches_per_cam = image_len // n_cameras

        encoder_segments: list[tuple[str, int, int]] = []
        pos = 0
        for i in range(n_cameras):
            encoder_segments.append((f"img{i + 1}", pos, pos + patches_per_cam))
            pos += patches_per_cam
        encoder_segments.append(("language", pos, pos + task_tokens.shape[1]))
        pos += task_tokens.shape[1]
        encoder_segments.append(("subtask", pos, pos + subtask_tokens.shape[1]))

        bsize = task_tokens.shape[0]
        noise = model.sample_noise(
            (bsize, model.config.chunk_size, model.config.max_action_dim), device,
        )
        time_tensor = torch.full((bsize,), timestep, dtype=torch.float32, device=device)
        if w_dtype == torch.bfloat16:
            noise = noise.to(torch.bfloat16)
            time_tensor = time_tensor.to(torch.bfloat16)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
            noise, time_tensor,
        )
        if w_dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)

        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        total_len  = prefix_len + suffix_len
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        combined = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)
        combined[:, :prefix_len, :prefix_len] = prefix_att_masks
        combined[:, prefix_len:, prefix_len:] = suffix_att_2d
        combined[:, prefix_len:, :prefix_len] = True  # suffix sees all prefix
        combined_pad = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        pad_2d = combined_pad[:, None, :] & combined_pad[:, :, None]
        att_2d = combined & pad_2d
        position_ids = torch.cumsum(combined_pad, dim=1) - 1
        att_2d_4d = model._prepare_attention_masks_4d(att_2d)

        return {
            "model": model,
            "prefix_embs": prefix_embs, "suffix_embs": suffix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "att_2d_4d": att_2d_4d, "position_ids": position_ids,
            "adarms_cond": adarms_cond,
            "prefix_len": prefix_len, "suffix_len": suffix_len,
            "encoder_segments": encoder_segments,
            "patches_per_cam": patches_per_cam,
            "task_tokens": task_tokens, "subtask_tokens": subtask_tokens,
            "images": images,
        }

    @torch.no_grad()
    def _capture_attention_viz(self, obs, task_str, timestep, layers):
        c = self._build_pi05_probe_inputs(obs, task_str, timestep)
        model = c["model"]

        _PROBING_CAPTURE["enabled"] = True
        _PROBING_CAPTURE["all_layers"] = True
        _PROBING_CAPTURE["requires_grad"] = False
        _PROBING_CAPTURE["attn_weights_by_layer"] = {}
        try:
            model.paligemma_with_expert.forward(
                attention_mask=c["att_2d_4d"],
                position_ids=c["position_ids"],
                past_key_values=None,
                inputs_embeds=[c["prefix_embs"], c["suffix_embs"]],
                use_cache=False,
                adarms_cond=[None, c["adarms_cond"]],
            )
        finally:
            _PROBING_CAPTURE["enabled"] = False
            _PROBING_CAPTURE["all_layers"] = False

        attn_by_layer = _PROBING_CAPTURE.get("attn_weights_by_layer", {})
        return self._pack_pi05_result(attn_by_layer, c, layers)

    def _capture_attention_jacobian(self, obs, task_str, timestep, layers):
        """Per-layer forward+backward, returns causal maps ``A * |dA|``.

        Mirrors the reference :func:`jacobian_probe_forward_multilayer` —
        each layer runs an independent forward+backward to avoid keeping all
        layers' attention in the graph simultaneously.
        """
        c = self._build_pi05_probe_inputs(obs, task_str, timestep)
        model = c["model"]
        chunk_size = int(model.config.chunk_size)

        # Default to all_layers when caller doesn't restrict — match viz path.
        n_layers = len(model.paligemma_with_expert.paligemma.language_model.layers)
        target_layers = list(layers) if layers else list(range(n_layers))

        causal_by_layer: dict[int, Tensor] = {}

        for layer_idx in target_layers:
            # Fresh detached inputs per iteration — each backward() frees its
            # graph, so we can't reuse the same tensors across layers.
            prefix_iter = c["prefix_embs"].detach()
            suffix_iter = c["suffix_embs"].detach()
            adarms_iter = c["adarms_cond"].detach() if c["adarms_cond"] is not None else None

            _PROBING_CAPTURE["enabled"] = True
            _PROBING_CAPTURE["all_layers"] = False
            _PROBING_CAPTURE["requires_grad"] = True
            _PROBING_CAPTURE["target_layer"] = layer_idx
            _PROBING_CAPTURE["attn_weights"] = None

            try:
                with torch.set_grad_enabled(True):
                    (prefix_out, suffix_out), _ = model.paligemma_with_expert.forward(
                        attention_mask=c["att_2d_4d"],
                        position_ids=c["position_ids"],
                        past_key_values=None,
                        inputs_embeds=[prefix_iter, suffix_iter],
                        use_cache=False,
                        adarms_cond=[None, adarms_iter],
                    )
                    attn_weights = _PROBING_CAPTURE["attn_weights"]
                    if attn_weights is None:
                        logging.warning(
                            f"[pi05/jacobian] layer {layer_idx}: no attention captured"
                        )
                        continue
                    action_hidden = suffix_out[:, -chunk_size:]
                    action_hidden = action_hidden.to(dtype=model.action_out_proj.weight.dtype)
                    action_pred = model.action_out_proj(action_hidden)
                    loss = torch.norm(action_pred, p=2)
                    loss.backward()

                if attn_weights.grad is not None:
                    causal = (attn_weights.detach() * attn_weights.grad.abs()).float().cpu()
                else:
                    logging.warning(
                        f"[pi05/jacobian] layer {layer_idx}: grad is None, falling back to raw attention"
                    )
                    causal = attn_weights.detach().float().cpu()
                causal_by_layer[layer_idx] = causal
            finally:
                _PROBING_CAPTURE["enabled"] = False
                _PROBING_CAPTURE["requires_grad"] = False
                _PROBING_CAPTURE.pop("target_layer", None)
                _PROBING_CAPTURE["attn_weights"] = None
                self._policy.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        return self._pack_pi05_result(causal_by_layer, c, target_layers)

    def _pack_pi05_result(self, attn_by_layer, c, layers):
        """Slice ``[B,H,total,total]`` attention into cross + self and wrap."""
        prefix_len = c["prefix_len"]
        cross_attn: dict[int, Tensor] = {}
        self_attn:  dict[int, Tensor] = {}
        wanted = set(layers) if layers is not None else set(attn_by_layer.keys())
        for layer_idx, attn in attn_by_layer.items():
            if layer_idx not in wanted:
                continue
            cross_attn[layer_idx] = attn[:, :, prefix_len:, :prefix_len]
            self_attn[layer_idx]  = attn[:, :, prefix_len:, prefix_len:]

        return AttentionCaptureResult(
            cross_attn_by_layer=cross_attn,
            self_attn_by_layer=self_attn,
            encoder_segments=c["encoder_segments"],
            encoder_pad_masks=c["prefix_pad_masks"],
            image_tensors=list(c["images"]),
            patches_per_cam=c["patches_per_cam"],
            task_tokens=c["task_tokens"],
            subtask_tokens=c["subtask_tokens"],
            tokenizer=c["model"]._paligemma_tokenizer,
        )

    # ── Critic / value head ──────────────────────────────────────────────────

    def _critic_forward(self, obs: dict[str, Tensor], task_str: str,
                        with_grad: bool = False) -> tuple[dict, Tensor | None, Tensor | None]:
        """Run policy.critic and return (out, vision_features, critic_text_embs).

        When ``with_grad`` is True, ``vision_features`` and ``critic_text_embs``
        are leaf tensors with ``requires_grad_``, so the caller can backprop.
        """
        device = self._device
        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        complementary_data = {
            "task": [task_str], "subtask": [""],
            "advantage": torch.tensor([[1.0]], device=device),
        }
        processed = self._preprocessor({
            TransitionKey.ACTION: torch.zeros(1, 1, 6, device=device),
            **obs_on_device,
            TransitionKey.COMPLEMENTARY_DATA: complementary_data,
        })

        actor_tokens = processed[OBS_LANGUAGE_TOKENS]
        actor_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK]
        critic_tokens      = processed.get("critic_tokens", actor_tokens)
        critic_token_masks = processed.get("critic_pad_mask", actor_masks)

        embed_layer = self._policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
        critic_text_embs = embed_layer(critic_tokens).detach()

        images, img_masks = self._policy._preprocess_images(obs_on_device)
        encoder = self._policy.critic
        vision_features = torch.cat(
            [encoder.embed_image(img) for img in images], dim=1,
        )

        if with_grad:
            vision_features = vision_features.detach().requires_grad_(True)
            critic_text_embs = critic_text_embs.detach().requires_grad_(True)

        out = self._policy.critic(vision_features, critic_text_embs, critic_token_masks)
        return out, vision_features, critic_text_embs

    @torch.no_grad()
    def predict_value(self, obs: dict[str, Tensor], task_str: str) -> float:
        out, _, _ = self._critic_forward(obs, task_str)
        return float(out["value"].item())

    @torch.no_grad()
    def predict_value_and_probs(
        self, obs: dict[str, Tensor], task_str: str,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        out, _, _ = self._critic_forward(obs, task_str)
        v = float(out["value"].item())
        probs = out["probs"].squeeze(0).float().cpu().numpy()
        bin_centers = self._policy.critic.bin_centers.detach().float().cpu().numpy()
        return v, probs, bin_centers

    def value_gradient_magnitude(self, obs: dict[str, Tensor], task_str: str) -> float:
        out, vision_features, _ = self._critic_forward(obs, task_str, with_grad=True)
        self._policy.critic.zero_grad()
        out["value"].sum().backward()
        return float(vision_features.grad.norm().item())

    # ── Representations ──────────────────────────────────────────────────────

    @torch.no_grad()
    def capture_representations(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,    # noqa: ARG002 — unused for pi05
        timestep: float = 1.0,
        gt_actions: Tensor | None = None,
        gt_subtask: str | None = None,
    ) -> dict[str, Tensor]:
        device = self._device
        if gt_actions is None:
            raise ValueError("Pi05Adapter.capture_representations needs gt_actions.")

        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        complementary_data = {
            "task":      [task_str],
            "subtask":   [gt_subtask or ""],
            "advantage": torch.tensor([[1.0]], device=device),
        }
        processed = self._preprocessor({
            TransitionKey.ACTION: gt_actions.unsqueeze(0).to(device),
            **obs_on_device,
            TransitionKey.COMPLEMENTARY_DATA: complementary_data,
        })

        images, img_masks = self._policy._preprocess_images(obs_on_device)
        task_tokens    = processed[OBS_LANGUAGE_TOKENS].to(device)
        task_masks     = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)
        subtask_tokens = processed[OBS_LANGUAGE_SUBTASK_TOKENS].to(device)
        subtask_masks  = processed[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK].to(device)
        action_tokens  = processed[ACTION_TOKENS].to(device)
        action_masks   = processed[ACTION_TOKEN_MASK].to(device)
        actions_padded = pad_vector(
            processed[ACTION].to(device), self._policy.config.max_action_dim,
        )

        noise = torch.randn_like(actions_padded)
        time_tensor = torch.full(
            (actions_padded.shape[0],), float(timestep),
            device=device, dtype=actions_padded.dtype,
        )

        chunk_size = int(self._cfg.policy.chunk_size)

        # paligemma_with_expert.forward returns ((prefix_out, suffix_out), ...).
        # register_forward_hook doesn't work here because the call chain uses
        # .forward() directly, bypassing PyTorch hooks — mirror the reference's
        # monkey-patch approach.
        with _capture_pi05_repr(self._policy.model.paligemma_with_expert) as captured:
            self._policy.model.forward(
                images, img_masks,
                task_tokens, task_masks,
                subtask_tokens, subtask_masks,
                action_tokens, action_masks,
                actions_padded,
                noise=noise,
                time=time_tensor,
            )

        out: dict[str, Tensor] = {}
        if "prefix_out" in captured:
            out["prefix"] = captured["prefix_out"].mean(dim=1).squeeze(0).cpu()
        if "suffix_out" in captured:
            suffix = captured["suffix_out"][:, -chunk_size:, :]
            out["suffix"] = suffix.mean(dim=1).squeeze(0).cpu()
        return out
