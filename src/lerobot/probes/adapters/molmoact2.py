"""
MolmoAct2 adapter for :class:`lerobot.probes.base.ProbablePolicy`.

Wraps a loaded molmoact2 policy + processors so probes can call a uniform API.

Unlike pi05, the molmoact2 policy exposes a top-level
``MolmoAct2Policy.predict_action_chunk(batch)`` that internally handles
autocast, action-mode dispatch, slicing to the configured action dim, and
float32 conversion. The adapter just builds the preprocessor input, calls
predict, and unnormalises the result.

``pred_subtask`` from predict_action_chunk is always ``None``: the policy no longer
generates one. The subtask reaches the model as a prompt clause the caller supplies,
so probes measure it by varying that input, not by reading a decode. Only absolute
action encoding needs no ``state``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    _MOLMOACT2_PROBING_CAPTURE,
    register_action_attention_probing,
)
from lerobot.probes.base import ActionSensitivityResult, AttentionCaptureResult, ProbablePolicy
from lerobot.probes.utils import find_normalizer_step, suppress_pack_dropout
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


class MolmoAct2Adapter(ProbablePolicy):

    @property
    def chunk_size(self) -> int:
        return int(self._cfg.policy.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self._cfg.policy.output_features[ACTION].shape[0])

    def _inference_action_mode(self) -> str:
        requested = getattr(self._cfg.policy, "inference_action_mode", None)
        if requested in {"continuous", "discrete"}:
            return str(requested)

        action_mode = getattr(self._cfg.policy, "action_mode", None)
        if action_mode in {"continuous", "discrete"}:
            return str(action_mode)

        training_mode_fn = getattr(self._policy, "_training_action_mode", None)
        training_mode = training_mode_fn() if callable(training_mode_fn) else None
        if training_mode in {"continuous", "discrete"}:
            return str(training_mode)

        raise ValueError(
            "MolmoAct2 probes need an explicit inference action mode when action_mode=both. "
            "Set policy.inference_action_mode to either continuous or discrete."
        )

    def _set_probe_cuda_graph_enabled(self, enabled: bool) -> None:
        set_enabled = getattr(self._policy, "_set_inference_cuda_graph_enabled", None)
        if callable(set_enabled):
            set_enabled(bool(enabled))

    def _restore_probe_cuda_graph_enabled(self) -> None:
        self._set_probe_cuda_graph_enabled(not bool(getattr(self._policy, "training", False)))

    def _make_batch(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        gt_actions: Tensor | None = None,
        subtask: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Build the preprocessor input for molmoact2 probe forwards."""
        device = self._device
        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        flat = {
            **obs_on_device,
            "task": task_str,
        }
        complementary: dict = {}
        if subtask:
            complementary["subtask"] = [subtask]
        if metadata is not None:
            complementary["metadata"] = metadata
        if complementary:
            flat[TransitionKey.COMPLEMENTARY_DATA] = complementary
        if gt_actions is not None:
            flat[ACTION] = gt_actions
        batch = self._preprocessor(flat)
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    @torch.no_grad()
    def normalize_gt_actions(self, gt_actions: Tensor, state: Tensor | None) -> Tensor:
        # Mirror the preprocessor pipeline: (anchor encode) → normalizer.
        actions = gt_actions
        action_encoding = getattr(self._cfg.policy, "action_encoding", "absolute")
        if state is not None and action_encoding in ("anchor", "delta"):
            anchor = state[: actions.shape[-1]].unsqueeze(0).cpu()
            if action_encoding == "anchor":
                actions = actions - anchor
            elif actions.shape[0] > 1:
                actions = torch.cat([actions[0:1] - anchor, torch.diff(actions, dim=0)], dim=0)
            else:
                actions = actions - anchor

        norm_step = find_normalizer_step(self._preprocessor)
        batch = {TransitionKey.ACTION: actions.unsqueeze(0).to(self._device)}
        out = norm_step(batch)
        return out[TransitionKey.ACTION].squeeze(0).float().cpu()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        state: Tensor | None = None,  # noqa: ARG002 — anchor comes from obs[OBS_STATE]
        advantage: float | None = None,  # noqa: ARG002 — molmoact2 prompts carry no advantage clause
        subtask: str | None = None,
        metadata: dict | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, str | None]:
        return self.predict_action_chunk_in_mode(
            obs,
            task_str,
            inference_action_mode=self._inference_action_mode(),
            state=state,
            advantage=advantage,
            subtask=subtask,
            metadata=metadata,
            generator=generator,
        )

    @torch.no_grad()
    def predict_action_chunk_in_mode(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        *,
        inference_action_mode: str,
        state: Tensor | None = None,  # noqa: ARG002 — anchor comes from obs[OBS_STATE]
        advantage: float | None = None,  # noqa: ARG002 — molmoact2 prompts carry no advantage clause
        subtask: str | None = None,
        metadata: dict | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, str | None]:
        """Predict through one explicitly selected MolmoAct2 action decoder."""
        if inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError(
                "inference_action_mode must be either 'continuous' or 'discrete', "
                f"got {inference_action_mode!r}."
            )
        batch = self._make_batch(obs, task_str, subtask=subtask, metadata=metadata)
        # MolmoAct2Policy.predict_action_chunk returns [B, n_action_steps, action_dim],
        # already sliced and float32. See modeling_molmoact2.py:2004.
        rtc_config = getattr(self._policy.config, "rtc_config", None)
        restore_rtc = bool(
            inference_action_mode == "discrete"
            and rtc_config is not None
            and bool(getattr(rtc_config, "enabled", False))
        )
        if restore_rtc:
            rtc_config.enabled = False
        try:
            norm_actions = self._policy.predict_action_chunk(
                batch,
                inference_action_mode=inference_action_mode,
                generator=generator,
            ).float()
        finally:
            if restore_rtc:
                rtc_config.enabled = True
        pred_norm = norm_actions.squeeze(0).float().cpu()
        action_encoding = getattr(self._cfg.policy, "action_encoding", "absolute")
        if action_encoding in ("anchor", "delta"):
            anchor = obs[OBS_STATE].to(self._device)[..., : self.action_dim]
            unnorm = self._postprocessor({ACTION: norm_actions, ANCHOR_KEY: anchor})
        else:
            unnorm = self._postprocessor(norm_actions)
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
        gt_actions: Tensor | None = None,
        subtask: str | None = None,
        metadata: dict | None = None,
    ) -> AttentionCaptureResult:
        # Register the action-expert hooks once per adapter. The hooks stay
        # installed (no-ops when the global flag is off), so this is safe.
        if not getattr(self, "_attn_hooks_registered", False):
            register_action_attention_probing(self._policy._action_expert())
            self._attn_hooks_registered = True

        if requires_grad:
            return self._capture_attention_jacobian(
                obs, task_str, timestep, layers, gt_actions, subtask, metadata
            )
        return self._capture_attention_viz(obs, task_str, timestep, layers, subtask, metadata)

    def capture_action_sensitivity(
        self,
        obs: dict[str, Tensor],
        task_str: str,
        *,
        gt_actions: Tensor,
        action_groups: dict[str, list[int]],
        timestep: float = 0.5,
        num_projections: int = 4,
        seed: int = 0,
        subtask: str | None = None,
        metadata: dict | None = None,
    ) -> ActionSensitivityResult:
        """Grouped ``d(flow velocity) / d(conditioning token)`` energy.

        The conditioning boundary is the real multimodal embedding sequence for
        this validation frame.  MolmoAct2's training graph normally stop-grads
        the encoder/action-expert handoff (knowledge insulation); probe capture
        preserves identical forward values but opens that handoff so autograd can
        measure the mathematical forward dependence.  Hutchinson VJPs estimate
        each raw Frobenius block norm without materialising a giant Jacobian.
        """
        if self._inference_action_mode() != "continuous":
            raise NotImplementedError(
                "Action sensitivity requires MolmoAct2's continuous flow action path."
            )
        if num_projections < 1:
            raise ValueError(f"num_projections must be >= 1, got {num_projections}.")

        obs_on_device, batch, model_inputs, timesteps_tensor = self._flow_probe_inputs(
            obs, task_str, timestep, gt_actions, subtask, metadata,
        )
        action_targets = batch[ACTION]
        num_flow_samples = int(timesteps_tensor.shape[1])
        noise_shape = (
            int(action_targets.shape[0]),
            num_flow_samples,
            int(action_targets.shape[1]),
            int(action_targets.shape[2]),
        )
        noise_generator = torch.Generator(device=self._device)
        noise_generator.manual_seed(int(seed))
        noise = torch.randn(
            noise_shape,
            device=self._device,
            dtype=next(self._policy._action_expert().parameters()).dtype,
            generator=noise_generator,
        )

        _MOLMOACT2_PROBING_CAPTURE.clear()
        _MOLMOACT2_PROBING_CAPTURE["capture_action_sensitivity"] = True
        self._set_probe_cuda_graph_enabled(False)
        scores_by_group: dict[str, Tensor] = {}
        token_count = 0
        try:
            with torch.set_grad_enabled(True):
                self._policy._compute_flow_matching_loss_joint_per_layer(
                    batch=batch,
                    model_inputs=model_inputs,
                    timesteps=timesteps_tensor,
                    noise=noise,
                    reduction="mean",
                )
                embeddings = _MOLMOACT2_PROBING_CAPTURE.get("conditioning_embeddings")
                pred_velocity = _MOLMOACT2_PROBING_CAPTURE.get("pred_velocity")
                if not torch.is_tensor(embeddings) or not torch.is_tensor(pred_velocity):
                    raise RuntimeError("MolmoAct2 sensitivity capture did not expose its graph tensors.")
                token_count = int(embeddings.shape[1])

                projection_generator = torch.Generator(device="cpu")
                projection_generator.manual_seed(int(seed) ^ 0x5EED5EED)
                group_items = list(action_groups.items())
                total_vjps = len(group_items) * int(num_projections)
                vjp_idx = 0
                for group_name, raw_indices in group_items:
                    indices = [int(idx) for idx in raw_indices]
                    if not indices or min(indices) < 0 or max(indices) >= self.action_dim:
                        raise ValueError(
                            f"Invalid action indices for {group_name!r}: {indices}; "
                            f"action_dim={self.action_dim}."
                        )
                    target = pred_velocity[..., indices]
                    score_sq = torch.zeros(token_count, dtype=torch.float32)
                    for _ in range(int(num_projections)):
                        signs_cpu = torch.empty(target.shape, dtype=torch.float32).bernoulli_(
                            0.5, generator=projection_generator
                        ).mul_(2.0).sub_(1.0)
                        signs = signs_cpu.to(device=target.device, dtype=target.dtype)
                        # Average over the model's Monte-Carlo flow samples so the
                        # reported value does not depend on num_flow_timesteps.
                        projection = (target * signs).sum() / (num_flow_samples ** 0.5)
                        vjp_idx += 1
                        grad = torch.autograd.grad(
                            projection,
                            embeddings,
                            retain_graph=vjp_idx < total_vjps,
                            create_graph=False,
                            allow_unused=False,
                        )[0]
                        score_sq += grad.detach().float().square().sum(dim=-1).squeeze(0).cpu()
                    scores_by_group[group_name] = (score_sq / float(num_projections)).sqrt()
        finally:
            _MOLMOACT2_PROBING_CAPTURE.clear()
            self._policy.zero_grad(set_to_none=True)
            self._restore_probe_cuda_graph_enabled()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Reuse the exact image-patch and decoded-prompt metadata used by the
        # attention suite.  A zero placeholder supplies only the encoder length;
        # it is never exposed as an attention map.
        placeholder = torch.zeros(1, 1, 1, token_count, dtype=torch.float32)
        token_metadata = self._pack_molmoact2_result(
            {0: placeholder}, {}, batch, obs_on_device
        )
        token_metadata.cross_attn_by_layer.clear()
        token_metadata.extras["_capture_caveat"] = (
            "action-output Jacobian at a teacher-forced flow state; conditioning is a real "
            "validation frame and the flow noise is fixed by the recorded seed"
        )
        return ActionSensitivityResult(
            scores_by_group=scores_by_group,
            token_metadata=token_metadata,
            action_groups={name: [int(i) for i in indices] for name, indices in action_groups.items()},
            timestep=float(timestep),
            num_flow_samples=num_flow_samples,
            num_projections=int(num_projections),
            extras={
                "seed": int(seed),
                "target": "flow_velocity",
                "conditioning_boundary": "multimodal_input_embeddings",
                "knowledge_insulation_opened_for_attribution": bool(
                    getattr(self._policy.config, "knowledge_insulation", False)
                ),
            },
        )

    def _flow_probe_inputs(
        self,
        obs,
        task_str,
        timestep,
        gt_actions: Tensor | None = None,
        subtask: str | None = None,
        metadata: dict | None = None,
    ):
        """Pack the batch + timesteps for a flow-loss capture.

        The batch carries an ACTION (the flow target), which is what flips
        ``build_action_labels`` in the pack step and arms subtask/metadata/summary/
        history/RGB dropout. `suppress_pack_dropout` holds them at zero so the
        capture sees exactly the prompt and the cameras the caller asked for.
        """
        device = self._device
        chunk_size = self.chunk_size
        action_dim = self.action_dim

        if gt_actions is None:
            action_targets = torch.zeros(1, chunk_size, action_dim, device=device)
        else:
            action_targets = gt_actions[:chunk_size, :action_dim].unsqueeze(0).to(device)

        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        with suppress_pack_dropout(self._preprocessor):
            batch = self._make_batch(
                obs, task_str, gt_actions=action_targets, subtask=subtask, metadata=metadata
            )
        model_inputs = self._policy._model_inputs(batch)
        num_t = max(1, int(getattr(self._policy.config, "num_flow_timesteps", 1)))
        action_dtype = next(self._policy._action_expert().parameters()).dtype
        timesteps_tensor = torch.full(
            (1, num_t), float(timestep), device=device, dtype=action_dtype,
        )
        return obs_on_device, batch, model_inputs, timesteps_tensor

    @torch.no_grad()
    def _capture_attention_viz(self, obs, task_str, timestep, layers, subtask=None, metadata=None):
        obs_on_device, batch, model_inputs, timesteps_tensor = self._flow_probe_inputs(
            obs, task_str, timestep, None, subtask, metadata,
        )
        _MOLMOACT2_PROBING_CAPTURE.clear()
        _MOLMOACT2_PROBING_CAPTURE["enabled"] = True
        _MOLMOACT2_PROBING_CAPTURE["requires_grad"] = False
        self._set_probe_cuda_graph_enabled(False)
        try:
            self._policy._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                timesteps=timesteps_tensor,
                reduction="mean",
            )
        finally:
            _MOLMOACT2_PROBING_CAPTURE["enabled"] = False
            _MOLMOACT2_PROBING_CAPTURE["requires_grad"] = False
            self._restore_probe_cuda_graph_enabled()

        cross_raw = _MOLMOACT2_PROBING_CAPTURE.get("cross_attn_by_layer", {})
        self_raw  = _MOLMOACT2_PROBING_CAPTURE.get("self_attn_by_layer", {})

        wanted = set(layers) if layers is not None else set(cross_raw.keys()) | set(self_raw.keys())
        cross_attn = {k: v for k, v in cross_raw.items() if k in wanted}
        self_attn  = {k: v for k, v in self_raw.items()  if k in wanted}
        return self._pack_molmoact2_result(cross_attn, self_attn, batch, obs_on_device)

    def _capture_attention_jacobian(
        self, obs, task_str, timestep, layers, gt_actions=None, subtask=None, metadata=None
    ):
        """Per-layer forward+backward through the training flow path, returns
        causal maps ``A * |dA|`` packed into ``cross_attn_by_layer`` /
        ``self_attn_by_layer``.

        Uses ``MolmoAct2Policy._compute_flow_matching_loss_joint_per_layer``
        (grad-enabled) instead of ``predict_action_chunk`` (no_grad). The flow
        loss is L2 on (pred_velocity - target_velocity) using ``gt_actions`` when
        the caller supplies them; backprop populates ``.grad`` on captured
        weights. Per-layer iteration prevents OOM by only
        routing the target layer through the grad-aware patched _attention; all
        other layers go through SDPA.
        """
        obs_on_device, batch, model_inputs, timesteps_tensor = self._flow_probe_inputs(
            obs, task_str, timestep, gt_actions, subtask, metadata,
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

    def _configured_image_keys(self) -> list[str]:
        """Image key order used by the MolmoAct2 input packer."""
        for source in (
            getattr(self._cfg, "policy", None),
            getattr(self._policy, "config", None),
        ):
            configured = list(getattr(source, "image_keys", []) or [])
            if configured:
                return [str(key) for key in configured]

        for step in getattr(self._preprocessor, "steps", []) or []:
            configured = list(getattr(step, "image_keys", []) or [])
            if configured:
                return [str(key) for key in configured]
        return []

    def _image_keys_for_obs(self, obs: dict[str, Tensor]) -> list[str]:
        configured = self._configured_image_keys()
        if configured and all(key in obs for key in configured):
            return configured
        keys = [key for key in obs if str(key).startswith("observation.images.")]
        if not keys:
            keys = [key for key in obs if str(key).startswith("observation.image")]
        return sorted(keys)

    def _tokenizer(self):
        tokenizer = getattr(self._policy, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        tokenizer = getattr(self._preprocessor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        for step in getattr(self._preprocessor, "steps", []) or []:
            processor = getattr(step, "processor", None)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
        return None

    def _image_patch_token_id(self) -> int | None:
        for source in (
            getattr(getattr(self._policy, "model", None), "config", None),
            getattr(getattr(self._policy, "config", None), "hf_config", None),
            getattr(self._policy, "config", None),
        ):
            value = getattr(source, "image_patch_id", None) if source is not None else None
            if value is not None:
                return int(value)
        backbone = getattr(self._policy, "_backbone", None)
        if callable(backbone):
            value = getattr(getattr(backbone(), "config", None), "image_patch_id", None)
            if value is not None:
                return int(value)
        return None

    @staticmethod
    def _safe_cam_name(image_key: str, index: int) -> str:
        name = str(image_key).split(".")[-1] or f"cam{index + 1}"
        clean = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        return f"img_{clean or f'cam{index + 1}'}"

    @staticmethod
    def _model_view_images_from_pixel_values(batch: dict, num_crops: int | None = None) -> list[Tensor]:
        """Reconstruct Molmo's model-view RGB crops from flattened patches.

        ``pixel_values`` is flat across all crops from all camera images. Crop 0
        for each image is the resized/global view; later crops are the local
        high-resolution tiles used by Molmo's visual encoder.
        """
        pixel_values = batch.get("pixel_values")
        if not torch.is_tensor(pixel_values) or pixel_values.ndim != 3:
            return []

        if num_crops is None:
            n = int(pixel_values.shape[0])
        else:
            n = min(int(num_crops), int(pixel_values.shape[0]))
        if n <= 0:
            return []

        n_patches = int(pixel_values.shape[1])
        flat_patch = int(pixel_values.shape[2])
        patch_area = flat_patch // 3
        patch_size = int(patch_area ** 0.5)
        grid_size = int(n_patches ** 0.5)
        if flat_patch != patch_size * patch_size * 3 or n_patches != grid_size * grid_size:
            return []

        patches = pixel_values[:n].detach().float().reshape(
            n, grid_size, grid_size, patch_size, patch_size, 3,
        )
        images = patches.permute(0, 5, 1, 3, 2, 4).reshape(
            n, 3, grid_size * patch_size, grid_size * patch_size,
        )

        # The shared renderer expects image tensors in roughly [-1, 1]. Molmo's
        # processor may emit [0, 1], [0, 255], or already-normalized patch pixels.
        img_min = float(images.min().item()) if images.numel() else 0.0
        img_max = float(images.max().item()) if images.numel() else 1.0
        if img_min >= 0.0 and img_max <= 1.5:
            images = images * 2.0 - 1.0
        elif img_min >= 0.0 and img_max > 1.5:
            images = images / 255.0 * 2.0 - 1.0

        return [images[i : i + 1] for i in range(n)]

    def _image_attention_metadata(self, batch: dict, obs_on_device: dict[str, Tensor], encoder_seq_len: int):
        input_ids = batch.get("input_ids")
        image_grids = batch.get("image_grids")
        if not torch.is_tensor(input_ids) or not torch.is_tensor(image_grids):
            return [("encoder", 0, encoder_seq_len)], [], 0, {}

        patch_id = self._image_patch_token_id()
        if patch_id is None:
            return [("encoder", 0, encoder_seq_len)], [], 0, {}

        row = input_ids[0].detach().cpu()
        patch_positions = (row == int(patch_id)).nonzero(as_tuple=False).flatten().tolist()
        if not patch_positions:
            return [("encoder", 0, encoder_seq_len)], [], 0, {}

        grids = image_grids.detach().cpu()
        pooled_counts = (grids[:, :2].prod(dim=1) + grids[:, 2:].prod(dim=1)).to(torch.long).tolist()
        global_pooled_counts = grids[:, :2].prod(dim=1).to(torch.long).tolist()

        image_num_crops = batch.get("image_num_crops")
        if torch.is_tensor(image_num_crops):
            crop_counts = image_num_crops.detach().cpu().to(torch.long).tolist()
        else:
            crop_counts = [1] * len(pooled_counts)

        image_token_pooling = batch.get("image_token_pooling")
        pooling_rows = (
            image_token_pooling.detach().cpu().to(torch.long)
            if torch.is_tensor(image_token_pooling)
            else None
        )

        image_keys = self._image_keys_for_obs(obs_on_device)
        num_images = min(len(image_keys), len(pooled_counts), len(crop_counts))
        total_crops = sum(max(0, int(crop_counts[idx])) for idx in range(num_images))
        model_view_crops = self._model_view_images_from_pixel_values(batch, total_crops)

        pixel_values = batch.get("pixel_values")
        n_patches_per_crop = (
            int(pixel_values.shape[1])
            if torch.is_tensor(pixel_values) and pixel_values.ndim == 3
            else 0
        )
        patch_grid = int(n_patches_per_crop ** 0.5) if n_patches_per_crop > 0 else 0
        if patch_grid * patch_grid != n_patches_per_crop:
            patch_grid = 0

        encoder_segments: list[tuple[str, int, int]] = []
        image_tensors: list[Tensor] = []
        patch_indices_by_segment: dict[str, list[int]] = {}
        patch_counts_by_segment: dict[str, int] = {}
        pooling_by_segment: dict[str, dict] = {}
        tensors_by_segment: dict[str, Tensor] = {}
        overlay_segments: list[str] = []

        token_offset = 0
        pooling_offset = 0
        crop_offset = 0
        for idx in range(num_images):
            count = int(pooled_counts[idx])
            crop_count = max(0, int(crop_counts[idx]))
            positions = patch_positions[token_offset : token_offset + count]
            token_offset += count
            if len(positions) != count or count <= 0:
                crop_offset += crop_count
                pooling_offset += count
                continue

            cam_name = self._safe_cam_name(image_keys[idx], idx)
            global_tokens = int(global_pooled_counts[idx]) if idx < len(global_pooled_counts) else count
            global_positions = positions[:global_tokens] if global_tokens > 0 else positions
            patch_indices_by_segment[cam_name] = [int(pos) for pos in global_positions]
            patch_counts_by_segment[cam_name] = len(global_positions)
            if global_positions:
                encoder_segments.append((cam_name, min(global_positions), max(global_positions) + 1))
            overlay_segments.append(cam_name)

            if crop_offset < len(model_view_crops):
                global_image = model_view_crops[crop_offset]
            else:
                global_image = obs_on_device[image_keys[idx]]
            image_tensors.append(global_image)
            tensors_by_segment[cam_name] = global_image

            image_pooling = None
            if pooling_rows is not None:
                image_pooling = pooling_rows[pooling_offset : pooling_offset + count]
            pooling_offset += count

            if image_pooling is not None and patch_grid > 0 and crop_count > 1:
                for crop_rel in range(1, crop_count):
                    low = crop_rel * n_patches_per_crop
                    high = (crop_rel + 1) * n_patches_per_crop
                    valid_for_crop = (image_pooling >= low) & (image_pooling < high)
                    token_rows = valid_for_crop.any(dim=1).nonzero(as_tuple=False).flatten()
                    if token_rows.numel() == 0:
                        continue

                    crop_name = f"{cam_name}_crop{crop_rel:02d}"
                    rows = image_pooling.index_select(dim=0, index=token_rows).clone()
                    local_rows = torch.where(
                        (rows >= low) & (rows < high),
                        rows - low,
                        torch.full_like(rows, -1),
                    )
                    crop_positions = [int(positions[int(token_idx)]) for token_idx in token_rows.tolist()]
                    patch_indices_by_segment[crop_name] = crop_positions
                    patch_counts_by_segment[crop_name] = len(crop_positions)
                    if crop_positions:
                        encoder_segments.append((crop_name, min(crop_positions), max(crop_positions) + 1))
                    pooling_by_segment[crop_name] = {
                        "pooling": local_rows.tolist(),
                        "patch_grid": patch_grid,
                    }
                    crop_tensor_idx = crop_offset + crop_rel
                    if crop_tensor_idx < len(model_view_crops):
                        tensors_by_segment[crop_name] = model_view_crops[crop_tensor_idx]
                    overlay_segments.append(crop_name)

            crop_offset += crop_count

        if not encoder_segments:
            return [("encoder", 0, encoder_seq_len)], [], 0, {}

        encoder_segments.sort(key=lambda segment: segment[1])
        first_count = next(iter(patch_counts_by_segment.values()), 0)
        patches_per_cam = (
            first_count if all(v == first_count for v in patch_counts_by_segment.values()) else 0
        )
        extras = {
            "image_patch_indices_by_segment": patch_indices_by_segment,
            "image_patch_counts_by_segment": patch_counts_by_segment,
            "image_overlay_segments": overlay_segments,
            "image_tensors_by_segment": tensors_by_segment,
        }
        attention_mask = batch.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
            valid_mask = attention_mask[0].detach().cpu().to(torch.bool)
            labels = batch.get("labels")
            if torch.is_tensor(labels) and labels.ndim >= 2:
                valid_mask &= labels[0].detach().cpu().eq(-100)
            valid_positions = valid_mask.nonzero(as_tuple=False).flatten()
            # Depth placeholders are prompt positions now, but they carry point-map
            # tokens, not words: they get their own segment and spatial overlay, so
            # counting them here too would put 192 columns of depth mass inside the
            # task clause of every prompt panel.
            depth_token_id = batch.get("depth_token_id")
            skip_ids = {int(patch_id)}
            if depth_token_id is not None:
                skip_ids.add(int(depth_token_id))
            text_positions = [
                int(pos) for pos in valid_positions.tolist()
                if int(row[int(pos)]) not in skip_ids
            ]
            if text_positions:
                extras["text_token_indices_by_segment"] = {"language": text_positions}
        if pooling_by_segment:
            extras["image_pooling_by_segment"] = pooling_by_segment
        return encoder_segments, image_tensors, patches_per_cam, extras

    def _depth_attention_extras(self, batch: dict, obs_on_device: dict[str, Tensor], encoder_seq_len: int):
        """Locate the point-map depth tokens inside the prefix.

        Depth used to be extra columns appended to the expert's cross-attention K/V, so
        the block was the tail past the prompt length. It is now ordinary prefix
        positions: the encoder's tokens are scattered onto DEPTH_TOKEN placeholders in
        build_input_embeddings, so the block is wherever those placeholder ids sit in
        ``input_ids``. Without this the depth positions are real attention mass that no
        segment names and every panel silently folds into "text".

        Returns ``{}`` when the policy has no depth path.
        """
        pointmap_config = getattr(self._cfg.policy, "pointmap_config", None)
        input_ids = batch.get("input_ids")
        depth_token_id = batch.get("depth_token_id")
        if pointmap_config is None or not torch.is_tensor(input_ids) or encoder_seq_len <= 0:
            return {}
        if depth_token_id is None:
            return {}
        ids = input_ids.reshape(input_ids.shape[0], -1)[0] if input_ids.ndim > 1 else input_ids
        indices = (ids == int(depth_token_id)).nonzero(as_tuple=True)[0].tolist()
        n_depth = len(indices)
        if n_depth == 0:
            return {}

        height, width = pointmap_config.image_size
        patch = int(pointmap_config.patch_size)
        grid_hw = (height // patch, width // patch)  # row-major, see pointmap.patchify
        if grid_hw[0] * grid_hw[1] != n_depth:
            logging.warning(
                f"[probe] {n_depth} depth positions do not match the {grid_hw} point-map grid; "
                "labelling the block but skipping its spatial overlay."
            )
            grid_hw = None

        depth = obs_on_device.get(f"observation.depth.{pointmap_config.depth_key}")
        image = None
        if torch.is_tensor(depth):
            # Display-scale the raw 0.1 mm levels over the configured working range,
            # in the [-1, 1] convention the overlay renderer expects. Invalid pixels
            # (0) stay at the floor rather than dominating the scale.
            units = float(pointmap_config.depth_units_mm)
            lo = float(pointmap_config.z_min_mm) / units
            hi = float(pointmap_config.z_max_mm) / units
            frame = depth.detach().float().cpu().reshape(1, 1, *depth.shape[-2:])
            valid = frame > 0
            scaled = ((frame - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0) * valid
            image = (scaled * 2.0 - 1.0).expand(1, 3, -1, -1).contiguous()

        return {
            "depth_segment": {
                "name": "depth",
                "indices": indices,
                "grid_hw": grid_hw,
                "image": image,
                "is_null_bank": not torch.is_tensor(depth),
            }
        }

    def _pack_molmoact2_result(self, cross_attn, self_attn, batch, obs_on_device):
        """Wrap captured attention dicts in an AttentionCaptureResult."""
        encoder_seq_len = 0
        if cross_attn:
            encoder_seq_len = int(next(iter(cross_attn.values())).shape[-1])

        encoder_segments, image_tensors, patches_per_cam, image_extras = self._image_attention_metadata(
            batch, obs_on_device, encoder_seq_len
        )
        extras = {"_capture_caveat": "viz path: last flow-matching step"}
        extras.update(image_extras)

        depth_extras = self._depth_attention_extras(batch, obs_on_device, encoder_seq_len)
        extras.update(depth_extras)
        if depth_extras:
            segment = depth_extras["depth_segment"]
            encoder_segments = list(encoder_segments) + [
                (segment["name"], segment["indices"][0], segment["indices"][-1] + 1)
            ]

        return AttentionCaptureResult(
            cross_attn_by_layer=cross_attn,
            self_attn_by_layer=self_attn,
            encoder_segments=encoder_segments,
            encoder_pad_masks=batch.get("attention_mask"),
            image_tensors=image_tensors,
            patches_per_cam=patches_per_cam,
            task_tokens=batch.get("input_ids"),
            subtask_tokens=None,
            tokenizer=self._tokenizer(),
            extras=extras,
        )

    # ── Critic / value head ──────────────────────────────────────────────────

    def _critic_batch(self, obs: dict[str, Tensor], task_str: str) -> dict:
        # No advantage clause: the critic is trained on prompts without one
        # (update_critic never threads advantage into the
        # critic forward). Injecting "positive"/"negative" here would feed the
        # critic an out-of-distribution prompt it never saw during training.
        return self._make_batch(obs, task_str)

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
        gt_subtask: str | None = None,
        metadata: dict | None = None,
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
            batch = self._make_batch(obs, task_str, subtask=gt_subtask, metadata=metadata)
            self._set_probe_cuda_graph_enabled(False)
            # NOTE: predict_action_chunk runs the full flow-matching loop.
            # Captured hidden states reflect the LAST step. Same caveat as
            # capture_attention.
            self._policy.predict_action_chunk(batch, inference_action_mode=self._inference_action_mode())
        finally:
            self._restore_probe_cuda_graph_enabled()
            h_enc.remove()
            h_act.remove()

        out: dict[str, Tensor] = {}
        for site, tensor in captured.items():
            # tensor: [B, seq, hidden_dim] — mean over seq, squeeze batch.
            out[site] = tensor.mean(dim=1).squeeze(0).cpu()
        return out
