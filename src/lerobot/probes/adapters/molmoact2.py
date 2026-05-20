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
        norm_actions = self._policy.predict_action_chunk(batch, inference_action_mode=self._inference_action_mode())
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
        gt_actions: Tensor | None = None,
    ) -> AttentionCaptureResult:
        # Register the action-expert hooks once per adapter. The hooks stay
        # installed (no-ops when the global flag is off), so this is safe.
        if not getattr(self, "_attn_hooks_registered", False):
            register_action_attention_probing(self._policy._action_expert())
            self._attn_hooks_registered = True

        if requires_grad:
            return self._capture_attention_jacobian(obs, task_str, timestep, layers, gt_actions)
        return self._capture_attention_viz(obs, task_str, timestep, layers)

    def _flow_probe_inputs(self, obs, task_str, timestep, gt_actions: Tensor | None = None):
        device = self._device
        chunk_size = self.chunk_size
        action_dim = self.action_dim

        if gt_actions is None:
            action_targets = torch.zeros(1, chunk_size, action_dim, device=device)
        else:
            action_targets = gt_actions[:chunk_size, :action_dim].unsqueeze(0).to(device)

        obs_on_device = {k: v.to(device) for k, v in obs.items()}
        batch = self._make_batch(obs, task_str, advantage=1.0, gt_actions=action_targets)
        model_inputs = self._policy._model_inputs(batch)
        num_t = max(1, int(getattr(self._policy.config, "num_flow_timesteps", 1)))
        action_dtype = next(self._policy._action_expert().parameters()).dtype
        timesteps_tensor = torch.full(
            (1, num_t), float(timestep), device=device, dtype=action_dtype,
        )
        return obs_on_device, batch, model_inputs, timesteps_tensor

    @torch.no_grad()
    def _capture_attention_viz(self, obs, task_str, timestep, layers):
        obs_on_device, batch, model_inputs, timesteps_tensor = self._flow_probe_inputs(
            obs, task_str, timestep,
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

    def _capture_attention_jacobian(self, obs, task_str, timestep, layers, gt_actions=None):
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
            obs, task_str, timestep, gt_actions,
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
            text_positions = [
                int(pos) for pos in valid_positions.tolist()
                if int(row[int(pos)]) != int(patch_id)
            ]
            if text_positions:
                extras["text_token_indices_by_segment"] = {"language": text_positions}
        if pooling_by_segment:
            extras["image_pooling_by_segment"] = pooling_by_segment
        return encoder_segments, image_tensors, patches_per_cam, extras

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
