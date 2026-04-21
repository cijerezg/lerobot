"""Attention-map wrapper for Pi0.5 / `pi05_rl` policies.

Mirrors the public surface of `ACTPolicyWithAttention` from
https://github.com/villekuosmanen/physical-AI-interpretability so the runner
script can use the same `(action, attention_maps)` + `visualize_attention`
pattern, but for the dual-stack PaliGemma + Gemma-300m architecture used by
LeRobot's `pi05_full` / `pi05_rl` policies.

What we capture
---------------
During inference (`predict_action_chunk`), Pi0.5 first prefills the prefix
(per-camera SigLIP image patches + tokenized text) into the PaliGemma KV
cache, then runs `num_inference_steps` flow-matching denoise steps. Each
denoise step calls `gemma_expert.model.forward(inputs_embeds=[None, suffix])`
with the cached prefix; the expert's `self_attn.forward` therefore computes
attention with shape `[B, num_heads, chunk_size, prefix_len + chunk_size]`.

We register a forward hook on `gemma_expert.model.layers[target_layer_idx]
.self_attn`, filter for calls where `q_len == chunk_size` (i.e. the suffix
calls during denoising, not the standalone prefix prefill), average across
heads, action positions, and denoise steps, and slice the per-camera image
columns out of the prefix region. The prefix layout begins with each
camera's SigLIP image tokens in the order of `config.input_features`, so
camera spans are simply `[i*tokens_per_image : (i+1)*tokens_per_image]`.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch


class PI05PolicyWithAttention:
    """Composition wrapper around a `PI05FullPolicy` / `PI05RLPolicy`.

    The wrapper is intentionally small: it does not subclass the policy. It
    holds a reference to the loaded policy and a preprocessor pipeline, and
    on each `select_action` call it installs a forward hook on the action
    expert's self-attention to capture per-denoise-step attention weights.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        preprocessor,
        *,
        target_layer_idx: int = -1,
        capture_strategy: str = "mean",
        task_str: str | None = None,
        inference_advantage: float | None = None,
    ) -> None:
        """Wrap a Pi0.5 policy.

        Args:
            policy: Loaded `PI05FullPolicy` (or `PI05RLPolicy`).
            preprocessor: Policy preprocessor pipeline returned by
                `make_pre_post_processors(...)` or
                `make_pi05_full_processors_with_upgrade(...)`.
            target_layer_idx: Which expert layer's self-attention to hook.
                `-1` = last layer (default).
            capture_strategy: `"mean"` averages over heads, action positions,
                and denoise steps. `"last_action"` slices only the last
                action token's attention (the next-to-execute action).
            task_str: Optional override for the task prompt. Defaults to
                `policy.config.task` if present, else read per-call from the
                observation dict's `"task"` key.
            inference_advantage: Optional override for the `inference_advantage`
                field that `pi05_rl`'s prepare-state step reads from
                `complementary_data`. Defaults to `policy.config.inference_advantage`
                when present, else `1.0`.
        """
        if capture_strategy not in {"mean", "last_action"}:
            raise ValueError(
                f"capture_strategy must be 'mean' or 'last_action', got {capture_strategy!r}"
            )

        self.policy = policy
        self.preprocessor = preprocessor
        self.config = policy.config
        self.target_layer_idx = target_layer_idx
        self.capture_strategy = capture_strategy

        self._task_str = (
            task_str
            if task_str is not None
            else (getattr(policy.config, "task", None) or "")
        )
        self._inference_advantage = (
            inference_advantage
            if inference_advantage is not None
            else float(getattr(policy.config, "inference_advantage", 1.0) or 1.0)
        )

        self.image_keys = self._infer_image_keys()
        self.num_images = len(self.image_keys)
        if self.num_images == 0:
            raise ValueError(
                "PI05PolicyWithAttention found no image features in policy.config.input_features"
            )

        try:
            self._target_self_attn = (
                policy.model.paligemma_with_expert.gemma_expert.model.layers[
                    target_layer_idx
                ].self_attn
            )
        except (AttributeError, IndexError) as exc:
            raise AttributeError(
                "Policy structure does not match expected pi05/pi05_rl layout. "
                "Could not access policy.model.paligemma_with_expert.gemma_expert"
                f".model.layers[{target_layer_idx}].self_attn"
            ) from exc

        self.tokens_per_image = self._probe_tokens_per_image()

        self.last_observation: Optional[dict[str, Any]] = None
        self.last_attention_maps: Optional[list[Optional[np.ndarray]]] = None

    def _infer_image_keys(self) -> list[str]:
        keys: list[str] = []
        input_features = getattr(self.config, "input_features", {}) or {}
        for key, feat in input_features.items():
            ftype = getattr(feat, "type", None)
            type_str = str(ftype).upper() if ftype is not None else ""
            if "VISUAL" in type_str or "IMAGE" in key:
                keys.append(key)
        return keys

    def _probe_tokens_per_image(self) -> int:
        """Run `embed_image` once on a zero tensor to derive the SigLIP grid.

        Returns the number of image tokens emitted per camera (e.g. 256 for
        PaliGemma 224x224 with patch size 14).
        """
        try:
            paligemma_with_expert = self.policy.model.paligemma_with_expert
        except AttributeError as exc:
            raise AttributeError(
                "Could not access policy.model.paligemma_with_expert to probe "
                "image token count."
            ) from exc

        device = next(paligemma_with_expert.parameters()).device
        first_image_key = self.image_keys[0]
        feat = self.config.input_features[first_image_key]
        c, h, w = feat.shape
        dummy = torch.zeros(1, c, h, w, device=device, dtype=torch.float32)
        try:
            q_proj_dtype = (
                paligemma_with_expert.paligemma.language_model.layers[0]
                .self_attn.q_proj.weight.dtype
            )
        except AttributeError:
            q_proj_dtype = torch.float32
        if q_proj_dtype == torch.bfloat16:
            dummy = dummy.to(dtype=torch.bfloat16)

        with torch.no_grad():
            img_emb = paligemma_with_expert.embed_image(dummy)
        return int(img_emb.shape[1])

    def _build_raw_observation(
        self, observation: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a raw observation dict for the preprocessor.

        Mirrors `_inject_pi05_complementary_data` in
        `lerobot.async_inference.policy_server_drtc`: pi05_rl's
        `Pi05FullPrepareStateTokenizerProcessorStep` requires
        `complementary_data` to contain `task`, `subtask`, and `advantage`
        (a (1, 1) float tensor).
        """
        obs = dict(observation)

        task_str = obs.get("task")
        if isinstance(task_str, str):
            task_value = task_str
        elif self._task_str:
            task_value = self._task_str
        else:
            task_value = ""

        comp = obs.get("complementary_data")
        if not isinstance(comp, dict):
            comp = {}
        comp.setdefault("task", [task_value])
        comp.setdefault("subtask", [""])
        if "advantage" not in comp:
            comp["advantage"] = torch.tensor(
                [[self._inference_advantage]], dtype=torch.float32
            )
        obs["complementary_data"] = comp
        obs.setdefault("robot_type", "")
        return obs

    @staticmethod
    def _force_eager_attention(policy: torch.nn.Module) -> None:
        """Match `sample_actions`: force eager attention on both stacks so
        `eager_attention_forward` (and hence the hook output) is used."""
        try:
            pwe = policy.model.paligemma_with_expert
            pwe.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            pwe.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        except AttributeError:
            pass

    def select_action(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, list[Optional[np.ndarray]]]:
        """Run one inference + capture attention for the action expert.

        Args:
            observation: Raw observation dict containing image keys
                (`(1, C, H, W)` or `(C, H, W)` float32 in `[0, 1]`),
                `OBS_STATE` (`(1, state_dim)` or `(state_dim,)`), and
                optionally `task` (str) and `complementary_data` (dict).

        Returns:
            (`action_chunk`, `attention_maps`) where `action_chunk` is the
            full `(B, chunk_size, action_dim)` tensor returned by
            `predict_action_chunk` and `attention_maps` is a list with one
            `(H_grid, W_grid)` numpy array per camera (globally min-max
            normalized across cameras), or `None` for cameras whose
            attention could not be extracted.
        """
        self.last_observation = dict(observation)

        raw_obs = self._build_raw_observation(observation)
        batch = self.preprocessor(raw_obs)

        self._force_eager_attention(self.policy)
        chunk_size = int(self.config.chunk_size)

        captured: list[torch.Tensor] = []

        def hook(module, input_args, output):  # noqa: ARG001
            if not isinstance(output, tuple) or len(output) < 2:
                return
            attn_w = output[1]
            if attn_w is None:
                return
            if attn_w.dim() != 4 or attn_w.shape[-2] != chunk_size:
                return
            captured.append(attn_w.detach().to("cpu").float())

        handle = self._target_self_attn.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                action_chunk = self.policy.predict_action_chunk(batch)
        finally:
            handle.remove()

        if not captured:
            print(
                "Warning (PI05PolicyWithAttention): No attention weights captured. "
                "Check that the expert layer was reached and that "
                "`_attn_implementation == 'eager'` is set."
            )
            attention_maps: list[Optional[np.ndarray]] = [None] * self.num_images
        else:
            attention_maps = self._aggregate_to_camera_maps(captured)

        self.last_attention_maps = attention_maps
        return action_chunk, attention_maps

    def _aggregate_to_camera_maps(
        self, captured: list[torch.Tensor]
    ) -> list[Optional[np.ndarray]]:
        """Reduce per-step `[B, H, chunk, prefix+chunk]` tensors into one 2D
        attention map per camera, globally min-max normalized."""
        stacked = torch.stack(captured, dim=0)
        if self.capture_strategy == "last_action":
            attn_q = stacked[:, :, :, -1, :]
        else:
            attn_q = stacked.mean(dim=-2)

        attn = attn_q.mean(dim=(0, 1, 2))

        kv_len = attn.shape[0]
        prefix_len = kv_len - self.config.chunk_size
        if prefix_len <= 0:
            print(
                f"Warning (PI05PolicyWithAttention): unexpected kv_len={kv_len}; "
                f"expected prefix + chunk_size ({self.config.chunk_size})."
            )
            return [None] * self.num_images

        side = int(round(self.tokens_per_image ** 0.5))
        if side * side != self.tokens_per_image:
            print(
                f"Warning (PI05PolicyWithAttention): tokens_per_image="
                f"{self.tokens_per_image} is not a perfect square; cannot reshape."
            )
            return [None] * self.num_images

        raw_maps: list[Optional[np.ndarray]] = []
        for i in range(self.num_images):
            start = i * self.tokens_per_image
            end = start + self.tokens_per_image
            if end > prefix_len:
                print(
                    f"Warning (PI05PolicyWithAttention): camera {i} span [{start}, {end}) "
                    f"exceeds prefix_len={prefix_len}. Skipping."
                )
                raw_maps.append(None)
                continue
            cam_attn = attn[start:end].reshape(side, side).cpu().numpy()
            raw_maps.append(cam_attn)

        valid = [m for m in raw_maps if m is not None]
        if not valid:
            return raw_maps

        global_min = float(min(m.min() for m in valid))
        global_max = float(max(m.max() for m in valid))
        if global_max <= global_min:
            return [
                None if m is None else np.zeros_like(m, dtype=np.float32)
                for m in raw_maps
            ]
        scale = global_max - global_min
        return [
            None if m is None else ((m - global_min) / scale).astype(np.float32)
            for m in raw_maps
        ]

    def _extract_images(self, observation: dict[str, Any]) -> list[Optional[torch.Tensor]]:
        images: list[Optional[torch.Tensor]] = []
        for key in self.image_keys:
            value = observation.get(key)
            if value is None:
                images.append(None)
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            images.append(value)
        return images

    def visualize_attention(
        self,
        attention_maps: Optional[list[Optional[np.ndarray]]] = None,
        observation: Optional[dict[str, Any]] = None,
        *,
        images: Optional[list[Optional[torch.Tensor]]] = None,
        use_rgb: bool = False,
        overlay_alpha: float = 0.5,
    ) -> list[Optional[np.ndarray]]:
        """Overlay attention heatmaps on the original camera frames.

        Returns a list of `(H, W, 3)` uint8 frames (BGR by default; RGB if
        `use_rgb=True`), one per camera. `None` entries are returned for any
        camera whose attention map or image is missing.
        """
        if attention_maps is None:
            attention_maps = self.last_attention_maps
            if attention_maps is None:
                raise ValueError(
                    "No attention_maps provided and no cached maps available."
                )

        if images is None:
            source_obs = observation if observation is not None else self.last_observation
            if source_obs is None:
                raise ValueError(
                    "No observation provided and no cached observation available."
                )
            images = self._extract_images(source_obs)

        visualizations: list[Optional[np.ndarray]] = []
        for img, attn_map in zip(images, attention_maps):
            if img is None or attn_map is None:
                visualizations.append(None)
                continue

            img_np = img.detach().to("cpu") if isinstance(img, torch.Tensor) else img
            if isinstance(img_np, torch.Tensor):
                if img_np.dim() == 4:
                    img_np = img_np.squeeze(0)
                if img_np.dim() != 3:
                    visualizations.append(None)
                    continue
                if img_np.shape[0] in (1, 3):
                    img_np = img_np.permute(1, 2, 0)
                img_np = img_np.float().numpy()
            else:
                img_np = np.asarray(img_np)
                if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
                    img_np = np.transpose(img_np, (1, 2, 0))

            if img_np.dtype != np.float32:
                img_np = img_np.astype(np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

            h, w = img_np.shape[:2]
            attn_resized = cv2.resize(attn_map.astype(np.float32), (w, h))
            heatmap_bgr = cv2.applyColorMap(
                np.uint8(255 * np.clip(attn_resized, 0.0, 1.0)),
                cv2.COLORMAP_JET,
            )

            img_bgr_uint8 = np.uint8(255 * np.clip(img_np, 0.0, 1.0))
            if img_bgr_uint8.shape[-1] == 3:
                img_bgr_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_RGB2BGR)

            vis = cv2.addWeighted(
                img_bgr_uint8, 1.0 - overlay_alpha,
                heatmap_bgr, overlay_alpha, 0,
            )

            if use_rgb:
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            visualizations.append(vis)

        return visualizations

    def __getattr__(self, name: str) -> Any:
        if name in {"policy", "preprocessor"} or name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.policy, name)


__all__ = ["PI05PolicyWithAttention"]
