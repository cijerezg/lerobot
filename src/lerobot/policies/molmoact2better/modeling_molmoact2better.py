#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect

import torch
from torch import Tensor

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

from .configuration_molmoact2better import MolmoAct2BetterConfig


class MolmoAct2BetterPolicy(MolmoAct2Policy):
    """MolmoAct2 policy with low-risk inference hot-loop fixes.

    The meaningful speedup here is for batched `select_action`: the base policy
    can regenerate a full batch chunk once per empty item queue. This wrapper
    generates the batch chunk once and fans each row into its queue. The async
    DRTC server calls `predict_action_chunk` directly, so it does not exercise
    that queue-refill fix.
    """

    config_class = MolmoAct2BetterConfig
    name = "molmoact2better"

    def _ensure_eval_mode(self) -> None:
        # `from_pretrained` and `policy_server_drtc` already put policies in
        # eval mode. Keep this as a safety net for direct callers, but avoid an
        # unconditional `self.eval()` because walking the full MolmoAct2 module
        # tree on every inference call is measurable overhead.
        model = getattr(self, "model", None)
        if self.training or (model is not None and model.training):
            self.eval()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        # DRTC uses this path. In normal server usage this should be performance
        # equivalent to MolmoAct2 because the server already calls `.eval()`;
        # the guard only prevents accidental training-mode inference elsewhere.
        self._ensure_eval_mode()
        return super().predict_action_chunk(batch, **kwargs)

    def _generate_actions_from_inputs_with_rtc(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim_is_pad: Tensor | None,
        num_steps: int | None,
        generator: torch.Generator | None,
        inference_delay: int | None,
        prev_chunk_left_over: Tensor | None,
        execution_horizon: int | None,
    ) -> Tensor:
        backbone = self._backbone()
        action_expert = self._action_expert()
        outputs = backbone(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoder_kv_states = backbone._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        depth_gate, depth_mask = backbone._depth_gate_from_condition(
            input_ids=model_inputs.get("input_ids"),
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = backbone._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states,
            depth_mask,
            depth_gate,
        )

        steps = int(num_steps or backbone.config.flow_matching_num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {steps}.")
        source_tensor = encoder_kv_states[0][0]
        batch_size = int(source_tensor.shape[0])
        device = source_tensor.device
        trajectory = torch.randn(
            batch_size,
            self._generation_action_horizon(),
            int(backbone.config.max_action_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.config.mask_action_dim_padding:
            trajectory = self._mask_action_dim_tensor(trajectory, action_dim_is_pad)

        action_context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=trajectory.dtype)
            for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )

        dt = 1.0 / steps
        mask_enabled = self.config.mask_action_dim_padding
        use_rtc_guidance = (
            self._rtc_enabled()
            and self.rtc_processor is not None
            and prev_chunk_left_over is not None
            and inference_delay is not None
        )
        rtc_accepts_num_steps = False
        if use_rtc_guidance:
            rtc_accepts_num_steps = (
                "num_flow_matching_steps" in inspect.signature(self.rtc_processor.denoise_step).parameters
            )

        for idx, _flow_timestep in enumerate(flow_timesteps):
            modulation = modulation_cache[idx]

            def denoise_step(input_trajectory: Tensor, step_modulation=modulation) -> Tensor:
                velocity = action_expert.forward_with_context(
                    input_trajectory,
                    step_modulation.conditioning,
                    context=action_context,
                    modulation=step_modulation,
                )
                if mask_enabled:
                    velocity = self._mask_action_dim_tensor(velocity, action_dim_is_pad)
                return velocity

            if use_rtc_guidance:

                def rtc_denoise_step(input_trajectory: Tensor) -> Tensor:
                    return -denoise_step(input_trajectory)

                rtc_kwargs = {
                    "x_t": trajectory,
                    "prev_chunk_left_over": prev_chunk_left_over,
                    "inference_delay": int(inference_delay),
                    "time": 1.0 - idx / steps,
                    "original_denoise_step_partial": rtc_denoise_step,
                    "execution_horizon": execution_horizon,
                }
                if rtc_accepts_num_steps:
                    rtc_kwargs["num_flow_matching_steps"] = steps
                rtc_velocity = self.rtc_processor.denoise_step(**rtc_kwargs)
                velocity = -rtc_velocity
            else:
                # In the async server the policy can be marked RTC-enabled
                # before any previous chunk is available. Base MolmoAct2 still
                # calls into the RTC wrapper in that case; the wrapper returns
                # the plain denoiser result, but it also builds kwargs, inspects
                # signatures, and synchronizes a scalar timestep every flow
                # step. Skip that no-op wrapper until there is prefix guidance.
                velocity = denoise_step(trajectory)

            trajectory = trajectory + dt * velocity
            if mask_enabled:
                trajectory = self._mask_action_dim_tensor(trajectory, action_dim_is_pad)
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=idx / steps, x_t=trajectory, v_t=velocity)

        return trajectory

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self._ensure_eval_mode()
        if self._rtc_enabled():
            raise AssertionError("RTC is not supported for select_action, use it with predict_action_chunk")

        model_inputs = self._model_inputs(batch)
        batch_size = int(next(iter(model_inputs.values())).shape[0])
        empty_indices = [idx for idx in range(batch_size) if not self._action_queues[idx]]

        if empty_indices:
            # Base MolmoAct2 refills inside the per-item loop, so when all B
            # queues are empty it computes `predict_action_chunk(batch)` B
            # times. One chunk already contains rows for every batch item.
            chunk = self.predict_action_chunk(batch, **kwargs)
            for batch_idx in empty_indices:
                queue = self._action_queues[batch_idx]
                for step in torch.unbind(chunk[batch_idx], dim=0):
                    queue.append(step)

        actions: list[Tensor] = []
        for batch_idx in range(batch_size):
            queue = self._action_queues[batch_idx]
            if not queue:
                raise RuntimeError("MolmoAct2Better produced an empty action chunk.")
            actions.append(queue.popleft())
        return torch.stack(actions, dim=0)
