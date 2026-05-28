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

"""MolmoAct2 RLT policy: frozen MolmoAct2 VLA + lightweight RL-token actor/critic.

This mirrors the tiny PI0.5 RLT wrappers while using MolmoAct2's prompt/image
prefix embeddings as the RL-token autoencoder input. The reusable RLT pieces
are imported from `rlt_pi05` to keep the actor, critic, losses, replay, and
checkpoint format shared across RLT backbones.
"""

from __future__ import annotations

import builtins
import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy, _torch_dtype
from lerobot.rl.rlt_pi05 import (  # noqa: F401  re-exported for convenience
    RLTActorHead,
    RLTCriticEnsemble,
    RLTCriticHead,
    RLTokenAutoencoder,
    rlt_actor_loss,
    rlt_critic_loss,
    save_rlt_head_checkpoint,
    soft_update_rlt_target,
)
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


def _resolve_molmo_hidden_dim(policy: MolmoAct2Policy) -> int:
    """Resolve the Molmo text hidden width used by prompt/image embeddings."""
    candidates = [
        getattr(policy._backbone(), "config", None),
        getattr(policy._hf_model(), "config", None),
        getattr(policy.model, "config", None),
    ]
    for cfg in candidates:
        if cfg is None:
            continue
        for attr in ("hidden_size", "d_model", "model_dim", "n_embd"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

    get_input_embeddings = getattr(policy._backbone(), "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        embedding_dim = getattr(embeddings, "embedding_dim", None)
        if isinstance(embedding_dim, int) and embedding_dim > 0:
            return int(embedding_dim)
        weight = getattr(embeddings, "weight", None)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return int(weight.shape[-1])

    raise RuntimeError("Could not infer MolmoAct2 hidden dimension for RLT.")


@PreTrainedConfig.register_subclass("molmoact2_rlt")
@dataclass
class MolmoAct2RLTConfig(MolmoAct2Config):
    """MolmoAct2 wrapper config for RLT.

    The base `MolmoAct2Config` controls the frozen VLA architecture; the
    `rlt_*` fields configure the RL-token autoencoder plus online actor/critic
    heads layered on top.
    """

    rlt_enabled: bool = False
    rlt_embedding_checkpoint: str | None = None
    rlt_head_checkpoint: str | None = None
    rlt_chunk_size: int = 10
    # If None, defaults to the Molmo hidden width at construction time.
    rlt_token_dim: int | None = None
    rlt_token_max_seq_len: int = 1024
    rlt_token_encoder_layers: int = 2
    rlt_token_decoder_layers: int = 2
    rlt_token_num_heads: int = 8
    rlt_actor_hidden_dim: int = 256
    rlt_critic_hidden_dim: int = 256
    rlt_actor_hidden_dims: list[int] | None = None
    rlt_critic_hidden_dims: list[int] | None = None
    rlt_actor_residual_scale: float = 0.25
    rlt_actor_mode: Literal["gaussian", "residual"] = "gaussian"
    rlt_action_std: float = 0.05
    rlt_num_critics: int = 1
    rlt_bc_beta: float = 1.0
    rlt_bc_action_weights: list[float] | None = None
    rlt_jerk_beta: float = 0.0
    rlt_reference_dropout_p: float = 0.5
    advantage_scaling: float = 1.0
    molmoact2_checkpoint: str | None = None

    @classmethod
    def from_base_config(
        cls, base_config: MolmoAct2Config, **overrides: Any
    ) -> "MolmoAct2RLTConfig":
        kwargs: dict[str, Any] = {}
        for field_info in fields(MolmoAct2Config):
            if field_info.init and hasattr(base_config, field_info.name):
                kwargs[field_info.name] = copy.deepcopy(getattr(base_config, field_info.name))
        kwargs.update(overrides)
        return cls(**kwargs)

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "rlt_embedding": {"lr": 1e-4},
                "rlt_actor": {"lr": 3e-4},
                "rlt_critic": {"lr": 3e-4},
            },
        )


class MolmoAct2RLTPolicy(MolmoAct2Policy):
    """RLT wrapper around `MolmoAct2Policy`.

    The frozen MolmoAct2 VLA stays unchanged. With no actor checkpoint loaded,
    `predict_action_chunk` is an exact pass-through of the VLA reference chunk.
    """

    config_class = MolmoAct2RLTConfig
    name = "molmoact2_rlt"

    @classmethod
    def from_pretrained(
        cls: builtins.type["MolmoAct2RLTPolicy"],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> "MolmoAct2RLTPolicy":
        if config is None:
            base_config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
            config = MolmoAct2RLTConfig.from_base_config(
                base_config,
                molmoact2_checkpoint=str(pretrained_name_or_path),
                rtc_config=None,
            )
        elif not isinstance(config, MolmoAct2RLTConfig):
            config = MolmoAct2RLTConfig.from_base_config(
                config,
                molmoact2_checkpoint=str(pretrained_name_or_path),
                rtc_config=None,
            )

        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def __init__(self, config: MolmoAct2RLTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: MolmoAct2RLTConfig

        hidden_dim = _resolve_molmo_hidden_dim(self)
        token_dim = int(config.rlt_token_dim) if config.rlt_token_dim is not None else hidden_dim
        action_dim = self.config.output_features[ACTION].shape[0]
        proprio_dim = self.config.input_features[OBS_STATE].shape[0]

        self.rlt_embedding = RLTokenAutoencoder(
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            max_seq_len=config.rlt_token_max_seq_len,
            num_heads=config.rlt_token_num_heads,
            encoder_layers=config.rlt_token_encoder_layers,
            decoder_layers=config.rlt_token_decoder_layers,
        )
        self.rlt_actor = RLTActorHead(
            token_dim=token_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            chunk_size=config.rlt_chunk_size,
            hidden_dim=config.rlt_actor_hidden_dims or config.rlt_actor_hidden_dim,
            residual_scale=config.rlt_actor_residual_scale,
            actor_mode=config.rlt_actor_mode,
            action_std=config.rlt_action_std,
        )
        critic_kwargs = {
            "token_dim": token_dim,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "chunk_size": config.rlt_chunk_size,
            "hidden_dim": config.rlt_critic_hidden_dims or config.rlt_critic_hidden_dim,
        }
        if config.rlt_num_critics > 1:
            self.rlt_critic = RLTCriticEnsemble(num_critics=config.rlt_num_critics, **critic_kwargs)
        else:
            self.rlt_critic = RLTCriticHead(**critic_kwargs)
        self.rlt_critic_target = copy.deepcopy(self.rlt_critic)

        self.config.rlt_token_dim = token_dim

        self._freeze_vla()
        self._rlt_actor_loaded = False
        self._rlt_loaded_head_step = 0
        self._load_rlt_checkpoints()

    def _freeze_vla(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def _load_rlt_checkpoints(self) -> None:
        if self.config.rlt_embedding_checkpoint:
            checkpoint = torch.load(self.config.rlt_embedding_checkpoint, map_location="cpu")
            state = checkpoint.get("rlt_embedding", checkpoint)
            self.rlt_embedding.load_state_dict(state, strict=True)
            logger.info("Loaded RLT embedding checkpoint from %s", self.config.rlt_embedding_checkpoint)

        if self.config.rlt_head_checkpoint:
            checkpoint = torch.load(self.config.rlt_head_checkpoint, map_location="cpu")
            self.rlt_actor.load_state_dict(checkpoint["rlt_actor"], strict=True)
            self._load_critic_state(self.rlt_critic, checkpoint["rlt_critic"], strict=False)
            if "rlt_critic_target" in checkpoint:
                self._load_critic_state(
                    self.rlt_critic_target,
                    checkpoint["rlt_critic_target"],
                    strict=False,
                )
            else:
                self.rlt_critic_target.load_state_dict(self.rlt_critic.state_dict())
            self._rlt_loaded_head_step = int(checkpoint.get("step", 0))
            self._rlt_actor_loaded = True
            logger.info("Loaded RLT actor/critic checkpoint from %s", self.config.rlt_head_checkpoint)

        self.rlt_critic_target.requires_grad_(False)
        self.rlt_critic_target.eval()

    @staticmethod
    def _load_critic_state(critic: nn.Module, state_dict: dict[str, Tensor], *, strict: bool) -> None:
        if isinstance(critic, RLTCriticEnsemble) and not any(
            key.startswith("critics.") for key in state_dict
        ):
            for head in critic.critics:
                head.load_state_dict(state_dict, strict=strict)
            return
        critic.load_state_dict(state_dict, strict=strict)

    def get_optim_params(self) -> dict[str, Any]:
        return {
            "rlt_embedding": self.rlt_embedding.parameters(),
            "rlt_actor": self.rlt_actor.parameters(),
            "rlt_critic": self.rlt_critic.parameters(),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _prefix_valid_mask(model_inputs: dict[str, Tensor], seq_len: int, device: torch.device) -> Tensor:
        attention_mask = model_inputs.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            valid = attention_mask.to(device=device, dtype=torch.bool)
            if valid.shape[1] == seq_len:
                return valid
            if valid.shape[1] > seq_len:
                return valid[:, :seq_len]
            padded = torch.zeros(valid.shape[0], seq_len, dtype=torch.bool, device=device)
            padded[:, : valid.shape[1]] = valid
            return padded

        input_ids = model_inputs.get("input_ids")
        if torch.is_tensor(input_ids) and input_ids.ndim == 2:
            valid = input_ids.to(device=device) != 0
            valid = valid & (input_ids.to(device=device) != -1)
            if valid.shape[1] == seq_len:
                return valid

        batch_size = int(next(iter(model_inputs.values())).shape[0])
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    def compute_prefix_embeddings(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Return (prefix_embs, prefix_valid_masks, prefix_att_masks) from MolmoAct2.

        MolmoAct2 builds a single prompt/image prefix embedding sequence before
        the text transformer. The RLT token is trained to reconstruct that
        sequence, masking padded prompt tokens using the processor attention mask.
        """
        model_inputs = self._model_inputs(batch)
        model_dtype = _torch_dtype(self.config.model_dtype)
        device = next(self.parameters()).device
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=model_dtype)
            if device.type in {"cuda", "cpu"} and model_dtype in {torch.bfloat16, torch.float16}
            else nullcontext()
        )
        with autocast_context:
            prefix_embs, _causal_mask, _position_ids, _cache_position = (
                self._prepare_joint_training_backbone_inputs(model_inputs)
            )
        prefix_pad_masks = self._prefix_valid_mask(
            model_inputs,
            int(prefix_embs.shape[1]),
            prefix_embs.device,
        )
        return prefix_embs, prefix_pad_masks, prefix_pad_masks

    @torch.no_grad()
    def extract_rl_token(self, batch: dict[str, Tensor]) -> Tensor:
        prefix_embs, prefix_pad_masks, _ = self.compute_prefix_embeddings(batch)
        return self.rlt_embedding.encode(prefix_embs, prefix_pad_masks)

    @torch.no_grad()
    def predict_vla_reference_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        return MolmoAct2Policy.predict_action_chunk(self, batch, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        reference = self.predict_vla_reference_chunk(batch, **kwargs)
        if not self.config.rlt_enabled or not self._rlt_actor_loaded:
            return reference

        prefix_embs, prefix_pad_masks, _ = self.compute_prefix_embeddings(batch)
        rl_token = self.rlt_embedding.encode(prefix_embs, prefix_pad_masks)
        proprio = batch[OBS_STATE].to(dtype=rl_token.dtype, device=rl_token.device)
        actor_ref = reference[:, : self.config.rlt_chunk_size].to(
            dtype=rl_token.dtype, device=rl_token.device
        )
        refined_prefix = self.rlt_actor(rl_token, proprio, actor_ref)
        return torch.cat([refined_prefix, reference[:, self.config.rlt_chunk_size :]], dim=1)

    def forward(
        self,
        batch: dict[str, Tensor],
        model: Literal["rlt_embedding", "rlt_actor", "rlt_critic"] | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor] | tuple[Tensor, dict[str, Any]]:
        if model == "rlt_embedding":
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, _ = self.compute_prefix_embeddings(batch)
                target = prefix_embs.detach()
            _, reconstruction = self.rlt_embedding(prefix_embs.detach(), prefix_pad_masks)
            loss = self.rlt_embedding.reconstruction_loss(reconstruction, target, prefix_pad_masks)
            return {"loss_rlt_embedding": loss}
        if model is None:
            return MolmoAct2Policy.forward(self, batch, reduction=reduction)
        raise NotImplementedError(
            "MolmoAct2RLTPolicy currently supports offline `rlt_embedding` training "
            "and online RLT actor/critic training through the DRTC server."
        )
