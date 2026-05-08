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

"""TinyPI0.5v2 RLT policy: frozen tinypi05v2 VLA + lightweight RL-token actor/critic.

Mirrors `lerobot.rl.rlt_tinypi05` but targets the self-contained `tinypi05v2`
backbone (no `transformers` / `pistar06` / `pi05` dependency chain). The public
surface that RLT needs (`model.embed_prefix`, `model.sample_actions`,
`_preprocess_images`, `OBS_LANGUAGE_TOKENS` / `OBS_LANGUAGE_ATTENTION_MASK`
batch keys) is identical between v1 and v2, so the wrapper only swaps the base
config / policy class. The reusable RLT pieces (`RLTokenAutoencoder`,
`RLTActorHead`, `RLTCriticHead`, `RLTCriticEnsemble`, losses, target-network
helpers, checkpoint helpers) come straight from `rlt_pi05`.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.policies.tinypi05v2.configuration_tinypi05v2 import TinyPI05V2Config
from lerobot.policies.tinypi05v2.modeling_tinypi05v2 import TinyPI05V2Policy
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
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

logger = logging.getLogger(__name__)


@PreTrainedConfig.register_subclass("tinypi05v2_rlt")
@dataclass
class TinyPI05V2RLTConfig(TinyPI05V2Config):
    """TinyPI0.5v2 wrapper config for RLT.

    The base `TinyPI05V2Config` controls the frozen VLA architecture; the
    `rlt_*` fields configure the lightweight RL-token autoencoder plus
    actor/critic heads layered on top. Field semantics match
    `TinyPI05RLTConfig`; only the underlying VLA implementation differs.
    """

    rlt_enabled: bool = False
    rlt_embedding_checkpoint: str | None = None
    rlt_head_checkpoint: str | None = None
    rlt_chunk_size: int = 10
    # If None, defaults to vlm_width at construction time so the RL-token
    # bottleneck matches the prefix hidden dim.
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
    # Paper Eq. (4): pi(a|x, ã) = N(mu_theta(x, ã), sigma^2 I). "gaussian" matches
    # the paper; "residual" preserves the legacy ã + scale*tanh(MLP(...)) head.
    rlt_actor_mode: Literal["gaussian", "residual"] = "gaussian"
    # Fixed exploration std used when sampling actions during online data
    # collection. Set 0 to disable noise (pure mean).
    rlt_action_std: float = 0.05
    rlt_num_critics: int = 1
    rlt_bc_beta: float = 1.0
    rlt_bc_action_weights: list[float] | None = None
    rlt_jerk_beta: float = 0.0
    rlt_reference_dropout_p: float = 0.5
    advantage_scaling: float = 1.0
    # Naming kept for parity with the DRTC server, which logs/reads this field
    # uniformly across `pi05_rlt`, `tinypi05_rlt`, and `tinypi05v2_rlt`.
    pi05_checkpoint: str | None = None

    @classmethod
    def from_base_config(
        cls, base_config: TinyPI05V2Config, **overrides: Any
    ) -> "TinyPI05V2RLTConfig":
        kwargs: dict[str, Any] = {}
        for field_info in fields(TinyPI05V2Config):
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


class TinyPI05V2RLTPolicy(TinyPI05V2Policy):
    """RLT wrapper around `TinyPI05V2Policy`.

    The frozen tinypi05v2 VLA stays unchanged. With no actor checkpoint
    loaded, `predict_action_chunk` is an exact pass-through of the VLA
    reference chunk (matching the behavior of `TinyPI05RLTPolicy`).
    """

    config_class = TinyPI05V2RLTConfig
    name = "tinypi05v2_rlt"

    def __init__(self, config: TinyPI05V2RLTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: TinyPI05V2RLTConfig

        arch = config.resolved_architecture()
        hidden_dim = arch.vlm_width
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

        # Persist the resolved token_dim so downstream consumers (DRTC server,
        # offline head trainer) can recover it from the config.
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
            logger.info(
                "Loaded RLT embedding checkpoint from %s", self.config.rlt_embedding_checkpoint
            )

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
            logger.info(
                "Loaded RLT actor/critic checkpoint from %s", self.config.rlt_head_checkpoint
            )

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

    def compute_prefix_embeddings(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (prefix_embs, prefix_pad_masks, prefix_att_masks) from the frozen VLA.

        `TinyPI05V2Pytorch.embed_prefix` returns `(embs, pad_masks, att_masks)`
        with the same semantics as the v1/`PiStar06Pytorch` baseline; the
        downstream RL-token autoencoder uses `prefix_pad_masks` for
        key-padding.
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        return self.model.embed_prefix(images, img_masks, tokens, masks)

    @torch.no_grad()
    def extract_rl_token(self, batch: dict[str, Tensor]) -> Tensor:
        prefix_embs, prefix_pad_masks, _ = self.compute_prefix_embeddings(batch)
        return self.rlt_embedding.encode(prefix_embs, prefix_pad_masks)

    @torch.no_grad()
    def predict_vla_reference_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

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
    ) -> dict[str, Tensor]:
        if model == "rlt_embedding":
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, _ = self.compute_prefix_embeddings(batch)
                target = prefix_embs.detach()
            _, reconstruction = self.rlt_embedding(prefix_embs.detach(), prefix_pad_masks)
            loss = self.rlt_embedding.reconstruction_loss(reconstruction, target, prefix_pad_masks)
            return {"loss_rlt_embedding": loss}
        raise NotImplementedError(
            "TinyPI05V2RLTPolicy currently supports offline `rlt_embedding` training only."
        )
