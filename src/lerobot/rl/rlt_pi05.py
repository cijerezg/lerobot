import copy
import logging
import time
import builtins
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig, MultiAdamConfig
from lerobot.policies.pi05_full.configuration_pi05 import PI05FullConfig
from lerobot.policies.pi05_full.modeling_pi05 import PI05FullPolicy, get_gemma_config
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

import lerobot.rl.rl_pi05  # noqa: F401  # Register pi05_rl configs used by existing checkpoints.

logger = logging.getLogger(__name__)


class RLTokenAutoencoder(nn.Module):
    """Compact bottleneck readout trained to reconstruct frozen VLA prefix embeddings."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_dim: int = 2048,
        max_seq_len: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.max_seq_len = max_seq_len

        self.rl_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.decoder_queries = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)

        self.to_token = nn.Linear(hidden_dim, token_dim)
        self.from_token = nn.Linear(token_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, prefix_embs: Tensor, prefix_pad_mask: Tensor | None = None) -> Tensor:
        prefix_embs = prefix_embs.to(dtype=self.rl_query.dtype)
        batch_size = prefix_embs.shape[0]
        query = self.rl_query.expand(batch_size, -1, -1).to(dtype=prefix_embs.dtype, device=prefix_embs.device)
        encoder_input = torch.cat([prefix_embs, query], dim=1)

        key_padding_mask = None
        if prefix_pad_mask is not None:
            query_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=prefix_pad_mask.device)
            full_mask = torch.cat([prefix_pad_mask.bool(), query_mask], dim=1)
            key_padding_mask = ~full_mask

        encoded = self.encoder(encoder_input, src_key_padding_mask=key_padding_mask)
        return self.to_token(encoded[:, -1])

    def decode(self, rl_token: Tensor, seq_len: int) -> Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        memory = self.from_token(rl_token).unsqueeze(1)
        queries = self.decoder_queries[:, :seq_len].expand(rl_token.shape[0], -1, -1)
        queries = queries.to(dtype=memory.dtype, device=memory.device)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=memory.device),
            diagonal=1,
        )
        decoded = self.decoder(queries, memory, tgt_mask=causal_mask)
        return self.output_proj(decoded)

    def forward(self, prefix_embs: Tensor, prefix_pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        rl_token = self.encode(prefix_embs, prefix_pad_mask)
        reconstruction = self.decode(rl_token, seq_len=prefix_embs.shape[1])
        return rl_token, reconstruction

    @staticmethod
    def reconstruction_loss(
        reconstruction: Tensor,
        target: Tensor,
        prefix_pad_mask: Tensor | None = None,
    ) -> Tensor:
        loss = F.mse_loss(reconstruction.float(), target.float(), reduction="none").mean(dim=-1)
        if prefix_pad_mask is None:
            return loss.mean()
        mask = prefix_pad_mask.to(dtype=loss.dtype, device=loss.device)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class RLTActorHead(nn.Module):
    """Small residual chunk policy conditioned on the VLA reference action chunk."""

    def __init__(
        self,
        *,
        token_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.residual_scale = residual_scale
        in_dim = token_dim + proprio_dim + chunk_size * action_dim
        out_dim = chunk_size * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, rl_token: Tensor, proprio: Tensor, reference_chunk: Tensor) -> Tensor:
        flat_ref = reference_chunk.reshape(reference_chunk.shape[0], -1)
        x = torch.cat([rl_token, proprio, flat_ref], dim=-1)
        residual = torch.tanh(self.net(x)).reshape_as(reference_chunk)
        return reference_chunk + self.residual_scale * residual


class RLTCriticHead(nn.Module):
    """Small chunk-level critic for online RLT training."""

    def __init__(
        self,
        *,
        token_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = token_dim + proprio_dim + chunk_size * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rl_token: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        flat_action = action_chunk.reshape(action_chunk.shape[0], -1)
        return self.net(torch.cat([rl_token, proprio, flat_action], dim=-1))


def rlt_critic_loss(policy: "PI05RLTPolicy", batch: dict[str, Tensor], discount: float) -> Tensor:
    """TD critic loss for compact chunk-level RLT transitions."""
    rewards = batch["reward"].to(dtype=batch["rl_token"].dtype)
    dones = batch["done"].to(dtype=batch["rl_token"].dtype)
    with torch.no_grad():
        next_actions = policy.rlt_actor(
            batch["next_rl_token"],
            batch["next_proprio"],
            batch["next_reference_chunk"],
        )
        next_q = policy.rlt_critic_target(
            batch["next_rl_token"],
            batch["next_proprio"],
            next_actions,
        )
        target_q = rewards + float(discount) * (1.0 - dones) * next_q
    pred_q = policy.rlt_critic(batch["rl_token"], batch["proprio"], batch["executed_chunk"])
    return F.mse_loss(pred_q, target_q)


def rlt_actor_loss(
    policy: "PI05RLTPolicy",
    batch: dict[str, Tensor],
    *,
    beta: float | None = None,
    reference_dropout_p: float | None = None,
) -> Tensor:
    """Actor objective with conservative BC/reference regularization."""
    ref_input = batch["reference_chunk"]
    p = policy.config.rlt_reference_dropout_p if reference_dropout_p is None else float(reference_dropout_p)
    if p > 0:
        keep = (torch.rand(ref_input.shape[0], 1, 1, device=ref_input.device) >= p).to(ref_input.dtype)
        ref_input = ref_input * keep

    actor_actions = policy.rlt_actor(batch["rl_token"], batch["proprio"], ref_input)
    q_value = policy.rlt_critic(batch["rl_token"], batch["proprio"], actor_actions)
    intervention_mask = batch["is_intervention"].to(dtype=actor_actions.dtype, device=actor_actions.device)
    bc_target = torch.where(
        intervention_mask > 0.5,
        batch["executed_chunk"].to(dtype=actor_actions.dtype),
        batch["reference_chunk"].to(dtype=actor_actions.dtype),
    )
    bc_beta = policy.config.rlt_bc_beta if beta is None else float(beta)
    return -q_value.mean() + bc_beta * F.mse_loss(actor_actions, bc_target)


def soft_update_rlt_target(policy: "PI05RLTPolicy", tau: float) -> None:
    """Soft-update `rlt_critic_target` toward `rlt_critic`."""
    tau = float(tau)
    with torch.no_grad():
        for target_param, source_param in zip(
            policy.rlt_critic_target.parameters(), policy.rlt_critic.parameters(), strict=True
        ):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def save_rlt_head_checkpoint(
    policy: "PI05RLTPolicy",
    path: str | Path,
    *,
    step: int,
    config: dict[str, Any] | None = None,
) -> None:
    """Save only the online-trainable RLT heads."""
    checkpoint = {
        "rlt_actor": policy.rlt_actor.state_dict(),
        "rlt_critic": policy.rlt_critic.state_dict(),
        "rlt_critic_target": policy.rlt_critic_target.state_dict(),
        "step": int(step),
    }
    if config is not None:
        checkpoint["config"] = dict(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


@PreTrainedConfig.register_subclass("pi05_rlt")
@dataclass
class PI05RLTConfig(PI05FullConfig):
    """PI0.5 wrapper for RLT: frozen VLA plus lightweight RL-token actor-critic heads."""

    rlt_enabled: bool = False
    rlt_embedding_checkpoint: str | None = None
    rlt_head_checkpoint: str | None = None
    rlt_chunk_size: int = 10
    rlt_token_dim: int = 2048
    rlt_token_max_seq_len: int = 1024
    rlt_token_encoder_layers: int = 2
    rlt_token_decoder_layers: int = 2
    rlt_token_num_heads: int = 8
    rlt_actor_hidden_dim: int = 256
    rlt_critic_hidden_dim: int = 256
    rlt_actor_residual_scale: float = 0.25
    rlt_bc_beta: float = 1.0
    rlt_reference_dropout_p: float = 0.5
    subtask_generation_enabled: bool = False
    advantage_scaling: float = 1.0
    pi05_checkpoint: str | None = None

    @classmethod
    def from_base_config(cls, base_config: PI05FullConfig, **overrides: Any) -> "PI05RLTConfig":
        kwargs: dict[str, Any] = {}
        for field_info in fields(PI05FullConfig):
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


class PI05RLTPolicy(PI05FullPolicy):
    """RLT policy wrapper.

    The base PI0.5 VLA remains frozen. If no RLT actor checkpoint is loaded, inference
    is an exact pass-through of the VLA reference chunk.
    """

    config_class = PI05RLTConfig
    name = "pi05_rlt"

    @classmethod
    def from_pretrained(
        cls: builtins.type["PI05RLTPolicy"],
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
    ) -> "PI05RLTPolicy":
        """Load PI0.5 weights into the frozen VLA, accepting full or `pi05_rl` checkpoints."""
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
            config = PI05RLTConfig.from_base_config(
                base_config,
                pi05_checkpoint=str(pretrained_name_or_path),
                rtc_config=None,
            )
        if not isinstance(config, PI05RLTConfig):
            config = PI05RLTConfig.from_base_config(
                config,
                pi05_checkpoint=str(pretrained_name_or_path),
                rtc_config=None,
            )

        model = cls(config, **kwargs)

        try:
            from safetensors.torch import load_file
            from transformers.utils import cached_file

            resolved_file = cached_file(
                str(pretrained_name_or_path),
                "model.safetensors",
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            state_dict = load_file(resolved_file)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load model.safetensors from {pretrained_name_or_path}"
            ) from e

        if any(key.startswith("actor.") for key in state_dict):
            state_dict = {
                key.removeprefix("actor."): value
                for key, value in state_dict.items()
                if key.startswith("actor.")
            }

        fixed_state_dict = model._fix_pytorch_state_dict_keys(state_dict, model.config)
        remapped_state_dict = {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in fixed_state_dict.items()
        }
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

        tie_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        if tie_key in missing_keys:
            paligemma = model.model.paligemma_with_expert.paligemma
            if cls._tie_or_copy_language_embeddings(paligemma):
                missing_keys = [key for key in missing_keys if key != tie_key]

        rlt_missing = [key for key in missing_keys if key.startswith("rlt_")]
        non_rlt_missing = [key for key in missing_keys if not key.startswith("rlt_")]
        if strict and (non_rlt_missing or unexpected_keys):
            raise RuntimeError(
                "Error loading PI05RLTPolicy state dict with strict=True\n"
                f"Missing keys: {non_rlt_missing}\n"
                f"Unexpected keys: {unexpected_keys}"
            )
        if non_rlt_missing:
            logger.warning("Missing non-RLT keys while loading PI05RLTPolicy: %s", non_rlt_missing[:10])
        if unexpected_keys:
            logger.warning("Unexpected keys while loading PI05RLTPolicy: %s", unexpected_keys[:10])
        if rlt_missing:
            logger.info("RLT-specific keys are initialized from scratch: %d missing", len(rlt_missing))

        model.to(config.device)
        model.eval()
        return model

    def __init__(self, config: PI05RLTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: PI05RLTConfig

        hidden_dim = get_gemma_config(config.paligemma_variant).width
        action_dim = self.config.output_features[ACTION].shape[0]
        proprio_dim = self.config.input_features[OBS_STATE].shape[0]

        self.rlt_embedding = RLTokenAutoencoder(
            hidden_dim=hidden_dim,
            token_dim=config.rlt_token_dim,
            max_seq_len=config.rlt_token_max_seq_len,
            num_heads=config.rlt_token_num_heads,
            encoder_layers=config.rlt_token_encoder_layers,
            decoder_layers=config.rlt_token_decoder_layers,
        )
        self.rlt_actor = RLTActorHead(
            token_dim=config.rlt_token_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            chunk_size=config.rlt_chunk_size,
            hidden_dim=config.rlt_actor_hidden_dim,
            residual_scale=config.rlt_actor_residual_scale,
        )
        self.rlt_critic = RLTCriticHead(
            token_dim=config.rlt_token_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            chunk_size=config.rlt_chunk_size,
            hidden_dim=config.rlt_critic_hidden_dim,
        )
        self.rlt_critic_target = copy.deepcopy(self.rlt_critic)

        self._freeze_vla()
        self._rlt_actor_loaded = False
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
            self.rlt_critic.load_state_dict(checkpoint["rlt_critic"], strict=False)
            if "rlt_critic_target" in checkpoint:
                self.rlt_critic_target.load_state_dict(checkpoint["rlt_critic_target"], strict=False)
            else:
                self.rlt_critic_target.load_state_dict(self.rlt_critic.state_dict())
            self._rlt_actor_loaded = True
            logger.info("Loaded RLT actor/critic checkpoint from %s", self.config.rlt_head_checkpoint)

        self.rlt_critic_target.requires_grad_(False)
        self.rlt_critic_target.eval()

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

    def _get_subtask_tokens(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, bool]:
        if not self.config.subtask_generation_enabled:
            return None, None, False

        current_time = time.time()
        interval = self.config.subtask_regeneration_interval
        should_regenerate = (
            self._cached_subtask_tokens is None
            or self._last_subtask_time is None
            or interval <= 0
            or (current_time - self._last_subtask_time) >= interval
        )
        if should_regenerate:
            subtask_tokens, subtask_masks = self.model.generate_subtask_tokens(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=self.config.tokenizer_max_length,
            )
            self._cached_subtask_tokens = subtask_tokens
            self._cached_subtask_masks = subtask_masks
            self._last_subtask_time = current_time
        return self._cached_subtask_tokens, self._cached_subtask_masks, should_regenerate

    def compute_prefix_embeddings(
        self,
        batch: dict[str, Tensor],
        *,
        use_cached_subtask: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        if use_cached_subtask:
            subtask_tokens, subtask_masks, _ = self._get_subtask_tokens(images, img_masks, tokens, masks)
        else:
            subtask_tokens = batch.get(OBS_LANGUAGE_SUBTASK_TOKENS)
            subtask_masks = batch.get(OBS_LANGUAGE_SUBTASK_ATTENTION_MASK)
            if not self.config.subtask_generation_enabled:
                subtask_tokens = None
                subtask_masks = None

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.model.embed_prefix(
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            subtask_tokens=subtask_tokens,
            masks=masks,
            subtask_masks=subtask_masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    @torch.no_grad()
    def extract_rl_token(self, batch: dict[str, Tensor]) -> Tensor:
        prefix_embs, prefix_pad_masks, _ = self.compute_prefix_embeddings(batch)
        return self.rlt_embedding.encode(prefix_embs, prefix_pad_masks)

    @torch.no_grad()
    def predict_vla_reference_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        profile = getattr(self.model, "_profile_inference", False)
        device_is_cuda = next(self.parameters()).device.type == "cuda"

        def _sync() -> None:
            if profile and device_is_cuda:
                torch.cuda.synchronize()

        def _now() -> float:
            _sync()
            return time.perf_counter()

        t_imgs_start = _now() if profile else 0.0
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        if profile:
            self.model._phase_timings_outer = {
                "preprocess_images_ms": (_now() - t_imgs_start) * 1000.0,
            }

        t_subtask_start = _now() if profile else 0.0
        subtask_tokens, subtask_masks, should_regenerate = self._get_subtask_tokens(
            images, img_masks, tokens, masks
        )
        if profile:
            self.model._phase_timings_outer["subtask_gen_ms"] = (_now() - t_subtask_start) * 1000.0
            self.model._phase_timings_outer["subtask_regenerated"] = bool(should_regenerate)

        t_sample_start = _now() if profile else 0.0
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            subtask_tokens,
            subtask_masks,
            **kwargs,
        )
        if profile:
            self.model._phase_timings_outer["sample_actions_ms"] = (_now() - t_sample_start) * 1000.0

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
        actor_ref = reference[:, : self.config.rlt_chunk_size].to(dtype=rl_token.dtype, device=rl_token.device)
        refined_prefix = self.rlt_actor(rl_token, proprio, actor_ref)
        return torch.cat([refined_prefix, reference[:, self.config.rlt_chunk_size :]], dim=1)

    def forward(
        self,
        batch: dict[str, Tensor],
        model: Literal["rlt_embedding", "rlt_actor", "rlt_critic"] | None = None,
    ) -> dict[str, Tensor]:
        if model == "rlt_embedding":
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, _ = self.compute_prefix_embeddings(batch, use_cached_subtask=False)
                target = prefix_embs.detach()
            _, reconstruction = self.rlt_embedding(prefix_embs.detach(), prefix_pad_masks)
            loss = self.rlt_embedding.reconstruction_loss(reconstruction, target, prefix_pad_masks)
            return {"loss_rlt_embedding": loss}
        raise NotImplementedError("PI05RLTPolicy currently supports offline `rlt_embedding` training only.")
