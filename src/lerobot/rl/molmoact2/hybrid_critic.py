"""MolmoAct2 critic: frozen text/RGB, trainable continuous state/depth, native-width fusion."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from lerobot.utils.constants import OBS_STATE


class CriticEncoder(nn.Module):
    """Frozen copies of the policy backbone's token embedding and vision backbone.

    Emits the prefix tokens the LLM would read: text embeddings with the image features
    added on the <im_patch> positions (MolmoAct2 build_input_embeddings, minus the
    embedding dropout, which is 0 in this backbone). Copied at init and never trained,
    so the critic's input is a fixed function of the observation and prompt whatever
    the actor learns, and a critic checkpoint carries everything it needs.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.wte = copy.deepcopy(backbone.transformer.wte)
        self.vision_backbone = copy.deepcopy(backbone.vision_backbone)
        self.image_patch_id = int(backbone.config.image_patch_id)
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Freezing weights alone would still allow RGB dropout in the shared encoder.
        return super().train(False)

    @property
    def hidden_size(self) -> int:
        return int(self.wte.embedding.shape[-1])

    def forward(self, input_ids: Tensor, images: Tensor | None, token_pooling: Tensor | None) -> Tensor:
        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        tokens = self.wte(input_ids)
        if images is not None:
            image_features = self.vision_backbone(images, token_pooling).to(tokens.device)
            is_image_patch = input_ids.view(-1) == self.image_patch_id
            tokens.view(-1, tokens.shape[-1])[is_image_patch] += image_features
        return tokens


class CriticFusion(nn.Module):
    """Trainable state/depth inputs and transformer, read through a learned value token.

    Runs at the encoder's width (no input projection), so no information is dropped
    before fusion. The target copies this entire module, including its input adapters.
    """

    def __init__(self, config: Any, hidden_size: int, *, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_tokens = int(config.critic_max_tokens)

        self.state_projector: nn.Linear | None = None
        state_feature = getattr(config, "input_features", {}).get(OBS_STATE)
        if state_feature is not None and state_feature.shape:
            self.state_projector = nn.Linear(int(state_feature.shape[0]), self.hidden_size)

        self.pointmap_encoder = None
        self.depth_visual = None
        self.register_parameter("depth_marker", None)
        self.pointmap_config = copy.deepcopy(getattr(config, "pointmap_config", None))
        if self.pointmap_config is not None:
            from lerobot.policies.depth_pointmap.modeling_pointmap import DepthPointmapEncoder
            from lerobot.policies.molmoact2.modeling_molmoact2 import DepthVisualBackbone

            if backbone is None:
                raise ValueError("The critic depth path requires a source RGB backbone.")
            checkpointing = bool(getattr(config, "gradient_checkpointing", False))
            self.pointmap_encoder = DepthPointmapEncoder(
                self.pointmap_config,
                d_mem=self.pointmap_config.token_width,
                gradient_checkpointing=checkpointing,
            )
            self.depth_visual = DepthVisualBackbone(
                self.pointmap_config,
                backbone.vision_backbone,
                gradient_checkpointing=checkpointing,
            )
            # Initialization matches the actor, but no trainable tensors are shared.
            with torch.no_grad():
                device = backbone.transformer.wte.embedding.device
                patch_id = torch.tensor([backbone.config.image_patch_id], device=device)
                marker = backbone.transformer.wte(patch_id)[0].detach().clone()
                self.depth_marker = nn.Parameter(marker)
            # Copies may inherit requires_grad=False from the source RGB modules.
            self.depth_visual.requires_grad_(True)

        # Position zero is reserved for the value token.
        self.position_embeddings = nn.Parameter(torch.empty(1, self.max_tokens + 1, self.hidden_size))
        self.value_token = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(config.critic_num_attention_heads),
            dim_feedforward=int(round(self.hidden_size * float(config.critic_mlp_ratio))),
            dropout=float(config.critic_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.ModuleList(copy.deepcopy(layer) for _ in range(int(config.critic_llm_depth)))
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.distribution_head = nn.Linear(self.hidden_size, int(config.num_value_bins))

        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.value_token, std=0.02)
        nn.init.normal_(self.distribution_head.weight, std=0.02)
        nn.init.zeros_(self.distribution_head.bias)

    def prepare_inputs(self, tokens: Tensor, input_ids: Tensor, batch: dict[str, Any]) -> Tensor:
        """Inject the batch's continuous modalities before the fusion transformer.

        Only the frozen text/RGB tokens are detached. State/depth stay on the TD
        gradient path and use this module's weights (live or EMA target).
        """
        tokens = tokens.detach().to(dtype=self.value_token.dtype).clone()
        states = batch.get("state_values")
        if states is not None:
            if self.state_projector is None:
                raise ValueError("state_values present but critic input_features lacks observation.state.")
            states = states.to(device=tokens.device, dtype=self.state_projector.weight.dtype)
            mask = batch["state_values_mask"].to(device=tokens.device, dtype=torch.bool)
            if states.ndim != 3 or mask.shape != states.shape[:2] or states.shape[0] != tokens.shape[0]:
                raise ValueError("critic state_values must be [B, N, D] with a matching [B, N] mask.")
            is_state = input_ids == int(batch["state_token_id"])
            counts, expected = is_state.sum(dim=1), mask.sum(dim=1)
            if not bool((counts == expected).all()):
                raise ValueError(
                    f"Continuous-state placeholders per sample {counts.tolist()} do not "
                    f"match the shipped state rows {expected.tolist()}."
                )
            tokens[is_state] = tokens[is_state] + self.state_projector(states)[mask].to(tokens.dtype)
        elif batch.get("state_token_id") is not None:
            raise ValueError("critic state_token_id supplied without state_values.")

        depth_token_id = batch.get("depth_token_id")
        if depth_token_id is not None:
            from lerobot.policies.molmoact2.modeling_molmoact2 import _soft_bound_depth_tokens

            if self.pointmap_encoder is None:
                raise ValueError("depth_token_id present but critic pointmap_config is disabled.")
            is_depth = input_ids == int(depth_token_id)
            counts = is_depth.sum(dim=1)
            expected = self.pointmap_config.num_pooled_tokens
            if not bool((counts == expected).all()):
                raise ValueError(f"Expected {expected} depth placeholders per sample, got {counts.tolist()}.")
            fine_tokens = self.pointmap_encoder.memory_from_batch(
                batch,
                batch_size=tokens.shape[0],
                device=tokens.device,
            )
            projected = self.depth_visual(fine_tokens)
            depth = _soft_bound_depth_tokens(
                projected + self.depth_marker.to(projected.dtype),
                float(self.pointmap_config.output_bound),
            )
            # Match the actor seam: replace the arbitrary placeholder embedding.
            tokens[is_depth] = depth.reshape(-1, self.hidden_size).to(tokens.dtype)
        return tokens

    def forward(self, tokens: Tensor, attention_mask: Tensor, *, detach_inputs: bool = True) -> Tensor:
        """Logits over the value bins, [B, num_value_bins]."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"critic expected tokens shaped [B, T, {self.hidden_size}], got {tuple(tokens.shape)}.")
        if attention_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"critic attention mask {tuple(attention_mask.shape)} does not match tokens {tuple(tokens.shape[:2])}."
            )
        batch_size, seq_len, _ = tokens.shape
        if seq_len > self.max_tokens:
            raise ValueError(f"critic received {seq_len} tokens, exceeding critic_max_tokens={self.max_tokens}.")

        dtype = self.value_token.dtype
        hidden_states = (tokens.detach() if detach_inputs else tokens).to(dtype=dtype)
        value_token = self.value_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([value_token, hidden_states], dim=1)
        hidden_states = hidden_states + self.position_embeddings[:, : seq_len + 1]

        attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        keep = torch.cat([torch.ones(batch_size, 1, device=hidden_states.device, dtype=torch.bool), attention_mask], dim=1)
        padding_mask = ~keep
        for block in self.blocks:
            hidden_states = block(hidden_states, src_key_padding_mask=padding_mask)
        return self.distribution_head(self.final_norm(hidden_states[:, 0]))


class MolmoAct2Critic(nn.Module):
    """Distributional V(s): frozen text/RGB + trainable state/depth → fusion → value bins.

    ``forward(..., fusion=critic_target)`` uses the target state/depth adapters and
    transformer. Only frozen text/RGB features are shared between live and target.
    """

    def __init__(self, config: Any, backbone: nn.Module) -> None:
        super().__init__()
        self.num_value_bins = int(config.num_value_bins)
        self.encoder = CriticEncoder(backbone)
        self.fusion = CriticFusion(config, self.encoder.hidden_size, backbone=backbone)

        bin_centers = torch.linspace(
            float(config.value_support_min),
            float(config.value_support_max),
            self.num_value_bins,
        )
        self.register_buffer("bin_centers", bin_centers, persistent=False)
        bin_width = (float(config.value_support_max) - float(config.value_support_min)) / (
            self.num_value_bins - 1
        )
        self.hl_gauss_sigma = float(config.hl_gauss_sigma_ratio) * bin_width

    def forward(
        self,
        input_ids: Tensor,
        images: Tensor | None,
        token_pooling: Tensor | None,
        attention_mask: Tensor,
        fusion: nn.Module | None = None,
        *,
        batch: dict[str, Any] | None = None,
        vision_grad: bool = False,
        input_grad: bool = False,
    ) -> dict[str, Tensor]:
        fusion = self.fusion if fusion is None else fusion
        with torch.no_grad():
            tokens = self.encoder(input_ids, images, token_pooling)
        if batch is not None:
            tokens = fusion.prepare_inputs(tokens, input_ids, batch)
        if vision_grad or input_grad:
            # Differentiate the complete continuous input to fusion, without
            # changing encoder freezing or any parameter's .grad buffer.
            tokens = tokens.detach().requires_grad_(True)
        logits = fusion(tokens, attention_mask, detach_inputs=False)
        probs = functional.softmax(logits, dim=-1)
        out = {"logits": logits, "probs": probs, "value": self.value_from_probs(probs)}
        if vision_grad or input_grad:
            grad, = torch.autograd.grad(out["value"].sum(), tokens)
            token_norms = (grad.float() * attention_mask.bool().unsqueeze(-1)).norm(dim=-1)
            if input_grad:
                # All visible input positions; excludes padding and the learned
                # value token. The per-token norms exactly partition the L2 norm.
                out["input_token_grad_norms"] = token_norms.detach()
                out["input_grad_norm"] = token_norms.norm(dim=-1).detach()
        if vision_grad:
            patches = (input_ids == self.encoder.image_patch_id) & attention_mask.bool()
            if images is None or not patches.any(dim=1).all():
                raise ValueError("Vision gradients require at least one visible image patch per frame.")
            out["vision_grad_norm"] = (grad.float() * patches.unsqueeze(-1)).flatten(1).norm(dim=1)
        return out

    def value_from_probs(self, probs: Tensor) -> Tensor:
        """Expected value under an already normalized distribution."""
        bin_centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype)
        return (probs * bin_centers).sum(dim=-1, keepdim=True)

    def value_from_logits(self, logits: Tensor) -> Tensor:
        return self.value_from_probs(functional.softmax(logits, dim=-1))

    def hl_gauss_target(self, target_v: Tensor) -> Tensor:
        """HL-Gauss target distribution over value bins."""
        if target_v.ndim == 2:
            target_v = target_v.squeeze(-1)
        target_v = target_v.to(device=self.bin_centers.device, dtype=self.bin_centers.dtype)
        internal_edges = 0.5 * (self.bin_centers[:-1] + self.bin_centers[1:])
        z = (internal_edges.unsqueeze(0) - target_v.unsqueeze(-1)) / (self.hl_gauss_sigma * (2.0**0.5))
        cdf_internal = 0.5 * (1.0 + torch.erf(z))
        zeros = torch.zeros_like(cdf_internal[:, :1])
        ones = torch.ones_like(cdf_internal[:, :1])
        cdf_full = torch.cat([zeros, cdf_internal, ones], dim=-1)
        return cdf_full[:, 1:] - cdf_full[:, :-1]

    def one_hot_target(self, target_v: Tensor) -> Tensor:
        """Nearest-bin one-hot target for exact terminal values."""
        if target_v.ndim == 2:
            target_v = target_v.squeeze(-1)
        target_v = target_v.to(device=self.bin_centers.device, dtype=self.bin_centers.dtype)
        idx = torch.argmin(torch.abs(self.bin_centers.unsqueeze(0) - target_v.unsqueeze(-1)), dim=-1)
        return functional.one_hot(idx, num_classes=self.num_value_bins).to(self.bin_centers.dtype)
