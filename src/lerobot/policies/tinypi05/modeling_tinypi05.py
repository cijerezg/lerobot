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

import logging

import torch
from torch import nn

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pistar06.modeling_pistar06 import (
    ActionExpertDirect,
    CONFIG_MAPPING,
    GemmaVariantConfig,
    PaliGemmaDirect,
    PaliGemmaWithExpertModel,
    PiStar06Pytorch,
    _log_mem,
)
from lerobot.policies.tinypi05.configuration_tinypi05 import TinyPI05Architecture, TinyPI05Config

try:
    from transformers import AutoModelForCausalLM, SiglipVisionModel
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
except ImportError:  # transformers may be unavailable in some test envs
    SiglipVisionModel = None
    PaliGemmaMultiModalProjector = None
    AutoModelForCausalLM = None


def _format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def _to_gemma_variant(
    *,
    width: int,
    depth: int,
    mlp_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> GemmaVariantConfig:
    return GemmaVariantConfig(
        width=width,
        depth=depth,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


class TinyPaliGemmaWithExpertModel(PaliGemmaWithExpertModel):
    """PaliGemma + action expert with configurable tiny dimensions.

    When ``pretrained_vision_model`` is provided, the random PaliGemma vision
    tower is replaced after construction with ``SiglipVisionModel.from_pretrained(...)``
    and the multi_modal_projector is rebuilt to map the loaded SigLIP hidden
    size to ``vlm_width``.  This is the recommended path -- training the vision
    tower from scratch on a small robotics dataset (~10k frames) almost always
    converges to a proprio-only local optimum that ignores vision entirely
    (see commit notes).
    """

    def __init__(
        self,
        vlm_config: GemmaVariantConfig,
        action_expert_config: GemmaVariantConfig,
        architecture: TinyPI05Architecture,
        use_adarms: list[bool] | None = None,
        precision: str = "bfloat16",
        image_size: int = 224,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        pretrained_vision_model: str | None = None,
        pretrained_language_embeddings: str | None = None,
    ):
        if CONFIG_MAPPING is None:
            raise ImportError("transformers is required to instantiate TinyPI05")
        if pretrained_vision_model is not None and SiglipVisionModel is None:
            raise ImportError(
                "transformers.SiglipVisionModel is required when "
                "pretrained_vision_model is set; please install/upgrade transformers."
            )
        if pretrained_language_embeddings is not None and AutoModelForCausalLM is None:
            raise ImportError(
                "transformers.AutoModelForCausalLM is required when "
                "pretrained_language_embeddings is set; please install/upgrade transformers."
            )
        if use_adarms is None:
            use_adarms = [False, False]
        nn.Module.__init__(self)
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.pretrained_vision_model = pretrained_vision_model
        self.pretrained_language_embeddings = pretrained_language_embeddings

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = precision
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None

        vision_config = vlm_config_hf.vision_config
        vision_config.image_size = image_size
        vision_config.hidden_size = architecture.vision_hidden_size
        vision_config.intermediate_size = architecture.vision_intermediate_size
        vision_config.num_hidden_layers = architecture.vision_num_hidden_layers
        vision_config.num_attention_heads = architecture.vision_num_attention_heads
        vision_config.patch_size = architecture.vision_patch_size
        vision_config.projection_dim = vlm_config.width
        vision_config.projector_hidden_act = "gelu_fast"
        vision_config.dtype = precision

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype=precision,
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        _log_mem("TinyPaliGemmaWithExpert: before paligemma init")
        self.paligemma = PaliGemmaDirect(config=vlm_config_hf)
        _log_mem("TinyPaliGemmaWithExpert: after paligemma init")

        if pretrained_vision_model is not None:
            self._swap_in_pretrained_vision_tower(
                vlm_config_hf=vlm_config_hf,
                pretrained_vision_model=pretrained_vision_model,
                expected_image_size=image_size,
            )
            _log_mem(
                f"TinyPaliGemmaWithExpert: after pretrained vision swap ({pretrained_vision_model})"
            )

        if pretrained_language_embeddings is not None:
            self._swap_in_pretrained_language_embeddings(
                vlm_config_hf=vlm_config_hf,
                pretrained_language_embeddings=pretrained_language_embeddings,
            )
            _log_mem(
                f"TinyPaliGemmaWithExpert: after pretrained embed_tokens swap "
                f"({pretrained_language_embeddings})"
            )

        self.gemma_expert = ActionExpertDirect(config=action_expert_config_hf)
        _log_mem("TinyPaliGemmaWithExpert: after gemma_expert init")
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def _swap_in_pretrained_vision_tower(
        self,
        *,
        vlm_config_hf,
        pretrained_vision_model: str,
        expected_image_size: int,
    ) -> None:
        """Replace the random vision tower with a pretrained SigLIP and rebuild the projector.

        The PaliGemma vision_tower slot is API-compatible with HF's
        ``SiglipVisionModel`` (same ``last_hidden_state`` semantics), so the only
        downstream change required is swapping the linear projector to accept
        the pretrained SigLIP's ``hidden_size`` instead of the random tower's.
        """
        loaded = SiglipVisionModel.from_pretrained(pretrained_vision_model)
        loaded_cfg = loaded.config

        if loaded_cfg.image_size != expected_image_size:
            raise ValueError(
                f"pretrained_vision_model={pretrained_vision_model!r} expects "
                f"image_size={loaded_cfg.image_size}, but TinyPI05Config.image_resolution "
                f"is {expected_image_size}. Either change the checkpoint or update "
                f"image_resolution to match."
            )

        self.paligemma.model.vision_tower = loaded

        # Update vision_config in-place so the rebuilt projector picks the right
        # input dim (projection_dim was already set to vlm_width above).
        vlm_config_hf.vision_config.hidden_size = loaded_cfg.hidden_size
        vlm_config_hf.vision_config.intermediate_size = loaded_cfg.intermediate_size
        vlm_config_hf.vision_config.num_hidden_layers = loaded_cfg.num_hidden_layers
        vlm_config_hf.vision_config.num_attention_heads = loaded_cfg.num_attention_heads
        vlm_config_hf.vision_config.image_size = loaded_cfg.image_size
        vlm_config_hf.vision_config.patch_size = loaded_cfg.patch_size

        self.paligemma.model.multi_modal_projector = PaliGemmaMultiModalProjector(vlm_config_hf)

    def _swap_in_pretrained_language_embeddings(
        self,
        *,
        vlm_config_hf,
        pretrained_language_embeddings: str,
    ) -> None:
        """Replace the random `embed_tokens` matrix with one from a pretrained CausalLM.

        Only the token embedding matrix is bootstrapped -- the transformer
        blocks remain random.  This is the cheap variant of "load a small LLM"
        described in the configuration docstring: the joint-attention kernel
        does not need to know anything about the source architecture, but the
        source's ``hidden_size`` must equal the random transformer's
        ``vlm_width`` (so the embedding output dim matches the rest of the
        random VLM stack).

        We also resize the language model's effective vocab to the source's so
        that the matched HF tokenizer (typically the source repo's) can encode
        the full ID range.
        """
        loaded = AutoModelForCausalLM.from_pretrained(pretrained_language_embeddings)
        # Walk the loaded model to find its `embed_tokens` regardless of wrapper class.
        loaded_inner = getattr(loaded, "model", loaded)
        loaded_embed = getattr(loaded_inner, "embed_tokens", None)
        if loaded_embed is None:
            raise ValueError(
                f"Could not locate `embed_tokens` on {pretrained_language_embeddings!r} "
                f"({type(loaded).__name__}); unsupported architecture."
            )

        loaded_vocab, loaded_hidden = loaded_embed.weight.shape
        expected_hidden = vlm_config_hf.text_config.hidden_size
        if loaded_hidden != expected_hidden:
            raise ValueError(
                f"pretrained_language_embeddings={pretrained_language_embeddings!r} has "
                f"hidden_size={loaded_hidden}, but vlm_width is {expected_hidden}. "
                f"Use an architecture preset whose vlm_width matches the source "
                f"hidden_size (e.g. `gemma3_270m_emb` for google/gemma-3-270m)."
            )

        # Detach the embed module from the source model so we don't keep the
        # rest of its weights resident in memory.
        loaded_embed_module = nn.Embedding(
            num_embeddings=loaded_vocab,
            embedding_dim=loaded_hidden,
            padding_idx=loaded_embed.padding_idx,
        )
        with torch.no_grad():
            loaded_embed_module.weight.copy_(loaded_embed.weight)
        loaded_embed_module.weight.requires_grad = True

        self.paligemma.model.language_model.embed_tokens = loaded_embed_module
        vlm_config_hf.text_config.vocab_size = loaded_vocab

        del loaded
        del loaded_inner
        del loaded_embed


class TinyPI05Pytorch(PiStar06Pytorch):
    """Core scaled-down PI0.5 model."""

    def __init__(self, config: TinyPI05Config, rtc_processor=None):
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = rtc_processor

        architecture = config.resolved_architecture()
        paligemma_config = _to_gemma_variant(
            width=architecture.vlm_width,
            depth=architecture.vlm_depth,
            mlp_dim=architecture.vlm_mlp_dim,
            num_heads=architecture.vlm_num_heads,
            num_kv_heads=architecture.vlm_num_kv_heads,
            head_dim=architecture.vlm_head_dim,
        )
        action_expert_config = _to_gemma_variant(
            width=architecture.expert_width,
            depth=architecture.expert_depth,
            mlp_dim=architecture.expert_mlp_dim,
            num_heads=architecture.expert_num_heads,
            num_kv_heads=architecture.expert_num_kv_heads,
            head_dim=architecture.expert_head_dim,
        )

        self.paligemma_with_expert = TinyPaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            architecture=architecture,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            pretrained_vision_model=config.pretrained_vision_model,
            pretrained_language_embeddings=config.pretrained_language_embeddings,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)


class TinyPI05Policy(PI05Policy):
    """Scaled-down PI0.5 policy for LeRobot."""

    config_class = TinyPI05Config
    name = "tinypi05"

    def __init__(
        self,
        config: TinyPI05Config,
        **kwargs,
    ):
        super(PI05Policy, self).__init__(config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = TinyPI05Pytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()
        self._log_parameter_counts()
        self._verify_trainable_components()

    def _log_parameter_counts(self) -> None:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        logging.info(
            "TinyPI05 parameters: total=%s (%d), trainable=%s (%d)",
            _format_param_count(total),
            total,
            _format_param_count(trainable),
            trainable,
        )

        components = {
            "vision": self.model.paligemma_with_expert.paligemma.model.vision_tower,
            "vlm_language": self.model.paligemma_with_expert.paligemma.model.language_model,
            "action_expert": self.model.paligemma_with_expert.gemma_expert,
            "action_projections": nn.ModuleList(
                [
                    self.model.action_in_proj,
                    self.model.action_out_proj,
                    self.model.time_mlp_in,
                    self.model.time_mlp_out,
                ]
            ),
        }
        for name, module in components.items():
            component_total = sum(param.numel() for param in module.parameters())
            component_trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
            logging.info(
                "TinyPI05 %s parameters: total=%s (%d), trainable=%s (%d)",
                name,
                _format_param_count(component_total),
                component_total,
                _format_param_count(component_trainable),
                component_trainable,
            )

    def get_optim_params(self) -> list[dict]:
        """Per-group learning rates so pretrained components aren't blown away.

        Mirrors ACT's `optimizer_lr_backbone` pattern, generalized to
        independently slice off the (pretrained) vision tower and the
        (pretrained) language embed_tokens, each with its own LR.  Anything
        that does not need a custom LR falls into the "default" group.

        Falls back to a single flat list when no custom LR is requested.
        """
        vision_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower"
        embed_prefix = (
            "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens"
        )

        use_vision_lr = (
            self.config.optimizer_lr_vision is not None
            and not self.config.freeze_vision_encoder
            and not self.config.train_expert_only
        )
        use_embed_lr = (
            self.config.optimizer_lr_language_embeddings is not None
            and self.config.pretrained_language_embeddings is not None
            and not self.config.train_expert_only
        )

        if not use_vision_lr and not use_embed_lr:
            return list(self.parameters())

        vision_params: list[nn.Parameter] = []
        embed_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if use_embed_lr and name.startswith(embed_prefix):
                embed_params.append(param)
            elif use_vision_lr and name.startswith(vision_prefix):
                vision_params.append(param)
            else:
                other_params.append(param)

        groups: list[dict] = [{"params": other_params}]
        if use_vision_lr and vision_params:
            groups.append({"params": vision_params, "lr": self.config.optimizer_lr_vision})
        if use_embed_lr and embed_params:
            groups.append(
                {"params": embed_params, "lr": self.config.optimizer_lr_language_embeddings}
            )
        return groups

    def _verify_trainable_components(self) -> None:
        if self.config.freeze_vision_encoder or self.config.train_expert_only:
            return

        components = {
            "vision encoder": self.model.paligemma_with_expert.paligemma.model.vision_tower,
            "VLM language model": self.model.paligemma_with_expert.paligemma.model.language_model,
            "action expert": self.model.paligemma_with_expert.gemma_expert,
            "action input projection": self.model.action_in_proj,
            "action output projection": self.model.action_out_proj,
            "time MLP input": self.model.time_mlp_in,
            "time MLP output": self.model.time_mlp_out,
        }
        frozen_components = [
            name
            for name, module in components.items()
            if not any(param.requires_grad for param in module.parameters())
        ]
        if frozen_components:
            joined = ", ".join(frozen_components)
            raise ValueError(f"TinyPI05 expected all major components to be trainable, but these are frozen: {joined}")

    def _get_default_peft_targets(self) -> dict[str, any]:
        common_projections = "action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
        target_modules = (
            rf"(model\.paligemma_with_expert\.gemma_expert\..*\.self_attn\.(q|v)_proj|"
            rf"model\.({common_projections}))"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
