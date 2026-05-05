import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Literal
import math
import copy
from transformers.models.gemma import modeling_gemma

from lerobot.policies.pi_gemma import (
    PiGemmaRMSNorm,
    _gated_residual,
    _get_pi_gemma_decoder_layer_base,
)
from lerobot.policies.pi05_full.configuration_pi05 import PI05FullConfig
from lerobot.optim.optimizers import AdamWConfig, MultiAdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.pi05_full.modeling_pi05 import PI05FullPolicy, PI05Pytorch, get_gemma_config, create_sinusoidal_pos_embedding
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05_full.processor_pi05 import Pi05FullPrepareStateTokenizerProcessorStep
from lerobot.processor import TokenizerProcessorStep
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE, 
    OBS_LANGUAGE_TOKENS, 
    OBS_LANGUAGE_ATTENTION_MASK, 
    ACTION,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    ACTION_TOKENS,
    ACTION_TOKEN_MASK,
)
        

from lerobot.policies.pi05_full.modeling_pi05 import make_att_2d_masks
        


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2

@dataclass
class ConcurrencyConfig:
    actor: str = "threads"
    learner: str = "threads"

@dataclass
class VisionEncoderTrainConfig:
    vision_tower: int | None = None      # None=frozen, int=train from this layer index onwards
    multi_modal_projector: bool = False  # True=train all, False=frozen

@dataclass
class TrainableParamsConfig:
    vision_encoder_from_layer: VisionEncoderTrainConfig = field(default_factory=VisionEncoderTrainConfig)
    language_from_layer: int | None = None         # None=frozen, int=train layers >= this index
    critic_language_from_layer: int | None = None  # same; critic norm/value_head/queries always on

@PreTrainedConfig.register_subclass("pi05_rl")
@dataclass
class PI05RLConfig(PI05FullConfig):
    # RL parameters
    task: str = ""
    # Active action dimension (unpadded). Used to slice replay-buffer actions
    # back down from `max_action_dim` (e.g. 32) to the robot's true DoF.
    action_dim: int = 6
    drop_n_last_frames: int = 2  # Drop the last n frames from the replay buffer
    critic_target_update_weight: float = 0.005
    num_critics: int = 1
    discount: float = 0.97
    
    # Reward parameters
    reward_normalization_constant: float = 1.0
    terminal_failure_reward: float = -10.0
    
    # Training parameter
    online_steps: int = 1000000
    online_buffer_capacity: int = 100000
    offline_buffer_capacity: int = 100000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    policy_update_freq: int = 1
    critic_warmup_steps: int = 0
    grad_clip_norm: float = 40.0
    gradient_accumulation_steps: int = 1
    
    # Learning rates
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    
    # UTD
    utd_ratio: int = 1
    
    # Device configuration
    actor_device: str | None = None
    learner_device: str | None = None

    # Critic parameters
    use_separate_critic: bool = True
    critic_llm_depth: int = 6
    # Critic network arguments
    critic_network_kwargs: dict | None = None

    # Trainable parameter configuration
    trainable_params: TrainableParamsConfig = field(default_factory=TrainableParamsConfig)

    # Offline training steps
    offline_steps: int = 10000

    # Inference parameters
    num_inference_steps: int = 5
    
    # Advantage parameters
    inference_advantage: float = 1.0
    advantage_scaling: float = 1.0
    
    # Checkpoint
    pi05_checkpoint: str | None = None
    
    # Dataset stats (inherited from PreTrainedConfig? No, SAC defines it explicitly)
    action_encoding_stats_path: str | None = None  # Path to precomputed action encoding stats (.pt)
    dataset_stats: dict | None = None
    
    # Storage device
    storage_device: str = "cpu"
    
    # Shared encoder (required by learner.py logic, set to False for Pi05)
    shared_encoder: bool = False
    
    # SAC compatibility (required by learner.py logic)
    num_discrete_actions: int | None = None
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = False
    
    # Actor-Learner
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=self.optimizer_weight_decay,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

from transformers import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE

# Hardcoded vocabulary size matching PaliGemmaWithExpertModel in modeling_pi05.py
PALIGEMMA_VOCAB_SIZE = 257152
PALIGEMMA_PAD_TOKEN_ID = 0 # Standard for Gemma/PaliGemma

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class Pi05TransformerCritic(nn.Module):
    """Transformer-based Value Critic using Gemma architecture.
    
    Features:
    - Separate 6-layer Gemma LLM (configurable).
    - Uses shared vision features from the actor (detached).
    - Has its own embedding layer (initialized from actor).
    - Uses multiple value query tokens (8) to avoid bottlenecks.
    - Input: Vision features + Text tokens (without advantage).
    """
    def __init__(self, config: PI05RLConfig):
        super().__init__()
        self.config = config
        self.dtype = getattr(torch, config.dtype) if hasattr(torch, config.dtype) else torch.float32
        
        # Get Gemma config to match actor architecture
        paligemma_config = get_gemma_config(config.paligemma_variant)
        hidden_dim = paligemma_config.width
        vocab_size = PALIGEMMA_VOCAB_SIZE
        
        # Configurable dimensions
        # Default to 6 layers as per PI06 design
        num_layers = getattr(config, "critic_llm_depth", 6)
        mlp_dim = paligemma_config.mlp_dim
        
        critic_gemma_config = CONFIG_MAPPING["gemma"](
            head_dim=256,
            hidden_size=hidden_dim,
            intermediate_size=mlp_dim,
            num_attention_heads=8,
            num_hidden_layers=num_layers,
            num_key_value_heads=1,
            vocab_size=vocab_size,
            hidden_activation="gelu_pytorch_tanh",
            dtype=self.dtype,
            use_adarms=False,
        )
        self.critic_gemma_config = critic_gemma_config
        
        # Learned query tokens for value prediction (32 tokens)
        # Initialize with magnitude similar to scaled text embeddings (~270)
        # Text norm ≈ sqrt(hidden_dim) * embedding_norm ≈ 45 * 6 ≈ 270
        # So we init queries with std ≈ 270 / sqrt(hidden_dim) ≈ 6
        self.num_query_tokens = 32
        query_init_std = 1.0  # Standard initialization
        self.value_queries = nn.Parameter(torch.randn(1, self.num_query_tokens, hidden_dim) * query_init_std)
        # Force contiguous gradients
        self.value_queries.register_hook(lambda grad: grad.contiguous())
        
        # Rotary Embeddings
        self.rotary_emb = GemmaRotaryEmbedding(critic_gemma_config)

        # Transformer layers — PiGemma so the bare init stays compatible with the actor's
        # adaRMS / gated-residual layers that initialize_weights_from_actor copies in.
        pi_gemma_decoder_layer_base = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList([
            pi_gemma_decoder_layer_base(critic_gemma_config, layer_idx=i)
            for i in range(num_layers)
        ])

        # Final normalization
        self.norm = PiGemmaRMSNorm(hidden_dim, eps=critic_gemma_config.rms_norm_eps)
        
        # Token-wise projection to reduce dimensionality before flattening
        self.token_proj = nn.Linear(hidden_dim, 512)
        
        # Value head (projects from flattened projected tokens -> 1 value)
        self.value_head = nn.Sequential(
            nn.Linear(512 * self.num_query_tokens, 512 * 2),
            SwiGLU(),
            nn.Linear(512, 1)
        )
        
        # Vision components (initialized in initialize_weights_from_actor)
        self.vision_tower = None
        self.multi_modal_projector = None
        
    def initialize_weights_from_actor(self, actor_model):
        """Initialize critic weights from the actor's pretrained weights."""
        # Access the underlying PaliGemma model
        if hasattr(actor_model, "paligemma_with_expert"):
             paligemma_model = actor_model.paligemma_with_expert.paligemma
        elif hasattr(actor_model, "paligemma"):
             paligemma_model = actor_model.paligemma
        else:
             raise ValueError(f"Could not find paligemma in actor model of type {type(actor_model)}")
             
        source_model = paligemma_model.model.language_model

        # Rotary embeddings are tiny and needed locally, deepcopy to avoid EMA bug
        self.rotary_emb = copy.deepcopy(source_model.rotary_emb)
        
        # Copy Layers (Deepcopy to ensure we get the exact same class and weights)
        num_critic_layers = self.critic_gemma_config.num_hidden_layers
        self.layers = nn.ModuleList([copy.deepcopy(source_model.layers[i]) for i in range(num_critic_layers)])
        
        # Copy Norm
        self.norm = copy.deepcopy(source_model.norm)
        
        # Copy Vision Components and truncate layers to 13
        self.vision_tower = copy.deepcopy(paligemma_model.model.vision_tower)
        if hasattr(self.vision_tower, 'vision_model') and hasattr(self.vision_tower.vision_model, 'encoder'):
            self.vision_tower.vision_model.encoder.layers = self.vision_tower.vision_model.encoder.layers[:13]
        self.multi_modal_projector = copy.deepcopy(paligemma_model.model.multi_modal_projector)

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed images using the critic's own vision tower and projector."""
        if self.vision_tower is None or self.multi_modal_projector is None:
            raise RuntimeError("Vision components not initialized. Call initialize_weights_from_actor first.")
        image_outputs = self.vision_tower(image, return_dict=True)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    def forward(self, vision_features: Tensor, text_embs: Tensor, token_masks: Tensor) -> Tensor:
        """
        Args:
            vision_features: [B, num_patches, hidden_dim]
            text_embs: [B, seq_len, hidden_dim]
            token_masks: [B, seq_len]
        Returns:
            value: [B, 1]
        """
        
        batch_size = text_embs.shape[0]
        
        # Expand queries
        queries = self.value_queries.repeat(batch_size, 1, 1) # [B, num_queries, D]
        
        # Concatenate: [Vision, Text, Queries]
        hidden_states = torch.cat([vision_features, text_embs, queries], dim=1)
        
        # Create Attention Mask
        vision_len = vision_features.shape[1]
        vision_mask = torch.ones(batch_size, vision_len, dtype=torch.bool, device=text_embs.device)
        query_mask = torch.ones(batch_size, self.num_query_tokens, dtype=torch.bool, device=text_embs.device)
        
        # Full mask: [Vision, Text, Queries]
        full_mask = torch.cat([vision_mask, token_masks, query_mask], dim=1)
        
        # Create 4D attention mask
        full_seq_len = full_mask.shape[1]
        attention_mask = full_mask[:, None, None, :].expand(batch_size, 1, full_seq_len, full_seq_len)
        attention_mask = torch.where(attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        
        # Position IDs
        position_ids = torch.cumsum(full_mask, dim=1) - 1
        
        # Rotary Embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        # Apply layers manually to support gated residuals from pretrained weights
        for i, layer in enumerate(self.layers):
            # Input Norm
            norm_out = layer.input_layernorm(hidden_states, cond=None)
            if isinstance(norm_out, tuple):
                hidden_states_norm, gate = norm_out
            else:
                hidden_states_norm, gate = norm_out, None

            # Attention Projections
            input_shape = hidden_states_norm.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            
            query_states = layer.self_attn.q_proj(hidden_states_norm).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(hidden_states_norm).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(hidden_states_norm).view(hidden_shape).transpose(1, 2)
            
            # Apply RoPE
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )
            
            # Attention Forward
            att_output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                layer.self_attn.scaling,
            )
            
            # Reshape back and Project
            att_output = att_output.reshape(batch_size, -1, self.critic_gemma_config.hidden_size)
            out_emb = layer.self_attn.o_proj(att_output)
            
            # First Residual (Gated)
            if gate is not None:
                out_emb = _gated_residual(hidden_states, out_emb, gate)
            else:
                out_emb = hidden_states + out_emb
                
            after_first_residual = out_emb.clone()
            
            # Post Attention Norm
            norm_out = layer.post_attention_layernorm(out_emb, cond=None)
            if isinstance(norm_out, tuple):
                out_emb, gate = norm_out
            else:
                out_emb, gate = norm_out, None
                
            # MLP
            out_emb = layer.mlp(out_emb)
            
            # Second Residual (Gated)
            if gate is not None:
                hidden_states = _gated_residual(after_first_residual, out_emb, gate)
            else:
                hidden_states = after_first_residual + out_emb
            
        # Final Norm
        hidden_states = self.norm(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # Extract Queries (At the END)
        start_idx = vision_features.shape[1] + text_embs.shape[1]
        queries_out = hidden_states[:, start_idx:, :] # [B, num_queries, D]
        
        # Token-wise projection (applied to each of the 32 tokens independently)
        queries_proj = self.token_proj(queries_out) # [B, num_queries, 256]
        
        # Flatten all projected tokens together
        queries_flat = queries_proj.reshape(batch_size, -1) # [B, num_queries * 256]

        # Value Head
        value = self.value_head(queries_flat.to(self.dtype))

        return value

class PI05RLPytorch(PI05Pytorch):
    """Subclass of PI05Pytorch to inject Advantage Conditioning."""
    
    def __init__(self, config: PI05RLConfig, rtc_processor=None):
        super().__init__(config, rtc_processor)
        self.log_counter = 0
        self.suppress_debug_log = False
        
    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep."""
        # Call parent to get standard embeddings and time embedding (as adarms_cond)
        return super().embed_suffix(noisy_actions, timestep)

    def forward(
        self,
        images,
        img_masks,
        high_level_task_tokens,
        high_level_task_masks,
        subtask_tokens,
        subtask_masks,
        action_tokens,
        action_masks,
        actions,
        noise=None,
        time=None,
        prefix_embs=None,
        prefix_pad_masks=None,
        prefix_att_masks=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Fix for BFloat16/Float32 mismatch: Ensure noise and time match action dtype
        if actions.dtype == torch.bfloat16:
            if noise.dtype != torch.bfloat16:
                noise = noise.to(dtype=torch.bfloat16)
            if time.dtype != torch.bfloat16:
                time = time.to(dtype=torch.bfloat16)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if prefix_embs is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks, image_len = self.embed_prefix(
                images, img_masks, high_level_task_tokens, subtask_tokens, high_level_task_masks, subtask_masks, action_tokens, action_masks
            )
        else:
            # Assume prefix_embs already contains everything needed
            # We need image_len for attention mask creation and loss calculation indexing.
            # We calculate it by subtracting other known lengths from the total prefix length.
            prefix_len = prefix_embs.shape[1]
            task_len_temp = high_level_task_tokens.shape[1] if high_level_task_tokens is not None else 0
            subtask_len_temp = subtask_tokens.shape[1] if subtask_tokens is not None else 0
            fast_len_temp = action_tokens.shape[1] if action_tokens is not None else 0
            
            image_len = prefix_len - task_len_temp - subtask_len_temp - fast_len_temp
            
            # The new forward constructs "combined_att_2d_masks" using this image_len implicitly (via previous logic if we were to reuse it)
            # But here we just needed image_len for the slicing below.


        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            if prefix_embs.dtype != torch.bfloat16:
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Re-create attention masks logic from PI05Pytorch.forward (new)
        # But adapted to use provided prefix_att_masks
        
        suffix_att_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        
        bsize = prefix_embs.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        total_len = prefix_len + suffix_len
        
        combined_att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=prefix_embs.device)
        
        # top-left: prefix attends to prefix
        combined_att_2d_masks[:, :prefix_len, :prefix_len] = prefix_att_masks
        
        # bottom-right: suffix attends to suffix
        combined_att_2d_masks[:, prefix_len:, prefix_len:] = suffix_att_masks
        
        # bottom-left: suffix attends to prefix EXCEPT FAST tokens
        # We need to know where FAST tokens are.
        # If we passed None for action_tokens, fast_len is 0.
        # If we passed pre-computed embeddings, we don't easily know fast_len.
        # However, for RL, we likely don't use FAST tokens (action_tokens=None).
        # So we assume suffix attends to ALL prefix.
        if action_tokens is not None:
            fast_len = action_tokens.shape[1]
            prefix_without_fast_len = prefix_len - fast_len
        else:
            prefix_without_fast_len = prefix_len
            
        combined_att_2d_masks[:, prefix_len:, :prefix_without_fast_len] = True
        
        combined_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        pad_2d_masks = combined_pad_masks[:, None, :] * combined_pad_masks[:, :, None]
        att_2d_masks = combined_att_2d_masks & pad_2d_masks
        
        
        position_ids = torch.cumsum(combined_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # Calculate losses
        # Flow loss
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype) # Ensure float32 if proj is float32

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # Only consider the unpadded action_dim entries; the rest are zero-padding.
        action_dim = self.config.action_dim
        flow_loss = F.mse_loss(u_t[:, :, :action_dim], v_t[:, :, :action_dim], reduction="none")

        flow_loss_mean = flow_loss.mean()
        
        subtask_ce_loss = torch.tensor(0.0, device=actions.device)
        action_ce_loss = torch.tensor(0.0, device=actions.device)

        # We need to calculate lengths manually as embed_prefix no longer returns them
        subtask_len = subtask_tokens.shape[1] if subtask_tokens is not None else 0
        fast_len = action_tokens.shape[1] if action_tokens is not None else 0

        # Note: We rely on prefix_embs having [Image, Task, Subtask, FastAction] structure.

        # Check if we need to compute logits (if either subtask or action tokens are present)
        if subtask_tokens is not None or action_tokens is not None:
            # Reuse prefix_out from the joint forward above. The prefix block of
            # combined_att_2d_masks is exactly prefix_att_masks (top-left quadrant in
            # lines 464-467), so prefix_out is mathematically equivalent to running the
            # prefix alone through the language model. Mirrors pi05_full at
            # policies/pi05_full/modeling_pi05.py:1213.
            logits = self.paligemma_with_expert.paligemma.lm_head(prefix_out)
            # Calculate Subtask Loss
            if subtask_tokens is not None:
                task_len = high_level_task_tokens.shape[1]
                # Indices: Image + Task -> Subtask
                
                start_idx = image_len + task_len
                end_idx = start_idx + subtask_len
                 
                 # Logits range: start_idx-1 to end_idx-1
                subtask_logits = logits[:, start_idx - 1 : end_idx - 1, :].contiguous()
                subtask_labels = subtask_tokens.contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                st_loss = loss_fct(subtask_logits.view(-1, PALIGEMMA_VOCAB_SIZE), subtask_labels.view(-1))
                 
                 # Reshape and mask
                st_loss = st_loss.view(subtask_labels.shape)
                if subtask_masks is not None:
                    st_loss = st_loss * subtask_masks
                    subtask_ce_loss = st_loss.sum() / (subtask_masks.sum() + 1e-6)
                else:
                    subtask_ce_loss = st_loss.mean()

             # Calculate Fast Action Loss
            if action_tokens is not None:
                task_len = high_level_task_tokens.shape[1]
                prev_len = image_len + task_len + subtask_len
                 
                start_idx = prev_len
                end_idx = start_idx + fast_len
                 
                 # Logits range: start_idx-1 to end_idx-1
                action_logits = logits[:, start_idx - 1 : end_idx - 1, :].contiguous()
                action_labels = action_tokens.contiguous()
                 
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                act_loss = loss_fct(action_logits.view(-1, PALIGEMMA_VOCAB_SIZE), action_labels.view(-1))
                
                # Reshape to [B, T] to match mask
                act_loss = act_loss.view(action_labels.shape)

                if action_masks is not None:
                    act_loss = act_loss * action_masks
                    action_ce_loss = act_loss.sum() / (action_masks.sum() + 1e-6)
                else:
                    action_ce_loss = act_loss.mean()

        # Weighted Sum

        total_loss = (
            self.config.loss_weight_flow * flow_loss.mean()
            + self.config.loss_weight_action_ce * action_ce_loss
            + self.config.loss_weight_subtask_ce * subtask_ce_loss
        )

        self.log_counter += 1
        if self.log_counter % 120 == 0 and not self.suppress_debug_log:
            print(f"[Actor Loss] Total: {total_loss.item():.4f} | Flow: {flow_loss.mean().item():.4f} | ActionCE: {action_ce_loss.item():.4f} | SubtaskCE: {subtask_ce_loss.item():.4f}")
            if subtask_tokens is not None:
                pred_ids = subtask_logits.argmax(dim=-1)  # [B, subtask_len]
                for i in range(min(2, pred_ids.shape[0])):
                    tgt  = self._paligemma_tokenizer.decode(subtask_labels[i], skip_special_tokens=True).strip()
                    pred = self._paligemma_tokenizer.decode(pred_ids[i], skip_special_tokens=True).strip()
                    print(f"  [subtask {i}] target: {tgt!r}  pred: {pred!r}")
            if action_tokens is not None:
                act_pred_ids = action_logits.argmax(dim=-1)  # [B, fast_len]
                for i in range(min(2, act_pred_ids.shape[0])):
                    mask_i = action_masks[i].bool() if action_masks is not None else slice(None)
                    tgt  = action_labels[i][mask_i].tolist()
                    pred = act_pred_ids[i][mask_i].tolist()
                    pairs = list(zip(tgt, pred))
                    print(f"  [action  {i}] (target,pred): {pairs}")

        return {
            "loss_actor": total_loss,
            "flow_mse_loss": flow_loss.mean(),
            "flow_loss_raw": flow_loss.detach(),
            "flow_time": time.detach(),
            "action_ce_loss": action_ce_loss,
            "subtask_ce_loss": subtask_ce_loss,
        }



    def sample_actions(self, images, img_masks, high_level_task_tokens, high_level_task_masks, subtask_tokens, subtask_masks, noise=None, num_steps=None, **kwargs) -> Tensor:
        """Sample actions."""
        # Just call super but handle None for subtasks if needed
        # The base implementation handles None for subtasks/fast tokens in embed_prefix
        return super().sample_actions(
            images, img_masks, high_level_task_tokens, high_level_task_masks, 
            subtask_tokens, subtask_masks, noise=noise, num_steps=num_steps, **kwargs
        )

class PI05RLPolicy(PI05FullPolicy):
    config_class = PI05RLConfig

    def __init__(self, config: PI05RLConfig):
        # We need to initialize the base class but swap the model for our RL version
        # PreTrainedPolicy.__init__ calls self.model = ...
        # So we override __init__ to use PI05RLPytorch
        
        # Skip PI05FullPolicy.__init__ to avoid creating PI05Pytorch
        # Call PreTrainedPolicy.__init__ directly
        super(PI05FullPolicy, self).__init__(config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        # Use our subclassed model
        self.model = PI05RLPytorch(config, rtc_processor=self.rtc_processor)
        
        self.critic_log_counter = 0

        # --- RL Integration ---
        # Initialize Value Critic
        if config.use_separate_critic:
            self.critic = Pi05TransformerCritic(config)
            self.critic_target = Pi05TransformerCritic(config)
            # We will sync weights after loading the actor (if pretrained)
            # or initialize from actor now if not loading from checkpoint
            
            # Note: Device placement is handled by the caller
            # Do not call self.critic.to(device) here
            
            self.critic_ensemble = self.critic 
        
        # Initialize Temperature (Alpha) - Unused but kept for interface
        self.actor = self.model

        # Load pretrained weights if pi05_checkpoint is specified
        if config.pi05_checkpoint:
            print(f"Loading pretrained Pi05 weights from {config.pi05_checkpoint}")
            
            # Check if it's an RL checkpoint (has critic or advantage_mlp)
            # We peek at the state dict first
            from safetensors.torch import load_file
            import os
            
            checkpoint_path = config.pi05_checkpoint
            if os.path.isdir(checkpoint_path):
                # Try to find model.safetensors or pytorch_model.bin
                if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
                    checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
                    state_dict = load_file(checkpoint_file)
                elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                    checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
                    state_dict = torch.load(checkpoint_file, map_location="cpu")
                else:
                    # Fallback to vanilla loading if no weight file found directly
                    print("No weight file found directly, falling back to vanilla loading")
                    state_dict = None
            elif os.path.isfile(checkpoint_path):
                if checkpoint_path.endswith(".safetensors"):
                    state_dict = load_file(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
            else:
                # Path looks local (contains os.sep or starts with '.') but doesn't exist — error out.
                if os.sep in checkpoint_path or checkpoint_path.startswith("."):
                    raise FileNotFoundError(
                        f"Checkpoint path '{checkpoint_path}' looks like a local path but does not exist."
                    )
                # Otherwise treat as HuggingFace Hub repo ID.
                print(f"'{checkpoint_path}' is not a local path; treating as HF Hub repo ID")
                state_dict = None

            is_rl_checkpoint = False
            if state_dict is not None:
                # Check for RL-specific keys
                if any("critic" in k for k in state_dict.keys()) or any("advantage_mlp" in k for k in state_dict.keys()):
                    is_rl_checkpoint = True
                    print("Detected RL checkpoint (contains critic/advantage layers)")
            
            if is_rl_checkpoint:
                # Load components separately to avoid aliasing issues (critic vs critic_ensemble)
                print("Loading actor and critic components separately...")
                
                actor_state_dict = {}
                critic_state_dict = {}
                
                for k, v in state_dict.items():
                    if k.startswith("actor."):
                        # Strip 'actor.' prefix for loading into self.model
                        new_key = k[6:] # len("actor.") == 6
                        actor_state_dict[new_key] = v
                    elif k.startswith("critic."):
                        # Strip 'critic.' prefix for loading into self.critic
                        new_key = k[7:] # len("critic.") == 7
                        critic_state_dict[new_key] = v
                
                # Handle tied weights for actor
                # Check for missing embed_tokens and populate from lm_head
                embed_key_suffix = "paligemma.model.language_model.embed_tokens.weight"
                lm_head_suffix = "paligemma.lm_head.weight"
                
                keys_to_add = {}
                for k, v in actor_state_dict.items():
                    if k.endswith(lm_head_suffix):
                        prefix = k[:-len(lm_head_suffix)]
                        embed_key = prefix + embed_key_suffix
                        if embed_key not in actor_state_dict:
                             print(f"Populating missing {embed_key} from {k} (tied weights)")
                             keys_to_add[embed_key] = v
                actor_state_dict.update(keys_to_add)

                # Load actor
                missing_actor, unexpected_actor = self.model.load_state_dict(actor_state_dict, strict=False)
                print(f"Actor loaded. Missing: {len(missing_actor)}, Unexpected: {len(unexpected_actor)}")
                if missing_actor:
                    print(f"Sample missing actor: {missing_actor[:5]}")
                
                # Load critic
                if hasattr(self, "critic"):
                    missing_critic, unexpected_critic = self.critic.load_state_dict(critic_state_dict, strict=False)
                    print(f"Critic loaded. Missing: {len(missing_critic)}, Unexpected: {len(unexpected_critic)}")
                    if missing_critic:
                        print(f"Sample missing critic: {missing_critic[:5]}")

                    # Sync critic_target
                    print("Syncing critic_target with loaded critic...")
                    self.critic_target.load_state_dict(self.critic.state_dict())

                # NOTE: PI05 uses external preprocessors for normalization instead of internal modules.
                print("✓ RL components loaded (Normalization is handled by external preprocessor)")
                    
            else:
                # Vanilla checkpoint loading (original logic)
                print("Loading as vanilla Pi05 checkpoint")
                # Load a vanilla PI05Policy to get the pretrained weights
                temp_policy = PI05FullPolicy.from_pretrained(
                    config.pi05_checkpoint, 
                    config=config,
                    strict=False
                )
                # Copy weights to our RL model (strict=False allows RL-specific layers to keep their init)
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    temp_policy.model.state_dict(), 
                    strict=False
                )
                
                # Initialize critic from actor weights since we don't have a trained critic
                if config.use_separate_critic:
                    print("Initializing critic from pretrained actor weights...")
                    self._init_critic_from_actor()
                
                print("✓ Pretrained Pi05 weights loaded successfully")
                print("  (Normalization is handled by external preprocessor, not policy)")
                del temp_policy  # Free memory
        else:
            # No checkpoint, initialize critic from actor (randomly initialized actor)
            if config.use_separate_critic:
                print("Initializing critic from actor weights (random init)...")
                self._init_critic_from_actor()

        # Finalize critic setup
        if config.use_separate_critic:
            # Sync target
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.requires_grad_(False)
            self.critic_target.eval()
            
            # Handle dtype
            if config.dtype == "bfloat16":
                self.critic = self.critic.to(dtype=torch.bfloat16)
                self.critic_target = self.critic_target.to(dtype=torch.bfloat16)
            elif config.dtype == "float16":
                self.critic = self.critic.to(dtype=torch.float16)
                self.critic_target = self.critic_target.to(dtype=torch.float16)

        # Ensure actor is also in the correct dtype
        if config.dtype == "bfloat16":
            self.model = self.model.to(dtype=torch.bfloat16)
        elif config.dtype == "float16":
            self.model = self.model.to(dtype=torch.float16)

        # Share embeddings to save memory
        if config.use_separate_critic:
            self._share_critic_embeddings()

        self.reset()
        
    def _init_critic_from_actor(self):
        """Initialize critic weights from the actor's pretrained weights."""
        if not hasattr(self, 'critic'):
            return
            
        # Use the helper method in Pi05TransformerCritic
        self.critic.initialize_weights_from_actor(self.model)
        
        # Initialize target critic to match
        if hasattr(self, 'critic_target'):
            # Initialize structure first
            self.critic_target.initialize_weights_from_actor(self.model)
            # Then load weights (redundant but safe)
            self.critic_target.load_state_dict(self.critic.state_dict())

    def _share_critic_embeddings(self):
        """Re-tie actor's embed_tokens.weight to lm_head.weight after loading.

        Why: PI05RLPolicy bypasses PI05FullPolicy.from_pretrained and copies
        weights via state_dict, which leaves embed_tokens and lm_head as two
        separate Parameter objects (~526M counted twice).
        """
        paligemma = self.model.paligemma_with_expert.paligemma
        if not PI05FullPolicy._tie_or_copy_language_embeddings(paligemma):
            raise RuntimeError(
                "Failed to tie embed_tokens.weight to lm_head.weight in PI05RLPolicy. "
                "Inspect PaliGemmaForConditionalGenerationWithPiGemma construction."
            )

    def get_optim_params(self) -> dict:
        params = {
            "actor": self.model.parameters(),
        }
        if hasattr(self, "critic"):
            params["critic"] = self.critic.parameters()
        return params

    def update_target_networks(self):
        """Update target critic."""
        if hasattr(self, "critic_target"):
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    if not param.requires_grad:
                        continue
                    target_param.data.mul_(1 - self.config.critic_target_update_weight)
                    target_param.data.add_(param.data * self.config.critic_target_update_weight)

    def get_vision_features(self, batch: dict[str, Tensor], use_grad: bool = True, encoder_module=None) -> tuple[Tensor, Tensor]:
        """Extract vision features and their padding masks from the batch."""
        with torch.set_grad_enabled(use_grad):
            current_batch = batch.copy()
            if "state" in batch:
                current_batch.update(batch["state"])
            images, img_masks = self._preprocess_images(current_batch)
            
            vision_features = []
            vision_pad_masks = []
            
            encoder = encoder_module if encoder_module is not None else self.model.paligemma_with_expert
            
            for img, img_mask in zip(images, img_masks):
                # img: [B, C, H, W]
                # img_mask: [B]
                feat = encoder.embed_image(img) # [B, N, D]
                vision_features.append(feat)
                
                B, N, _ = feat.shape
                # Create mask: [B, N]
                mask = img_mask[:, None].expand(B, N)
                vision_pad_masks.append(mask)
            
            vision_features = torch.cat(vision_features, dim=1)
            vision_pad_masks = torch.cat(vision_pad_masks, dim=1)
            
            return vision_features, vision_pad_masks

    def forward(
        self, 
        batch: dict[str, Tensor], 
        model: Literal["actor", "critic", "critic_value"] | None = None,
        external_metrics: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict] | dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training."""
        # --- Shared Computation: Vision Features ---
        # We always need vision features.
        # For 'actor', we need gradients.
        # For 'critic', we DO NOT need gradients for the vision encoder (it's detached).
        use_grad = (model != "critic")
        
        # Actor computes vision features inside embed_prefix; only critic/critic_value need them here.
        if model != "actor":
            vision_features, vision_pad_masks = self.get_vision_features(batch, use_grad=use_grad, encoder_module=self.critic)
            critic_vision_features = vision_features
            critic_vision_masks = vision_pad_masks

        # Actor needs tokens too
        current_batch = batch.copy()
        if "state" in batch:
            current_batch.update(batch["state"])
        actor_tokens = current_batch[OBS_LANGUAGE_TOKENS]
        actor_masks = current_batch[OBS_LANGUAGE_ATTENTION_MASK]

        # 2. Critic Tokens (No Advantage)
        critic_tokens = current_batch["critic_tokens"]
        critic_token_masks = current_batch["critic_pad_mask"] # Assuming this key
    
        # Embed critic tokens using Actor's embedding layer
        # We use the actor's embedding layer which is shared/frozen
        # We need to access it from the model
        actor_embed_layer = self.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
        critic_text_embs = actor_embed_layer(critic_tokens)
        
        # Detach embeddings to prevent critic updates from affecting the actor's language model.
        critic_text_embs = critic_text_embs.detach()


        if model != "critic":
            # Compute full prefix embeddings for actor
            # We need images and img_masks for embed_prefix, so re-extract them
            images, img_masks = self._preprocess_images(current_batch)
            
            # Extract subtask and action tokens from batch if available
            subtask_tokens = current_batch.get(OBS_LANGUAGE_SUBTASK_TOKENS, None)
            subtask_masks = current_batch.get(OBS_LANGUAGE_SUBTASK_ATTENTION_MASK, None)
            action_tokens = current_batch.get(ACTION_TOKENS, None)
            action_masks = current_batch.get(ACTION_TOKEN_MASK, None)

            prefix_embs, prefix_pad_masks, prefix_att_masks, image_len = self.model.embed_prefix(
                images, img_masks, actor_tokens, subtask_tokens, actor_masks, subtask_masks, action_tokens, action_masks
            )
            
        # --- Branching Logic ---
        
        
        if model == "actor":
            # Calculate Advantage
            # If external metrics provided, use them to avoid redundant computation
            advantage = external_metrics["advantage"]
            target_v = external_metrics.get("target_values", torch.tensor(0.0))
            current_v = external_metrics.get("critic_values", torch.tensor(0.0))
            reward = external_metrics.get("rewards", torch.tensor(0.0))

            actions = self.prepare_action(batch)
            
            # We pass precomputed prefix embeddings to avoid re-computation
            actor_output = self.model(
                images=None, 
                img_masks=None,
                high_level_task_tokens=actor_tokens,
                high_level_task_masks=actor_masks,
                subtask_tokens=subtask_tokens, # Pass captured subtask tokens
                subtask_masks=subtask_masks,
                action_tokens=action_tokens, # Pass captured action tokens
                action_masks=action_masks,
                actions=actions,
                noise=None, 
                time=None,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks
            )
            
            return {
                "loss_actor": actor_output["loss_actor"],
                "flow_mse_loss": actor_output["flow_mse_loss"],
                "flow_loss_raw": actor_output["flow_loss_raw"],
                "flow_time": actor_output["flow_time"],
                "action_ce_loss": actor_output["action_ce_loss"],
                "subtask_ce_loss": actor_output["subtask_ce_loss"],
                "advantage_mean": advantage.mean().item(),
                "target_value_mean": target_v.mean().item(),
                "reward_mean": reward.mean().item(),
                "critic_value_mean": current_v.mean().item(),
                "advantage_values": advantage,
                "critic_values": current_v,
                "target_values": target_v,
                "rewards": reward
            }
            
        elif model == "critic":
            # Critic Update
            current_v = self.critic(critic_vision_features, critic_text_embs, critic_token_masks)
            
            with torch.no_grad():
                # Next state processing (copied from actor block)
                if "next_state" in batch:
                    next_batch = batch["next_state"].copy()
                    next_images, next_img_masks = self._preprocess_images(next_batch)
                    
                    # Next vision features
                    next_vision_features = []
                    next_vision_pad_masks = []
                    for img, img_mask in zip(next_images, next_img_masks):
                        feat = self.critic_target.embed_image(img)
                        next_vision_features.append(feat)
                        B, N, _ = feat.shape
                        mask = img_mask[:, None].expand(B, N)
                        next_vision_pad_masks.append(mask)
                    next_vision_features = torch.cat(next_vision_features, dim=1).detach()
                    next_vision_pad_masks = torch.cat(next_vision_pad_masks, dim=1)
                    
                    # Next critic tokens
                    next_critic_tokens = next_batch["critic_tokens"]
                    next_critic_token_masks = next_batch["critic_pad_mask"]
                
                    # Embed next critic tokens
                    actor_embed_layer = self.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
                    next_critic_text_embs = actor_embed_layer(next_critic_tokens).detach()
                        
                    next_v = self.critic_target(next_vision_features, next_critic_text_embs, next_critic_token_masks)
                else:
                    # Handle case where next_state is missing (e.g. end of episode or not provided)
                    # For now, we assume next_state is always provided in RL batch
                    raise ValueError("next_state is required for critic update")
            
            # Loss
            reward = batch["reward"]
            done = batch["next.done"]

            # Unsqueeze to match next_v shape [B, 1]
            if reward.ndim == 1:
                reward = reward.unsqueeze(-1)
            if done.ndim == 1:
                done = done.unsqueeze(-1)
            
            # Ensure correct dtype for mixed precision training
            reward = reward.to(dtype=current_v.dtype)
            done = done.to(dtype=current_v.dtype)
            
            target_q = reward + self.config.discount * next_v * (1 - done)
            target_q = target_q.to(dtype=current_v.dtype)
            # By definition of reward structure, the values should are bounded by 0.
            # We leave a little headroom
            target_q = target_q.clamp(max=0.05)
            
            loss_critic_raw = F.mse_loss(current_v, target_q, reduction="none")
            loss_critic = loss_critic_raw.mean()
            
            self.critic_log_counter += 1
            if self.critic_log_counter % 200 == 0:
                print(f"[Critic] Loss: {loss_critic.item():.4f} | V_mean: {current_v.mean().item():.4f} | Target_mean: {target_q.mean().item():.4f} | TD_err: {torch.abs(current_v - target_q).mean().item():.4f}")
            
            # Metrics
            td_error = torch.abs(current_v - target_q)
            
            return {
                "loss_critic": loss_critic,
                "loss_critic_raw": loss_critic_raw.detach(),
                "critic_values": current_v,
                "target_values": target_q,
                "td_error": td_error,
                "td_error_mean": td_error.mean().item(),
                "critic_value_mean": current_v.mean().item(),
                "target_value_mean": target_q.mean().item(),
            }

        elif model == "critic_value":
            values = self.critic(critic_vision_features, critic_text_embs, critic_token_masks)
            return {
                "critic_values": values,
                "critic_value_mean": values.mean().item()
            }

        else:
            raise ValueError(f"Unknown model: {model}")


    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference."""
        # We need to inject advantage into the model for inference
        # The base select_action calls predict_action_chunk -> sample_actions
        # We override predict_action_chunk in PI05RLPolicy to pass it.
        return super().select_action(batch)

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict action chunk with subtask token generation.

        Uses time-based caching controlled by `config.subtask_regeneration_interval`
        (default 1s) to amortize the cost of autoregressive subtask decoding across
        the high-frequency inference loop.
        """
        # Preprocessor has already normalized and tokenized the inputs
        # Advantage is already in the tokens (passed from actor or defaulted)
        images, img_masks = self._preprocess_images(batch)

        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        import time as _time
        # --- Subtask Token Generation with Time-Based Caching ---
        current_time = _time.time()
        interval = self.config.subtask_regeneration_interval
        should_regenerate = (
            self._cached_subtask_tokens is None
            or self._last_subtask_time is None
            or interval <= 0  # 0 means regenerate every call
            or (current_time - self._last_subtask_time) >= interval
        )

        if should_regenerate:
            subtask_tokens, subtask_masks = self.model.generate_subtask_tokens(
                images, img_masks, tokens, masks,
                max_decoding_steps=self.config.tokenizer_max_length
            )
            self._cached_subtask_tokens = subtask_tokens
            self._cached_subtask_masks = subtask_masks
            self._last_subtask_time = current_time
        else:
            subtask_tokens = self._cached_subtask_tokens
            subtask_masks = self._cached_subtask_masks

        # Sample actions with subtask conditioning
        actions = self.model.sample_actions(
            images, img_masks, tokens, masks,
            subtask_tokens, subtask_masks, **kwargs
        )

        # Unpad actions to actual action dimension
        from lerobot.utils.constants import ACTION
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

