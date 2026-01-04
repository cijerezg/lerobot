import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Literal
import math

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.optim.optimizers import AdamWConfig, MultiAdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, get_gemma_config, create_sinusoidal_pos_embedding
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import TokenizerProcessorStep
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        


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

@PreTrainedConfig.register_subclass("pi05_rl")
@dataclass
class PI05RLConfig(PI05Config):
    # RL parameters
    task: str = ""
    drop_n_last_frames: int = 2  # Drop the last n frames from the replay buffer
    critic_target_update_weight: float = 0.005
    num_critics: int = 1
    discount: float = 0.97
    temperature_init: float = 1.0
    target_entropy: float | None = None
    
    # Training parameter
    online_steps: int = 1000000
    online_buffer_capacity: int = 100000
    offline_buffer_capacity: int = 100000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    policy_update_freq: int = 1
    grad_clip_norm: float = 40.0
    gradient_accumulation_steps: int = 1
    
    # Learning rates
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    temperature_lr: float = 3e-4
    
    # UTD
    utd_ratio: int = 1
    
    # Device configuration
    actor_device: str | None = None
    learner_device: str | None = None

    # Critic parameters
    use_separate_critic: bool = True
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256)
    critic_dropout: float = 0.0
    # Add critic_network_kwargs to satisfy config parser, even if we map it manually
    # Or define a nested config class. SAC uses CriticNetworkConfig.
    # Let's define it as dict or Any for simplicity, or import CriticNetworkConfig
    critic_network_kwargs: dict | None = None

    # Training constraints
    freeze_vision_tower: bool = True
    freeze_language_model: bool = True
    freeze_action_expert: bool = False
    
    # Advantage parameters
    inference_advantage: float = 1.0
    advantage_scaling: float = 1.0
    
    # Checkpoint
    pi05_checkpoint: str | None = None
    
    # Dataset stats (inherited from PreTrainedConfig? No, SAC defines it explicitly)
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
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

from transformers import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRMSNorm, GemmaRotaryEmbedding
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE

class Pi05TransformerCritic(nn.Module):
    """Transformer-based Value Critic using Gemma architecture.
    
    Takes prefix_embs from the actor and applies additional Gemma layers
    to process the sequence before predicting the value.
    """
    def __init__(self, config: PI05RLConfig):
        super().__init__()
        self.config = config
        self.dtype = getattr(torch, config.dtype) if hasattr(torch, config.dtype) else torch.float32
        
        # Get Gemma config to match actor architecture
        paligemma_config = get_gemma_config(config.paligemma_variant)
        hidden_dim = paligemma_config.width
        
        # Configurable dimensions (hardcoded for now as per user request)
        # 5 layers, hidden=2048, mlp=4096
        num_layers = 5
        mlp_dim = 4096
        
        critic_gemma_config = CONFIG_MAPPING["gemma"](
            head_dim=256,
            hidden_size=hidden_dim,
            intermediate_size=mlp_dim,
            num_attention_heads=8,
            num_hidden_layers=num_layers,
            num_key_value_heads=1,
            vocab_size=1,  # Not used
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype=self.dtype,
            use_adarms=False,
        )
        
        # Learned query token for value prediction
        self.value_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        # Force contiguous gradients to avoid DDP warnings and implicit copies
        self.value_query.register_hook(lambda grad: grad.contiguous())
        
        # Rotary Embeddings
        self.rotary_emb = GemmaRotaryEmbedding(critic_gemma_config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(critic_gemma_config, layer_idx=i)
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.norm = GemmaRMSNorm(hidden_dim, eps=critic_gemma_config.rms_norm_eps)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, prefix_embs: Tensor, prefix_pad_masks: Tensor) -> Tensor:
        """
        Args:
            prefix_embs: [B, seq_len, hidden_dim]
            prefix_pad_masks: [B, seq_len]
        Returns:
            value: [B, 1]
        """
        batch_size, seq_len, _ = prefix_embs.shape
        
        # Add value query token at the beginning
        # Use .repeat() to avoid DDP grad stride warnings (expand creates a view which can cause issues with DDP buckets)
        value_query = self.value_query.repeat(batch_size, 1, 1)
        hidden_states = torch.cat([value_query, prefix_embs], dim=1)
        
        # Update padding mask to include query token
        query_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=prefix_pad_masks.device)
        pad_mask = torch.cat([query_mask, prefix_pad_masks], dim=1)
        
        # Create attention mask
        full_seq_len = seq_len + 1
        attention_mask = pad_mask[:, None, None, :].expand(batch_size, 1, full_seq_len, full_seq_len)
        attention_mask = torch.where(attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        
        # Create position IDs
        position_ids = torch.arange(full_seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Compute Rotary Embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        
        # Apply transformer layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
        
        # Final normalization
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = self.norm(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # Extract query token output
        query_output = hidden_states[:, 0, :]
        
        # Predict value
        value = self.value_head(query_output.to(self.dtype))
        
        return value

class PI05RLPytorch(PI05Pytorch):
    """Subclass of PI05Pytorch to inject Advantage Conditioning."""
    
    def __init__(self, config: PI05RLConfig, rtc_processor=None):
        super().__init__(config, rtc_processor)
        
        action_expert_config = get_gemma_config(config.action_expert_variant)
        width = action_expert_config.width
        
        # Separate MLP for Advantage
        self.advantage_mlp_in = nn.Linear(1, width)
        self.advantage_mlp_out = nn.Linear(width, width)
        
        # Fusion MLP for Time + Advantage
        self.fusion_mlp = nn.Linear(2 * width, width)
        
    def embed_suffix(self, noisy_actions, timestep, advantage):
        """Embed noisy_actions, timestep AND advantage."""
        # Call parent to get standard embeddings and time embedding (as adarms_cond)
        embs, pad_masks, att_masks, time_emb = super().embed_suffix(noisy_actions, timestep)
        
        # --- Advantage Conditioning ---
        def advantage_mlp_func(adv):
            # Ensure advantage is [B, 1]
            if adv.dim() == 1:
                adv = adv.unsqueeze(-1)
            x = self.advantage_mlp_in(adv)
            x = F.silu(x)
            x = self.advantage_mlp_out(x)
            return F.silu(x)
            
        advantage_emb = self._apply_checkpoint(advantage_mlp_func, advantage)
        
        # Combine Time and Advantage for AdaRMS using Fusion MLP
        combined = torch.cat([time_emb, advantage_emb], dim=-1)
        adarms_cond = self.fusion_mlp(combined)
        # ------------------------------

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        advantage,
        noise=None,
        time=None,
        prefix_embs=None,
        prefix_pad_masks=None,
        prefix_att_masks=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        # Embed prefix (images + text)
        if prefix_embs is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Pass advantage to embed_suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time, advantage)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        # Helper from modeling_pi05 but we need to import it or reimplement
        # It's not exported. Let's reimplement simple version or import if possible.
        # It is not in __all__. We can access it via module.
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, OPENPI_ATTENTION_MASK_VALUE

        
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks[:, :pad_masks.shape[1]])
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")
    
    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, advantage):
        """Apply one denoising step with advantage conditioning."""
        # Embed suffix with advantage
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, advantage)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def sample_actions(self, images, img_masks, tokens, masks, advantage) -> Tensor:
        """Sample actions with advantage conditioning."""
        # Embed prefix        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
 
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Initialize flow state
        bs = images[0].shape[0]
        device = images[0].device
        
        # Start from noise
        x_t = self.sample_noise((bs, self.config.chunk_size, self.config.max_action_dim), device) * 0
        
        # Precompute Prefix KV Cache
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Euler integration
        steps = self.config.num_inference_steps
        dt = -1.0 / steps
        
        t = 1.0
        
        for _ in range(steps):
            time = torch.full((bs,), t, device=device, dtype=torch.float32)
            
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                time,
                advantage
            )
            
            # Euler step
            x_t = x_t + v_t * dt
            t += dt

            
        return x_t

class PI05RLPolicy(PI05Policy):
    config_class = PI05RLConfig

    def __init__(self, config: PI05RLConfig):
        # We need to initialize the base class but swap the model for our RL version
        # PreTrainedPolicy.__init__ calls self.model = ...
        # So we override __init__ to use PI05RLPytorch
        
        # Skip PI05Policy.__init__ to avoid creating PI05Pytorch
        # Call PreTrainedPolicy.__init__ directly
        super(PI05Policy, self).__init__(config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        # Use our subclassed model
        self.model = PI05RLPytorch(config, rtc_processor=self.rtc_processor)

        # --- RL Integration ---
        # Initialize Value Critic
        if config.use_separate_critic:
            self.critic = Pi05TransformerCritic(config)
            self.critic_target = Pi05TransformerCritic(config)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.requires_grad_(False)
            self.critic_target.eval()
            # Convert critic networks to match the policy's dtype
            # This is important because the batch data will be cast to bfloat16 in the learner
            if config.dtype == "bfloat16":
                self.critic = self.critic.to(dtype=torch.bfloat16)
                self.critic_target = self.critic_target.to(dtype=torch.bfloat16)
            elif config.dtype == "float16":
                self.critic = self.critic.to(dtype=torch.float16)
                self.critic_target = self.critic_target.to(dtype=torch.float16)
            
            # Note: Device placement is handled by the caller
            # Do not call self.critic.to(device) here
            
            self.critic_ensemble = self.critic 
        
        # Initialize Temperature (Alpha) - Unused but kept for interface
        # Device will be set when moving the whole policy
        self.log_alpha = nn.Parameter(torch.tensor([math.log(config.temperature_init)]))
        self.target_entropy = config.target_entropy
        
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
            else:
                 # It's a file
                if checkpoint_path.endswith(".safetensors"):
                    state_dict = load_file(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location="cpu")

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
                missing_critic, unexpected_critic = self.critic.load_state_dict(critic_state_dict, strict=False)
                print(f"Critic loaded. Missing: {len(missing_critic)}, Unexpected: {len(unexpected_critic)}")
                if missing_critic:
                    print(f"Sample missing critic: {missing_critic[:5]}")

                # Sync critic_target
                print("Syncing critic_target with loaded critic...")
                self.critic_target.load_state_dict(self.critic.state_dict())

                # NOTE: PI05 uses external preprocessors for normalization instead of internal modules.
                # The preprocessor is loaded separately via make_pre_post_processors() in the runner scripts.
                # So we don't expect to find 'normalize_inputs' in the state_dict here.
                print("✓ RL components loaded (Normalization is handled by external preprocessor)")
                    
            else:
                # Vanilla checkpoint loading (original logic)
                print("Loading as vanilla Pi05 checkpoint")
                # Load a vanilla PI05Policy to get the pretrained weights
                temp_policy = PI05Policy.from_pretrained(
                    config.pi05_checkpoint, 
                    config=config,
                    strict=False
                )
                # Copy weights to our RL model (strict=False allows RL-specific layers to keep their init)
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    temp_policy.model.state_dict(), 
                    strict=False
                )
                
                # NOTE: PI05 uses external preprocessor for normalization instead of internal modules.
                # The preprocessor is loaded separately via make_pre_post_processors() with the checkpoint path.
                # So there's no normalize_inputs/normalize_outputs to copy here - that's expected behavior.
                
                # Filter out expected missing keys (RL-specific layers)
                expected_missing = [
                    "advantage_mlp_in", "advantage_mlp_out", "fusion_mlp"
                ]
                actual_missing = [k for k in missing_keys if not any(exp in k for exp in expected_missing)]
                
                if actual_missing:
                    print(f"⚠ Missing keys: {actual_missing[:5]}")
                if unexpected_keys:
                    print(f"⚠ Unexpected keys: {unexpected_keys[:5]}")
                
                print("✓ Pretrained Pi05 weights loaded successfully")
                print("  (Normalization is handled by external preprocessor, not policy)")
                del temp_policy  # Free memory

        # Freeze parameters if requested
        if config.freeze_vision_tower:
            for param in self.model.paligemma_with_expert.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        
        if config.freeze_language_model:
             for param in self.model.paligemma_with_expert.paligemma.model.language_model.parameters():
                param.requires_grad = False

        self.reset()

    def get_optim_params(self) -> dict:
        params = {
            "actor": self.model.parameters(),
            "temperature": [self.log_alpha],
        }
        if hasattr(self, "critic"):
            params["critic"] = self.critic.parameters()
        return params

    def update_target_networks(self):
        """Update target critic."""
        if hasattr(self, "critic_target"):
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1 - self.config.critic_target_update_weight)
                    target_param.data.add_(param.data * self.config.critic_target_update_weight)

    def forward(
        self, 
        batch: dict[str, Tensor], 
        model: Literal["actor", "critic", "temperature", "critic_value"] | None = None
    ) -> tuple[Tensor, dict] | dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training."""
        
        if model == "temperature":
            return {"loss_temperature": torch.tensor(0.0, device=self.config.device, requires_grad=True)}

        # --- Shared Computation: Embeddings ---
        # We always need current state embeddings
        # For 'actor' and None (BC), we need gradients.
        # For 'critic', we don't need gradients for the encoder (it's detached).
        use_grad = (model != "critic")
        
        with torch.set_grad_enabled(use_grad):
            # Preprocess current observation
            # Flatten batch["state"] into a new dict so _preprocess_images can find image keys
            current_batch = batch.copy()
            if "state" in batch:
                current_batch.update(batch["state"])
            
            images, img_masks = self._preprocess_images(current_batch)
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION
            tokens = current_batch[OBS_LANGUAGE_TOKENS]
            masks = current_batch[OBS_LANGUAGE_ATTENTION_MASK]
            
            # Compute current state embeddings
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(images, img_masks, tokens, masks)

        # For RL modes (actor/critic), we also need next state embeddings for Target Value
        next_prefix_embs = None
        next_prefix_pad_masks = None
        
        if model in ["actor", "critic"]:
            with torch.no_grad():
                if "next_state" in batch:
                    next_batch = batch["next_state"].copy() # next_state is already a dict of features
                    # Ensure next_batch has all keys needed by _preprocess_images
                    
                    next_images, next_img_masks = self._preprocess_images(next_batch)
                    next_tokens = next_batch[OBS_LANGUAGE_TOKENS]
                    next_masks = next_batch[OBS_LANGUAGE_ATTENTION_MASK]
                    
                    next_prefix_embs, next_prefix_pad_masks, _ = self.model.embed_prefix(
                        next_images, next_img_masks, next_tokens, next_masks
                    )
                else:
                    # Fallback or error if next_state is missing in RL mode?
                    # Assuming it's present for RL
                    pass

        # --- Branching Logic ---
        
        if model is None:
            # BC Mode
            advantage = torch.full((batch["action"].shape[0], 1), self.config.inference_advantage, device=self.config.device)
            actions = self.prepare_action(batch)
            
            losses = self.model.forward(
                images, img_masks, tokens, masks, actions, advantage,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks
            )
            
            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[:, :, :original_action_dim]
            loss = losses.mean()
            
            loss_dict = {
                "loss": loss.item(),
                "loss_per_dim": losses.mean(dim=[0, 1]).detach().to(dtype=torch.float32).cpu().numpy().tolist(),
            }
            return loss, loss_dict

        elif model == "actor":
            # --- Actor Update ---
            # 1. Compute Advantage: A(s, a) = r + gamma * V(s') - V(s)
            
            with torch.no_grad():
                # Compute V(s) using current critic (detached input)
                current_v = self.critic(prefix_embs.detach(), prefix_pad_masks)
                
                # Compute V(s') using target critic
                next_v = self.critic_target(next_prefix_embs.detach(), next_prefix_pad_masks)
                
                reward = batch["reward"]
                if reward.ndim == 1:
                    reward = reward.unsqueeze(-1)
                    
                not_done = 1.0
                if "next.done" in batch:
                    not_done = 1.0 - batch["next.done"].float()
                    if not_done.ndim == 1:
                        not_done = not_done.unsqueeze(-1)
                
                target_v = reward + self.config.discount * next_v * not_done
                advantage = target_v - current_v
            
            # 2. Train Actor
            # Pass pre-computed embeddings
            # We need actions for the actor loss
            # In RL, actions come from the batch (behavior cloning / offline RL) 
            # or sampled? 
            # For offline RL (IQL/etc), we use batch actions.
            # For online, we use batch actions (on-policy or off-policy replay).
            actions = self.prepare_action(batch)
            actions = actions.to(dtype=torch.float32)
            advantage = advantage.to(dtype=torch.float32)
            
            
            loss_tensor = self.model.forward(
                images, 
                img_masks, 
                tokens, 
                masks, 
                actions, 
                advantage, 
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks
            )
            
            # Compute mean loss
            loss = loss_tensor.mean()
            
            # Create return dictionary with all expected metrics (matching learner expectations at lines 477-488)
            return {
                "loss_actor": loss,
                "advantage_mean": advantage.mean().item(),
                "advantage_std": advantage.std(unbiased=False).item(),
                "target_value_mean": target_v.mean().item(),
                "reward_mean": batch["reward"].mean().item(),
                "critic_value_mean": current_v.mean().item(),
                "critic_value_std": current_v.std(unbiased=False).item(),
                "q_values": current_v.mean().item(),  # Legacy compatibility
                "q_targets": target_v.mean().item(),  # Legacy compatibility
                "advantage_values": advantage.detach().cpu().flatten(),
                "critic_values": current_v.detach().cpu().flatten(),
            }

        elif model == "critic":
            # --- Critic Update ---
            # Minimize MSE(V(s), r + gamma * V(s')_target)
            
            # Compute V(s)
            current_v = self.critic(prefix_embs.detach(), prefix_pad_masks)
            
            # Compute Target
            with torch.no_grad():
                next_v = self.critic_target(next_prefix_embs.detach(), next_prefix_pad_masks)
                
                reward = batch["reward"]
                if reward.ndim == 1:
                    reward = reward.unsqueeze(-1)
                
                not_done = 1.0
                if "next.done" in batch:
                    not_done = 1.0 - batch["next.done"].float()
                    if not_done.ndim == 1:
                        not_done = not_done.unsqueeze(-1)
                        
                target = reward + self.config.discount * next_v * not_done
            
            loss_critic = F.mse_loss(current_v, target)
            
            # Collect metrics for wandb logging
            td_error = target - current_v
            return {
                "loss_critic": loss_critic,
                "critic_value_mean": current_v.mean().item(),
                "critic_value_std": current_v.std(unbiased=False).item(),
                "target_value_mean": target.mean().item(),
                "target_value_std": target.std(unbiased=False).item(),
                "td_error_mean": td_error.mean().item(),
                "td_error_std": td_error.std(unbiased=False).item(),
                "critic_values": current_v.detach().cpu().flatten(),  # For histogram logging
            }
            
        elif model == "critic_value":
            # --- Just get the critic value ---
            current_v = self.critic(prefix_embs.detach(), prefix_pad_masks)
            return {
                "critic_value_mean": current_v.mean().item(),
                "critic_value_std": current_v.std(unbiased=False).item(),
                "critic_values": current_v.detach().cpu().flatten(),
            }
            
        else:
            raise ValueError(f"Unknown model: {model}")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference."""
        # We need to inject advantage into the model for inference
        # The base select_action calls predict_action_chunk -> sample_actions
        # We need to override predict_action_chunk or sample_actions?
        # PI05Policy.predict_action_chunk calls self.model.sample_actions
        # PI05Pytorch.sample_actions is compiled.
        
        # We need to override `sample_actions` in `PI05RLPytorch` to accept advantage
        # And we need to override `predict_action_chunk` in `PI05RLPolicy` to pass it.
        
        # ... (Implementation details for inference override)
        # For now, let's assume we can pass kwargs or we need to implement it.
        # Let's implement `predict_action_chunk` in PI05RLPolicy.
        return super().select_action(batch)

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk with advantage conditioning."""
        # Preprocessor has already normalized and tokenized the inputs
        images, img_masks = self._preprocess_images(batch)
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        
        bs = images[0].shape[0]
        advantage = torch.full((bs, 1), self.config.inference_advantage, device=self.config.device)
        
        # Sample actions with advantage conditioning
        actions = self.model.sample_actions(images, img_masks, tokens, masks, advantage)
        
        # Unpad actions to actual action dimension
        from lerobot.utils.constants import ACTION
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        
        return actions

