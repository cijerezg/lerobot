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

class Pi05Critic(nn.Module):
    """Value Function Critic for Pi05 RL integration.
    
    Estimates V(s).
    Uses a lightweight CNN for images and MLP for state.
    """
    def __init__(self, config: PI05RLConfig):
        super().__init__()
        self.config = config
        
        # Get the dtype from config
        self.dtype = getattr(torch, config.dtype) if hasattr(torch, config.dtype) else torch.float32
        
        # Simple CNN for 224x224 images
        # Input: [B, 3, 224, 224]
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), # -> [32, 55, 55]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> [64, 26, 26]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> [64, 24, 24]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 256),
            nn.LayerNorm(256),
            nn.Tanh()
        )
        
        # State encoder - use actual state dimension from input features, not padded max_state_dim
        state_dim = config.input_features["observation.state"].shape[0]
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh()
        )
        
        # Value Network
        self.v_net = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # Process images
        # Find first available image key
        img_key = None
        for key in batch.keys():
            # Look for "images" (plural) in the key name, matching "observation.images.xxx"
            if "images" in key and "mask" not in key:
                img_key = key
                break
        
        if img_key:
            img = batch[img_key]
            # Convert to correct dtype and ensure correct shape [B, C, H, W]
            if img.dtype != self.dtype:
                img = img.to(self.dtype)
            if img.shape[1] != 3: # If channels last
                img = img.permute(0, 3, 1, 2)
            
            # Resize if needed (simple resize)
            if img.shape[-2:] != (224, 224):
                img = F.interpolate(img, size=(224, 224), mode='bilinear')
                
            img_emb = self.vision_encoder(img)
        else:
            # No images found - use observation.state to get device and batch size
            if "observation.state" not in batch:
                raise ValueError(
                    f"Critic received batch without images and without 'observation.state'. "
                    f"Available keys: {list(batch.keys())}"
                )
            state = batch["observation.state"]
            device = state.device
            bs = state.shape[0]
            img_emb = torch.zeros(bs, 256, device=device, dtype=self.dtype)

        # Process State  
        state = batch["observation.state"]
        if state.dtype != self.dtype:
            state = state.to(self.dtype)
            
        state_emb = self.state_encoder(state)
        
        # Combine
        import pdb; pdb.set_trace()
        combined = torch.cat([img_emb, state_emb], dim=-1)
        value = self.v_net(combined)
        
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

    def forward(self, images, img_masks, tokens, masks, actions, advantage, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        
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
        x_t = self.sample_noise((bs, self.config.chunk_size, self.config.max_action_dim), device)
        
        # Euler integration
        steps = self.config.num_inference_steps
        dt = 1.0 / steps
        
        for i in range(steps):
            t = i / steps
            time = torch.full((bs,), t, device=device, dtype=torch.float32)
            
            # Embed suffix with advantage
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time, advantage)
            
            if prefix_embs.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            
            from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
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
            v_t = self.action_out_proj(suffix_out)
            
            # Euler step
            x_t = x_t - v_t * dt
            
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

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Note: Device placement is handled by the caller (e.g., learner, actor)
        # Do not call self.model.to(device) here
        
        # --- RL Integration ---
        # Initialize Value Critic
        if config.use_separate_critic:
            self.critic = Pi05Critic(config)
            self.critic_target = Pi05Critic(config)
            self.critic_target.load_state_dict(self.critic.state_dict())
            
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
        model: Literal["actor", "critic", "temperature"] | None = None
    ) -> tuple[Tensor, dict] | dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training."""
        
        if model is None:
            # BC Mode - Use inference advantage or 0?
            # For BC, we might want to condition on "High Advantage" to clone expert behavior?
            # Or just 0 if expert data is assumed to be optimal?
            # Let's use 0 for pure BC compatibility or inference_advantage.
            # Let's use inference_advantage to be consistent.
            advantage = torch.full((batch["action"].shape[0], 1), self.config.inference_advantage, device=self.config.device)
            
            # Preprocessor has already normalized and tokenized the inputs
            images, img_masks = self._preprocess_images(batch)
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION
            tokens = batch[OBS_LANGUAGE_TOKENS]
            masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            
            actions = self.prepare_action(batch)
            
            losses = self.model.forward(images, img_masks, tokens, masks, actions, advantage)
            
            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[:, :, :original_action_dim]
            loss = losses.mean()
            
            loss_dict = {
                "loss": loss.item(),
                "loss_per_dim": losses.mean(dim=[0, 1]).detach().to(dtype=torch.float32).cpu().numpy().tolist(),
            }
            return loss, loss_dict

        elif model == "actor":
            # RL Actor Mode
            # 1. Compute Advantage
            with torch.no_grad():
                # V(s) - batch["state"] contains observation.state, observation.images.xxx, etc.
                current_v = self.critic(batch["state"])
                
                # V_target(s')
                next_v = self.critic_target(batch["next_state"])
                
                reward = batch["reward"]
                done = batch["done"]
                
                # Ensure reward and done have shape [B, 1]
                # Handle both 1D [B] and potentially wrong 2D shapes like [B, 2]
                batch_size = current_v.shape[0]
                reward = reward.reshape(batch_size, -1)
                done = done.reshape(batch_size, -1)
                
                # Take only first column if multiple columns exist
                if reward.shape[1] > 1:
                    reward = reward[:, :1]
                if done.shape[1] > 1:
                    done = done[:, :1]
                
                # A = (r + gamma * V(s')) - V(s)
                target_v = reward + (1 - done) * self.config.discount * next_v
                advantage = target_v - current_v
                
                # Scale advantage
                advantage = advantage * self.config.advantage_scaling
            
            # 2. Train Actor with Advantage Conditioning
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION
            
            # Flatten batch["state"] into a new dict so _preprocess_images can find image keys
            actor_batch = batch.copy()
            actor_batch.update(batch["state"])
            
            # Preprocessor has already tokenized the inputs
            tokens = actor_batch[OBS_LANGUAGE_TOKENS]
            masks = actor_batch[OBS_LANGUAGE_ATTENTION_MASK]

            images, img_masks = self._preprocess_images(actor_batch)
            actions = self.prepare_action(actor_batch)
            
            # Ensure actions and advantage are in float32 as expected by modeling_pi05.py
            # The model converts embeddings to bfloat16 internally when needed
            actions = actions.to(dtype=torch.float32)
            advantage = advantage.to(dtype=torch.float32)
            
            losses = self.model.forward(images, img_masks, tokens, masks, actions, advantage)
            
            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[:, :, :original_action_dim]
            loss = losses.mean()
            
            return {"loss_actor": loss}
            
        elif model == "critic":
            # Value Critic Loss
            # Target: r + gamma * V_target(s')
            # Current: V(s)

            with torch.no_grad():
                next_v = self.critic_target(batch["next_state"])
                reward = batch["reward"]
                done = batch["done"]
                
                # Ensure reward and done have shape [B, 1]
                # Handle both 1D [B] and potentially wrong 2D shapes like [B, 2]
                batch_size = next_v.shape[0]
                reward = reward.reshape(batch_size, -1)
                done = done.reshape(batch_size, -1)
                
                # Take only first column if multiple columns exist
                if reward.shape[1] > 1:
                    reward = reward[:, :1]
                if done.shape[1] > 1:
                    done = done[:, :1]
                    
                target = reward + (1 - done) * self.config.discount * next_v
            
            current_v = self.critic(batch["state"]) # Learner passes 'state' dict
            
            loss_critic = F.mse_loss(current_v, target)
            return {"loss_critic": loss_critic}
            
        elif model == "temperature":
            return {"loss_temperature": torch.tensor(0.0, device=self.config.device, requires_grad=True)}
            
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
        
        # We need to update sample_actions in PI05RLPytorch to accept advantage
        actions = self.model.sample_actions(images, img_masks, tokens, masks, advantage)
        
        return actions

