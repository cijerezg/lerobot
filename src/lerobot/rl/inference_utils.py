"""Observation conversion and action bounds used by the RTC actor runtime."""

import logging

import torch

from lerobot.utils.action_smoothing import bound_action_chunk

logger = logging.getLogger(__name__)


def bound_policy_actions(actions: torch.Tensor, latest_obs: dict, policy) -> torch.Tensor:
    """Apply the policy config's per-joint bounds to a decoded [T, D] chunk, relative
    to the observed state it was inferred from. Policies without the fields pass through.
    """
    limits = {
        name: getattr(policy.config, name, None)
        for name in ("action_lag_limits", "action_delta_limits", "action_clamp_limits", "action_step_limits")
    }
    if all(limit is None for limit in limits.values()):
        return actions
    from lerobot.utils.constants import OBS_STATE

    anchor = latest_obs[OBS_STATE].reshape(-1)
    width = anchor.shape[-1]
    # The chunk is the policy's padded width, the state is the rig's own; the pad
    # columns carry zero limits, so a zero anchor there is exact (zeroing the pad is
    # by design, so the log below skips those columns).
    anchor = torch.nn.functional.pad(anchor, (0, actions.shape[-1] - width))
    # One stage at a time, so the log names the binding stage and the tick it bit at.
    # The absolute box is capped by the driver's joint_limits, which the follower
    # clips at anyway (the model sits a few degrees past them at "closed" on every
    # chunk), so that stage logs at debug like the follower's own clip; the other
    # three only bind when the model leaves the demos' envelope.
    stages = (
        ("lag", {"lag_limits": limits["action_lag_limits"]}, logger.warning),
        ("excursion", {"delta_limits": limits["action_delta_limits"]}, logger.warning),
        ("absolute", {"clamp_limits": limits["action_clamp_limits"]}, logger.debug),
        ("rate", {"step_limits": limits["action_step_limits"]}, logger.warning),
    )
    bounded = actions
    for stage, limit, log in stages:
        before = bounded
        bounded = bound_action_chunk(before, anchor, **limit)
        moved, at = (bounded - before).abs()[:, :width].max(dim=0)
        joints = (moved > 1e-3).nonzero(as_tuple=True)[0].tolist()
        if joints:
            log(
                "[BOUND] %s stage moved joints %s by up to %s deg@tick",
                stage,
                joints,
                [f"{moved[j].item():.1f}@{at[j].item()}" for j in joints],
            )
    return bounded


def convert_env_obs_to_policy_format(env_obs: dict) -> dict:
    """Convert environment observation format to policy-expected format.
    Handles partial conversions to avoid breaking downstream Pi05 preprocessors.
    """
    policy_obs = {}
    
    has_policy_format = (
        'observation.state' in env_obs or
        any(k.startswith('observation.images.') for k in env_obs.keys())
    )

    if has_policy_format:
        for key in env_obs.keys():
            if key == 'observation.state' or key.startswith('observation.images.'):
                policy_obs[key] = env_obs[key]
    
    camera_mapping = {
        'wrist': 'observation.images.wrist',
        'top': 'observation.images.top',
        'side': 'observation.images.side',
    }
    
    pixels_dict = env_obs.get('pixels', env_obs)
    for env_key, policy_key in camera_mapping.items():
        if policy_key not in policy_obs:
            if env_key in pixels_dict:
                policy_obs[policy_key] = pixels_dict[env_key]
    
    if 'observation.state' not in policy_obs:
        joint_order = [
            'shoulder_pan.pos',
            'shoulder_lift.pos',
            'elbow_flex.pos',
            'wrist_flex.pos',
            'wrist_roll.pos',
            'gripper.pos',
        ]
        joint_values = []
        for joint_key in joint_order:
            if joint_key in env_obs:
                val = env_obs[joint_key]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor([val], dtype=torch.float32)
                elif val.dim() == 0:
                    val = val.unsqueeze(0)
                joint_values.append(val)
        
        if joint_values:
            policy_obs['observation.state'] = torch.cat(joint_values, dim=0)
    
    return policy_obs
