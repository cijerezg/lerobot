import glob
import logging
import os
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    ACTION_TOKENS,
    ACTION_TOKEN_MASK,
    OBS_STATE,
)
from lerobot.processor.core import TransitionKey

def preprocess_batch_for_pi05(
    policy: nn.Module,
    observations: dict,
    next_observations: dict,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    task: str,
) -> dict:
    """
    Preprocess batch for Pi05 policy (tokenization and normalization).
    
    Args:
        policy: Policy with preprocessor attached
        observations: Current observations
        next_observations: Next observations
        actions: Actions
        rewards: Rewards
        done: Done flags
        task: Task description
    
    Returns:
        Forward batch ready for policy.forward()
    """
    current_batch_size = actions.shape[0]
    
    # Preprocess current observations
    subtasks = observations.get("subtask", [""] * current_batch_size)

    # Use inference advantage for offline training
    # Handle DDP wrapping for config access
    policy_config = getattr(policy, "module", policy).config
    inference_advantage = getattr(policy_config, "inference_advantage", 0.0)

    # Construct Complementary Data
    complementary_data = {
        "task": [task] * current_batch_size,
        "subtask": subtasks,
        "advantage": torch.full((current_batch_size, 1), inference_advantage, device=actions.device)
    }

    action_encoding = getattr(policy_config, "action_encoding", "absolute")
    if action_encoding in ["anchor", "delta"]:
        if OBS_STATE in observations:
            anchor_state = observations[OBS_STATE]
            if action_encoding == "anchor":
                # anchor: d_t = a_t - s_0
                actions = actions - anchor_state[:, None, :]
            else:
                # delta: d_0 = a_0 - s_0, d_t = a_t - a_{t-1}
                d_0 = actions[:, 0, :] - anchor_state
                if actions.shape[1] > 1:
                    d_rest = torch.diff(actions, dim=1)
                    actions = torch.cat([d_0.unsqueeze(1), d_rest], dim=1)
                else:
                    actions = d_0.unsqueeze(1)
        else:
            logging.warning(f"[UTILS] action_encoding={action_encoding} but {OBS_STATE} not found!")

    # Construct EnvTransition for current step
    batch_for_proc = {
        TransitionKey.ACTION: actions,
        **observations,
        TransitionKey.COMPLEMENTARY_DATA: complementary_data
    }
    
    
    with torch.no_grad():
        # Access preprocessor - handle potential accelerate wrapping
        preprocessor = getattr(policy, 'preprocessor', None) or getattr(policy.module, 'preprocessor', None)
        processed_batch = preprocessor(batch_for_proc)
    
    # Preprocess next observations
    next_subtasks = next_observations.get("subtask", [""] * current_batch_size)

    next_complementary_data = {
        "task": [task] * current_batch_size,
        "subtask": next_subtasks,
        "advantage": complementary_data["advantage"]
    }

    next_batch_for_proc = {
        TransitionKey.ACTION: actions, # Required by preprocessor
        **next_observations,
        TransitionKey.COMPLEMENTARY_DATA: next_complementary_data
    }
    
    with torch.no_grad():
        processed_next_batch = preprocessor(next_batch_for_proc)
    
    # Build forward batch
    forward_batch = {
        ACTION: processed_batch[ACTION],
        "reward": rewards,
        "state": {},
        "next_state": {},
        "done": done,
        "observation_feature": None,
        "next_observation_feature": None,
        "task": complementary_data["task"],
        "subtask": complementary_data["subtask"],
        "advantage": complementary_data["advantage"],
        "next.done": done,
    }
    
    # Copy raw observations (policy.forward will re-preprocess if needed)
    for key in observations.keys():
        if key != "subtask":
            forward_batch["state"][key] = observations[key]
    
    for key in next_observations.keys():
        if key != "subtask":
            forward_batch["next_state"][key] = next_observations[key]
    
    # Add tokens from processor output
    if OBS_LANGUAGE_TOKENS in processed_batch:
        forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
        forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Add subtask tokens if present
    if OBS_LANGUAGE_SUBTASK_TOKENS in processed_batch:
        forward_batch["state"][OBS_LANGUAGE_SUBTASK_TOKENS] = processed_batch[OBS_LANGUAGE_SUBTASK_TOKENS]
        forward_batch["state"][OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]

    # Add action tokens if present in complementary data (as placed by ActionTokenizerProcessorStep)
    comp_data_out = processed_batch.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if ACTION_TOKENS in comp_data_out:
        forward_batch[ACTION_TOKENS] = comp_data_out[ACTION_TOKENS]
        forward_batch[ACTION_TOKEN_MASK] = comp_data_out[ACTION_TOKEN_MASK]
    elif ACTION_TOKENS in processed_batch: # Fallback if Flattened
        forward_batch[ACTION_TOKENS] = processed_batch[ACTION_TOKENS]
        forward_batch[ACTION_TOKEN_MASK] = processed_batch[ACTION_TOKEN_MASK]
    
    # Add critic tokens if present (from CriticTokenizerProcessorStep)
    if "critic_tokens" in processed_batch:
        forward_batch["critic_tokens"] = processed_batch["critic_tokens"]
        forward_batch["critic_pad_mask"] = processed_batch["critic_pad_mask"]
    
    # Add tokens for next state (CRITICAL for critic)
    if OBS_LANGUAGE_TOKENS in processed_next_batch:
        forward_batch["next_state"][OBS_LANGUAGE_TOKENS] = processed_next_batch[OBS_LANGUAGE_TOKENS]
        forward_batch["next_state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_next_batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Add subtask tokens for next state if present
    if OBS_LANGUAGE_SUBTASK_TOKENS in processed_next_batch:
        forward_batch["next_state"][OBS_LANGUAGE_SUBTASK_TOKENS] = processed_next_batch[OBS_LANGUAGE_SUBTASK_TOKENS]
        forward_batch["next_state"][OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = processed_next_batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]
    
    if "critic_tokens" in processed_next_batch:
        forward_batch["next_state"]["critic_tokens"] = processed_next_batch["critic_tokens"]
        forward_batch["next_state"]["critic_pad_mask"] = processed_next_batch["critic_pad_mask"]
    
    forward_batch[OBS_LANGUAGE_TOKENS] = forward_batch["state"].get(OBS_LANGUAGE_TOKENS)
    forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = forward_batch["state"].get(OBS_LANGUAGE_ATTENTION_MASK)
    
    return forward_batch

def save_video_with_critic_overlay(log_dir, critic_values, camera_names=None, fps=30, subtask_texts=None):
    """
    Generate a side-by-side video of configured camera views with a critic curve overlay.

    Args:
        subtask_texts: Optional list of subtask strings (one per frame).
                       Rendered in the top-left corner when provided.
    """
    if camera_names is None:
        camera_names = ["top", "side"]

    # Find all images for each camera
    camera_images = []
    for name in camera_names:
        imgs = sorted(glob.glob(os.path.join(log_dir, f"*_{name}.png")))
        if not imgs:
            logging.warning(f"No images found for camera: {name}")
            continue
        camera_images.append(imgs)

    if not camera_images:
        raise ValueError("No images found for video generation")

    # Ensure we have the same number of images and critic values across all cameras
    num_frames = len(critic_values)
    for imgs in camera_images:
        num_frames = min(num_frames, len(imgs))

    # Video settings
    # Each view is 224x224, resized to 448x448. Side-by-side is N * 448 x 448.
    num_cameras = len(camera_images)
    frame_width = num_cameras * 448
    frame_height = 448
    video_path = os.path.join(log_dir, "episode_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    # Prepare critic curve data for plotting
    # Normalize critic values for plotting (0 to frame_height)
    critic_np = np.array(critic_values[:num_frames])
    c_min, c_max = -1.1, 0.1
    critic_norm = (critic_np - c_min) / (c_max - c_min)
    critic_norm = np.clip(critic_norm, 0, 1)

    # Map to pixel coordinates (inverted Y for image space)
    # Restrict to lower half (frame_height // 2 to frame_height)
    lower_half_height = frame_height // 2
    margin = 10
    plot_y = (lower_half_height - 2 * margin) * (1 - critic_norm) + (frame_height // 2 + margin)
    plot_x = np.linspace(0, frame_width, num_frames)

    def get_y(val):
        norm = (val - c_min) / (c_max - c_min)
        return int((lower_half_height - 2 * margin) * (1 - norm) + (frame_height // 2 + margin))

    for i in range(num_frames):
        # Load and resize images
        imgs_to_stack = []
        for cam_imgs in camera_images:
            img = cv2.imread(cam_imgs[i])
            img_resized = cv2.resize(img, (448, 448))
            imgs_to_stack.append(img_resized)

        # Concatenate side-by-side
        if len(imgs_to_stack) > 1:
            frame = np.hstack(imgs_to_stack)
        else:
            frame = imgs_to_stack[0]

        # Create an overlay for the curve
        overlay = frame.copy()

        # --- Subtask text overlay (top-left) ---
        if subtask_texts is not None and i < len(subtask_texts) and subtask_texts[i]:
            text = f"{subtask_texts[i]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            pad = 6
            cv2.rectangle(
                overlay,
                (10 - pad, 10 - pad),
                (10 + text_size[0] + pad, 10 + text_size[1] + pad),
                (0, 0, 0), cv2.FILLED
            )
            cv2.putText(overlay, text, (10, 10 + text_size[1]),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Draw vertical axis and ticks
        axis_x = 50
        tick_length = 10
        ticks = [-1.0, -0.75, -0.5, -0.25, 0.0]

        # Draw axis line
        y_bottom = get_y(-1.0)
        y_top = get_y(0.0)
        cv2.line(overlay, (axis_x, y_top), (axis_x, y_bottom), (105, 0, 0), 1)

        # Draw ticks and labels
        for tick_val in ticks:
            y = get_y(tick_val)
            # Tick line
            cv2.line(overlay, (axis_x, y), (axis_x - tick_length, y), (105, 0, 0), 1)
            # Label
            label = f"{tick_val}"
            # Adjust text position
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = axis_x - tick_length - text_size[0] - 5
            text_y = y + text_size[1] // 2
            cv2.putText(overlay, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (105, 0, 0), 1, cv2.LINE_AA)

        # 1. Draw full curve with low alpha (faint dark blue)
        points = np.vstack((plot_x, plot_y)).T.astype(np.int32)
        cv2.polylines(overlay, [points], isClosed=False, color=(200, 100, 100), thickness=1)

        # 2. Draw progressing curve with high alpha (dark blue)
        prog_points = points[:i+1]
        if len(prog_points) > 1:
            cv2.polylines(overlay, [prog_points], isClosed=False, color=(105, 0, 0), thickness=3)

            # Draw a vertical dashed line at current position
            curr_x, curr_y = prog_points[-1]
            cv2.line(overlay, (curr_x, frame_height), (curr_x, curr_y), (105, 0, 0), 1, lineType=cv2.LINE_AA)

        # Blend overlay with original frame
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        out.write(frame)

    out.release()

    # Cleanup: Remove individual images after video generation
    all_temp_images = [img for imgs in camera_images for img in imgs]
    for img_path in all_temp_images:
        try:
            os.remove(img_path)
        except Exception:
            pass


def cast_to_bf16(item):
    """
    Helper function to cast tensors in a structure to bfloat16.
    """
    if isinstance(item, torch.Tensor):
        if item.dtype == torch.float32:
            return item.to(dtype=torch.bfloat16)
        return item
    elif isinstance(item, dict):
        return {k: cast_to_bf16(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [cast_to_bf16(v) for v in item]
    return item
