#!/usr/bin/env python

import json
import logging
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.utils import init_logging, get_safe_torch_device
from lerobot.utils.control_utils import predict_action, init_keyboard_listener
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.constants import OBS_STR, ACTION

# Configuration
CONFIG_PATH = "config-hiserl.json"

def generate_heatmap(data, shape=(224, 224)):
    # data: [1, N, D]
    if len(data.shape) == 3:
        feat = data[0] # [N, D]
        if feat.shape[0] == 256:
            feat = feat.reshape(16, 16, -1)
        elif feat.shape[0] == 257: # with CLS
            feat = feat[1:].reshape(16, 16, -1)
        else:
            return None
        
        heatmap = torch.norm(feat, dim=-1).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = cv2.resize(heatmap, shape)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
    return None

def main():
    init_logging()
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    # 1. Initialize Robot
    logging.info("Initializing robot...")
    robot_cfg = RobotConfig(**config["env"]["robot"])
    robot = make_robot_from_config(robot_cfg)
    
    # 2. Initialize Policy
    logging.info("Initializing policy...")
    policy_cfg_dict = config["policy"]
    policy_path = policy_cfg_dict["pi05_checkpoint"]
    
    # Load policy config
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy = make_policy(policy_cfg)
    device = get_safe_torch_device(policy_cfg.device)
    policy.to(device)
    policy.eval()
    
    # 3. Initialize Processors
    stats_path = Path(policy_path) / "dataset_stats.json"
    dataset_stats = None
    if stats_path.exists():
        with open(stats_path, "r") as f:
            dataset_stats = json.load(f)
            from lerobot.datasets.utils import unflatten_dict
            dataset_stats = unflatten_dict({k: torch.tensor(v) for k, v in dataset_stats.items()})

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset_stats,
    )
    
    # 4. Setup Hooks
    attention_data = defaultdict(list)
    hooks = []
    
    try:
        model = policy.model
        if hasattr(model, "paligemma_with_expert"):
            vlm = model.paligemma_with_expert.paligemma
        elif hasattr(model, "actor") and hasattr(model.actor.model, "paligemma_with_expert"):
            vlm = model.actor.model.paligemma_with_expert.paligemma
        else:
            raise AttributeError("Could not find paligemma model in policy structure.")
            
        vision_layers = vlm.vision_tower.vision_model.encoder.layers
        for i, layer in enumerate(vision_layers):
            attn_module = layer.self_attn
            def attn_weights_hook(module, input, output, idx=i):
                attention_data[f"vision_attn_{idx}"].append(output[0].detach().cpu())
            hooks.append(attn_module.register_forward_hook(attn_weights_hook))
        logging.info(f"Successfully setup hooks for {len(vision_layers)} vision layers.")
    except Exception as e:
        logging.error(f"Failed to setup hooks: {e}")

    # 5. Run Loop
    robot.connect()
    listener, events = init_keyboard_listener()
    
    logging.info("Starting visualization loop. Press 'q' or 'Esc' to quit.")
    task = policy_cfg_dict.get("task", "do something")
    fps = config["env"].get("fps", 10)
    
    try:
        while not events["stop_recording"]:
            start_loop_t = time.perf_counter()
            
            # Get observation
            obs = robot.get_observation()
            
            # Manual mapping from robot observation to policy input features
            observation_frame = {}
            
            # Images: Map "wrist" -> "observation.images.wrist", etc.
            for cam_key in ["wrist", "top", "side"]:
                if cam_key in obs:
                    img = obs[cam_key]
                    # Normalize to [0, 1] as expected by policy._preprocess_images
                    img = img.astype(np.float32) / 255.0
                    observation_frame[f"observation.images.{cam_key}"] = img
            
            # State: Map joint positions to "observation.state"
            joint_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            state = []
            for joint in joint_order:
                key = f"{joint}.pos"
                if key in obs:
                    state.append(obs[key])
            if state:
                observation_frame["observation.state"] = np.array(state, dtype=np.float32)
            
            # Predict action (triggers hooks)
            attention_data.clear()
            predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy_cfg.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            
            # Display cameras with heatmaps
            cam_keys = list(config["env"]["robot"]["cameras"].keys())
            for i, cam_key in enumerate(cam_keys):
                if cam_key in obs:
                    cam_img = obs[cam_key]
                    # Convert from RGB to BGR for OpenCV
                    display_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                    
                    num_layers = len(hooks)
                    last_key = f"vision_attn_{num_layers-1}"
                    if last_key in attention_data and i < len(attention_data[last_key]):
                        heatmap = generate_heatmap(attention_data[last_key][i], 
                                                   shape=(display_img.shape[1], display_img.shape[0]))
                        if heatmap is not None:
                            display_img = cv2.addWeighted(display_img, 0.6, heatmap, 0.4, 0)
                    
                    cv2.putText(display_img, f"Task: {task}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow(f"Camera: {cam_key}", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
            
    finally:
        cv2.destroyAllWindows()
        for hook in hooks:
            hook.remove()
        if listener:
            listener.stop()
        robot.disconnect()

if __name__ == "__main__":
    main()
