"""A flux3_rl checkpoint on frames of its training roots: predicted future video next to the recorded one.

    uv run python lerobot/adhocs/flux3_smoke/video_checkpoint.py \
        --checkpoint outputs/runs/rebot-flux3-tidy-20261126/checkpoints/<step>/pretrained_model \
        --data outputs/data/rebot_teleop_2026-11-26-small-objects-cut-v2-annotated-v1 --frames 300 2000 6000

Run from the workspace root (the checkpoint's config names the video VAE and the text encoder by relative path).
The inputs are the training inputs of the frame (rl/flux3/windows.py): 3 snapshots and 3 states at 0 / -1 / -2 s,
the command before now, and the caption with the frame's annotated step and speed and "No mistakes are made.".
Output per frame: one mp4, recorded canvas on top and predicted canvas below, both starting at t+1.
"""

import argparse
import json
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as F_vision
from einops import rearrange

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flux3 import Flux3Policy, make_flux3_pre_post_processors
from lerobot.policies.flux3.f3 import packing, sampling
from lerobot.rl.flux3 import windows
from lerobot.rl.flux3.inference import COMMAND_HISTORY
from lerobot.rl.flux3.rl_flux3 import SPEED_LEVELS, render_caption
from lerobot.utils.constants import ACTION, OBS_STATE

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True, type=Path, help="<run>/checkpoints/<step>/pretrained_model")
ap.add_argument("--data", required=True, type=Path, help="one of the training roots")
ap.add_argument("--frames", type=int, nargs="+", required=True, help="dataset frame indices")
ap.add_argument("--out", type=Path, default=Path("outputs/analysis/flux3_video_2026-11-26"))
ap.add_argument("--subtask", default=None, help="step in the caption; default = the frame's annotated step")
ap.add_argument("--speed", default=None, choices=SPEED_LEVELS, help="default = the frame's annotated speed")
ap.add_argument("--tag", default="")
args = ap.parse_args()

policy = Flux3Policy.from_pretrained(args.checkpoint).to("cuda").eval()
config = policy.config
pre, _ = make_flux3_pre_post_processors(config)
vae = policy.frozen.video_vae
cameras = config.camera_order

# the observation times, then every frame of the chunk
times = [offset / 30 for offset in config.obs_offsets] + [k / 30 for k in range(1, config.chunk_size + 1)]
dataset = LeRobotDataset(args.data.name, root=args.data, delta_timestamps={key: times for key in cameras})
table = windows.read_frames(args.data)
states = windows.windows(windows.states(table), table, config.obs_offsets)
commands = windows.windows(np.stack(table[ACTION]).astype(np.float32), table, range(-config.n_obs_steps, 0))
speeds = windows.speed_levels(args.data, len(table))
steps = [
    window
    for episode in json.loads((args.data / "meta/subtask_windows.json").read_text())["episodes"].values()
    for window in episode
]

# predict_action_chunk returns the actions only; keep the denoised video tokens of the same call
denoised = {}
solver = sampling.cosmos_unipc_order2


def keep_video(*a, **kw):
    out = solver(*a, **kw)
    denoised["x_video"] = out["x_video"]
    return out


sampling.cosmos_unipc_order2 = keep_video

args.out.mkdir(parents=True, exist_ok=True)
for index in args.frames:
    item = dataset[index]
    subtask = args.subtask or next(w["subtask"] for w in steps if w["from_index"] <= index < w["to_index"])
    caption = render_caption(item["task"], subtask, False, args.speed or SPEED_LEVELS[speeds[index]])
    # each view squashed to the buffer's size, as in training
    views = {key: F_vision.resize(item[key], list(config.image_storage_size)) for key in cameras}
    batch = {key: views[key][None, : config.n_obs_steps] for key in cameras}
    batch[OBS_STATE] = torch.as_tensor(states[index])[None]
    batch[COMMAND_HISTORY] = torch.as_tensor(commands[index])[None]
    batch["task"] = [caption]
    t0 = time.perf_counter()
    with torch.no_grad():
        policy.predict_action_chunk(pre(batch))
        h, w = config.future_latent_hw
        latent = rearrange(denoised["x_video"], "b (t h w) c -> b c t h w", h=h, w=w)
        predicted = vae.decode(latent.to("cuda", torch.bfloat16))[0].float()  # (3, T, H, W) in [-1, 1]
        recorded = packing.materialize_video(
            torch.stack([views[key][config.n_obs_steps :: config.future_video_stride] for key in cameras]),
            None, "cuda", layout=config.camera_layout, canvas_hw=config.future_canvas_hw,
        )
    n = min(predicted.shape[1], recorded.shape[1])
    both = torch.cat([recorded[:, :n], predicted[:, :n]], dim=2)  # recorded on top
    frames = ((both.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()
    path = args.out / f"{args.data.name}_frame{index}{args.tag}.mp4"
    imageio.mimwrite(path, frames, fps=30 / config.future_video_stride, macro_block_size=None)
    print(f"{path}  '{caption}'  {n} frames  {time.perf_counter() - t0:.1f} s")
