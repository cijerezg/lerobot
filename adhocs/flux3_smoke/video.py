"""FLUX 3 Action base on rebot recordings: predicted future video next to the recorded one.

    uv run python lerobot/adhocs/flux3_smoke/video.py --base outputs/models/flux-3-action-base \
        --data outputs/data/rebot_teleop_2026-11-15-tube3-cut-v4 --frames 100 600 1200

The policy is the base trunk with fresh action / state heads (no rebot training), so only the video is worth
looking at. Input: the top + wrist frame at t, the state at t and the task string. Output per frame: one mp4,
recorded canvas on top and predicted canvas below, both starting at t+1.
"""

import argparse
import json
import time
from pathlib import Path

import imageio
import torch
from einops import rearrange

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors
from lerobot.policies.flux3.f3 import packing, sampling
from lerobot.utils.constants import ACTION, OBS_STATE

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, type=Path, help="flux-3-action-base folder")
ap.add_argument("--data", required=True, type=Path, help="dataset root")
ap.add_argument("--frames", type=int, nargs="+", required=True, help="dataset frame indices")
ap.add_argument("--out", type=Path, default=Path("outputs/analysis/flux3_video_2026-11-26"))
ap.add_argument("--chunk", type=int, default=42)
ap.add_argument("--canvas", type=int, nargs=2, default=[448, 896])
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--future-stride", type=int, default=5, help="keep every k-th future video frame (1 = all)")
ap.add_argument("--future-canvas", type=int, nargs=2, default=[256, 512], help="canvas of the future video")
ap.add_argument("--guidance", type=float, default=1.0)
ap.add_argument("--task", default=None, help="caption; default = the recording's task")
ap.add_argument("--tag", default="")
args = ap.parse_args()

cameras = ["observation.images.top", "observation.images.wrist"]
future = [k / 30 for k in range(args.chunk + 1)]
dataset = LeRobotDataset(args.data.name, root=args.data, delta_timestamps={key: future for key in cameras})
state_stats = json.loads((args.data / "meta/stats.json").read_text())[OBS_STATE]
dim = len(state_stats["q01"])

config = Flux3Config(
    input_features={
        **{key: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)) for key in cameras},
        OBS_STATE: PolicyFeature(FeatureType.STATE, (dim,)),
    },
    output_features={ACTION: PolicyFeature(FeatureType.ACTION, (dim,))},
    camera_keys=cameras,
    delta_absolute_dims=[-1],
    chunk_size=args.chunk,
    n_action_steps=args.chunk,
    canvas_hw=tuple(args.canvas),
    future_video_stride=args.future_stride,
    future_canvas_hw=tuple(args.future_canvas),
    trunk_weights=str(args.base / "flux-3-action-base.safetensors"),
    video_vae_id=str(args.base / "video_vae.safetensors"),
    text_encoder_id=str(args.base / "text_encoder"),
    # action quantiles are placeholders: the action head is untrained
    normalization_stats={
        "state": {q: state_stats[q] for q in ("q01", "q99")},
        "action": {"q01": [-1.0] * dim, "q99": [1.0] * dim},
    },
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
    device="cuda",
)
policy = Flux3Policy(config).to("cuda").eval()
pre, _ = make_flux3_pre_post_processors(config)
vae = policy.frozen.video_vae

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
    task = args.task or item["task"]
    batch = {key: item[key][0] for key in cameras}  # frame t
    batch[OBS_STATE] = item[OBS_STATE]
    batch["task"] = task
    t0 = time.perf_counter()
    with torch.no_grad():
        policy.predict_action_chunk(pre(batch))
        h, w = config.future_latent_hw
        latent = rearrange(denoised["x_video"], "b (t h w) c -> b c t h w", h=h, w=w)
        predicted = vae.decode(latent.to("cuda", torch.bfloat16))[0].float()  # (3, T, H, W) in [-1, 1]
        recorded = packing.materialize_video(
            torch.stack([item[key][1 :: args.future_stride] for key in cameras]), None, "cuda",
            layout=config.camera_layout, canvas_hw=config.future_canvas_hw,
        )
    n = min(predicted.shape[1], recorded.shape[1])
    both = torch.cat([recorded[:, :n], predicted[:, :n]], dim=2)  # recorded on top
    frames = ((both.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()
    path = args.out / f"{args.data.name}_frame{index}{args.tag}.mp4"
    imageio.mimwrite(path, frames, fps=30 / args.future_stride, macro_block_size=None)
    print(f"{path}  task '{task}'  {n} frames  {time.perf_counter() - t0:.1f} s")
