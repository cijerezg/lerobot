"""FLUX 3 Action first run: build a rebot-shaped policy on the base trunk and time predict_action_chunk.

    uv run python lerobot/adhocs/flux3_smoke/smoke.py --base outputs/models/flux-3-action-base

The trunk is the released base; the action / state heads are fresh (7 wide, no released checkpoint has
them), the quantiles are placeholders and the observations are noise images + a zero state. The numbers are
load / latency / memory only; the actions mean nothing.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors
from lerobot.policies.flux3.f3 import packing
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, type=Path, help="flux-3-action-base folder")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--chunk", type=int, default=42)
ap.add_argument("--canvas", type=int, nargs=2, default=[448, 896], help="height width of the two-camera canvas")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--future-stride", type=int, default=5, help="keep every k-th future video frame (1 = all)")
ap.add_argument("--future-canvas", type=int, nargs=2, default=[256, 512], help="canvas of the future video")
ap.add_argument("--guidance", type=float, default=1.0, help="video guidance (1.0 = off, theirs 4.0)")
ap.add_argument("--snapshots", type=int, default=3, help="observations: one image snapshot + state token each")
ap.add_argument("--obs-stride", type=int, default=30, help="frames between the observations")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--video-frames", type=int, default=None, help="predicted latent video frames (timing only)")
ap.add_argument("--task", default="insert the tube into the peg")
args = ap.parse_args()

dim = 7  # rebot: 6 joints + gripper
state_dim = 3 * dim  # position, velocity, torque of each motor
cameras = ["observation.images.top", "observation.images.wrist"]
device = torch.device("cuda")
config = Flux3Config(
    input_features={
        **{key: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)) for key in cameras},
        OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
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
    n_obs_steps=args.snapshots,
    history_snapshots=args.snapshots,
    obs_stride=args.obs_stride,
    normalization_stats={
        s: {"q01": [-1.0] * n, "q99": [1.0] * n} for s, n in (("action", dim), ("state", state_dim))
    },
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
    compile_model=args.compile,
    device="cuda",
)

t0 = time.perf_counter()
policy = Flux3Policy(config).to(device).eval()
pre, post = make_flux3_pre_post_processors(config)
if args.video_frames is not None:
    # what the trunk costs with fewer future video tokens; the model was not trained this way
    policy.packer = policy.packer._replace(
        predicted_latent_frames=lambda cfg: args.video_frames,
        predicted_video_times=lambda cfg, b: packing.video_time_ids(
            args.video_frames, packing.latent_frames(cfg.n_obs_steps), b, fps=cfg.video_position_fps
        ),
    )
print(f"load {time.perf_counter() - t0:.1f} s")
print(
    f"chunk {config.chunk_size} x {config.action_dim} at {config.fps} Hz, n_obs_steps {config.n_obs_steps}, "
    f"snapshots {config.history_snapshots} at frames {config.obs_offsets}, state {config.state_dim} wide, canvas {config.canvas_hw}, future video "
    f"{policy.packer.predicted_latent_frames(config)} latent frames x {config.future_latent_hw}, "
    f"{config.sampler} {config.num_inference_steps} steps, guidance {config.guidance_scale} / "
    f"{config.guidance_scale_action}, dtype {config.dtype}"
)
print(f"memory after load {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

rng = np.random.default_rng(0)
times = []
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    for i in range(args.n):
        obs = {key: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for key in cameras}
        obs[OBS_STATE] = np.zeros(state_dim, dtype=np.float32)
        batch = prepare_observation_for_inference(obs, device, task=args.task)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk = post(policy.predict_action_chunk(pre(batch)))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        print(f"call {i}: {times[-1] * 1000:.0f} ms")

print(f"chunk {tuple(chunk.shape)}, finite {bool(torch.isfinite(chunk).all())}")
print(f"latency median of the last {len(times) - 2}: {np.median(times[2:]) * 1000:.0f} ms")
print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
