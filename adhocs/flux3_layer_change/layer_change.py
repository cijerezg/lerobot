"""Base FLUX 3 Action vs a flux3_rl checkpoint: where do the weights and the representations differ?

    uv run python lerobot/adhocs/flux3_layer_change/layer_change.py \
        --checkpoint outputs/runs/<run>/checkpoints/<step>/pretrained_model \
        --data outputs/data/rebot_teleop_2026-11-26-small-objects-cut-v2-annotated-v1 --frames 300 2000 6000 9000

Run from the workspace root. Both models get the same training inputs of each frame (same noise, same flow
time). The base has no heads for the rebot streams, so it runs with the checkpoint's `emb_in` / `final_layer`:
every difference comes from the pretrained weights that moved.

Per layer and stream, at the block's output: |h_ckpt - h_base| / |h_base| over all tokens, and the mean
cosine per token. The difference at a layer includes what the layers before it changed. The weight table is
per block: |w - w0| / |w0|.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F_vision

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flux3 import Flux3Policy, make_flux3_pre_post_processors
from lerobot.policies.flux3.f3.wiring import build_action_dit, fresh_module_names
from lerobot.policies.flux3.modeling_flux3 import TRUNK_WEIGHTS_FILENAME
from lerobot.policies.flux3.utils import resolve_weights
from lerobot.rl.flux3 import windows
from lerobot.rl.flux3.rl_flux3 import SPEED_LEVELS, render_caption
from lerobot.utils.constants import ACTION, OBS_STATE

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True, type=Path, help="<run>/checkpoints/<step>/pretrained_model")
ap.add_argument("--data", required=True, type=Path, help="one of the training roots")
ap.add_argument("--frames", type=int, nargs="+", required=True, help="dataset frame indices")
ap.add_argument("--out", type=Path, default=Path("outputs/analysis/flux3_layer_change_2026-11-26"))
args = ap.parse_args()

policy = Flux3Policy.from_pretrained(args.checkpoint).to("cuda").eval()
config = policy.config
trained = policy.dit
base = build_action_dit(
    policy.dit_params,
    resolve_weights(config.trunk_weights, TRUNK_WEIGHTS_FILENAME),
    modality=policy.modality,
    device="cuda",
    dtype=policy.dtype_,
)
head_prefixes = tuple(name + "." for name in fresh_module_names(policy.modality))
trained_weights = trained.state_dict()
base.load_state_dict({k: v for k, v in trained_weights.items() if k.startswith(head_prefixes)}, strict=False)
streams = list(trained.in_channels) + ["txt"]


# ---------------------------------------------------------------- weights
def block_of(key: str) -> str:
    parts = key.split(".")
    if parts[0] == "content_mode_blocks":
        return ".".join(parts[:3])
    if parts[0] in ("txt_mode_blocks", "single_blocks") or parts[1] in streams:
        return ".".join(parts[:2])
    return parts[0]


moved, size = defaultdict(float), defaultdict(float)
for key, w0 in base.state_dict().items():
    if key.startswith(head_prefixes):
        continue
    w0 = w0.double()
    moved[block_of(key)] += (trained_weights[key].double() - w0).square().sum().item()
    size[block_of(key)] += w0.square().sum().item()
weight_change = {block: (moved[block] / size[block]) ** 0.5 for block in moved}


# ---------------------------------------------------------------- representations
def block_outputs(dit, kwargs) -> dict:
    """The output of every block per stream, keyed (layer, stream); the model's predictions under layer "out"."""
    lengths = {}
    for key, stream in dit.sequence.items():
        if key in kwargs:
            lengths[stream] = lengths.get(stream, 0) + kwargs[key].shape[1]
    out, hooks = {}, []

    def early(layer, stream):
        return lambda module, inputs, output: out.__setitem__((layer, stream), output.float())

    def joint(layer):
        def hook(module, inputs, output):
            txt, content = output
            out[(layer, "txt")] = txt.float()
            for stream, part in zip(lengths, content.split(list(lengths.values()), dim=1), strict=True):
                out[(layer, stream)] = part.float()

        return hook

    for stream, blocks in dit.content_mode_blocks.items():
        hooks += [block.register_forward_hook(early(d, stream)) for d, block in enumerate(blocks)]
    hooks += [block.register_forward_hook(early(d, "txt")) for d, block in enumerate(dit.txt_mode_blocks)]
    hooks += [block.register_forward_hook(joint(dit.depth + i)) for i, block in enumerate(dit.single_blocks)]
    with torch.no_grad():
        for key, value in dit(**kwargs).items():
            out[("out", key)] = value.float()
    for hook in hooks:
        hook.remove()
    return out


# the training inputs of a frame: the observations, then the future frames the video target keeps
future = [1 + k for k in range(0, config.chunk_size, config.future_video_stride)]
times = [offset / 30 for offset in config.obs_offsets + future]
cameras = config.camera_order
dataset = LeRobotDataset(args.data.name, root=args.data, delta_timestamps={key: times for key in cameras})
table = windows.read_frames(args.data)
states = windows.windows(windows.states(table), table, config.obs_offsets)
commands = windows.windows(
    np.stack(table[ACTION]).astype(np.float32), table, range(-config.n_obs_steps, config.chunk_size)
)
speeds = windows.speed_levels(args.data, len(table))
mistakes = windows.mistakes(args.data, len(table))
steps = [
    window
    for episode in json.loads((args.data / "meta/subtask_windows.json").read_text())["episodes"].values()
    for window in episode
]
pre, _ = make_flux3_pre_post_processors(config)

# the inputs the policy hands its DiT
passed = {}
trained.register_forward_pre_hook(lambda module, a, kwargs: passed.update(kwargs), with_kwargs=True)

torch.manual_seed(0)
relative, cosine, norm = defaultdict(list), defaultdict(list), defaultdict(list)
for index in args.frames:
    item = dataset[index]
    subtask = next(w["subtask"] for w in steps if w["from_index"] <= index < w["to_index"])
    batch = {key: F_vision.resize(item[key], list(config.image_storage_size))[None].cuda() for key in cameras}
    batch[OBS_STATE] = torch.as_tensor(states[index])[None].cuda()
    batch[ACTION] = torch.as_tensor(commands[index])[None].cuda()
    batch["task"] = [render_caption(item["task"], subtask, bool(mistakes[index]), SPEED_LEVELS[speeds[index]])]
    with torch.no_grad():
        _, metrics = policy.forward(pre(batch))
    kwargs = dict(passed)
    flow_time = kwargs[f"x_{policy.modality}_timesteps"].flatten()[0].item()
    print(f"frame {index}  flow time {flow_time:.2f}  " + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
    h_base, h_trained = block_outputs(base, kwargs), block_outputs(trained, kwargs)
    for key, h0 in h_base.items():
        h1 = h_trained[key]
        relative[key].append(((h1 - h0).norm() / h0.norm()).item())
        cosine[key].append(torch.cosine_similarity(h1, h0, dim=-1).mean().item())
        norm[key].append(h0.norm(dim=-1).mean().item())

# ---------------------------------------------------------------- report
depth, n_joint = trained.depth, len(trained.single_blocks)
names = [f"own {d}" for d in range(depth)] + [f"joint {i}" for i in range(n_joint)]
print(f"\nrepresentation change |h_ckpt - h_base| / |h_base| at each block's output, mean of {len(args.frames)} frames")
print(f"{'layer':>4} {'block':<9}" + "".join(f"{s:>24}" for s in streams))
for layer, name in enumerate(names):
    print(f"{layer + 1:>4} {name:<9}" + "".join(f"{np.mean(relative[(layer, s)]):>24.4f}" for s in streams))
print("predictions: " + "  ".join(f"{k[1]} {np.mean(v):.4f}" for k, v in relative.items() if k[0] == "out"))
for title, table in (("cosine per token, checkpoint vs base", cosine), ("token norm in the base", norm)):
    print(f"\n{title}")
    print(f"{'layer':>4} {'block':<9}" + "".join(f"{s:>24}" for s in streams))
    for layer, name in enumerate(names):
        print(f"{layer + 1:>4} {name:<9}" + "".join(f"{np.mean(table[(layer, s)]):>24.4f}" for s in streams))

print("\nweight change |w - w0| / |w0| per block")
print(f"{'layer':>4} {'block':<9}" + "".join(f"{s:>24}" for s in streams))
for d in range(depth):
    row = [weight_change[f"content_mode_blocks.{s}.{d}"] for s in streams[:-1]] + [weight_change[f"txt_mode_blocks.{d}"]]
    print(f"{d + 1:>4} {names[d]:<9}" + "".join(f"{value:>24.4f}" for value in row))
for i in range(n_joint):
    print(f"{depth + i + 1:>4} {names[depth + i]:<9}{weight_change[f'single_blocks.{i}']:>24.4f}")
for block, value in weight_change.items():
    if not block.startswith(("content_mode_blocks", "txt_mode_blocks", "single_blocks")):
        print(f"     {block:<44}{value:.4f}")

args.out.mkdir(parents=True, exist_ok=True)
path = args.out / f"{args.checkpoint.parents[2].name}_{args.checkpoint.parent.name}.json"
path.write_text(
    json.dumps(
        {
            "checkpoint": str(args.checkpoint),
            "data": str(args.data),
            "frames": args.frames,
            "streams": streams,
            "relative": {f"{k[0]}/{k[1]}": float(np.mean(v)) for k, v in relative.items()},
            "cosine": {f"{k[0]}/{k[1]}": float(np.mean(v)) for k, v in cosine.items()},
            "norm_base": {f"{k[0]}/{k[1]}": float(np.mean(v)) for k, v in norm.items()},
            "weight_change": weight_change,
        },
        indent=1,
    )
)
print(path)
