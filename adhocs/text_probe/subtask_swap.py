"""Step swap: how far does the chunk follow the step text, for a checkpoint and two crossings of it with base.

    uv run python lerobot/adhocs/text_probe/subtask_swap.py --config_path=config_rl_validate.yaml

``checkpoint`` is loaded whole with its own saved policy config and processors, as in ``backbone_cross.py``.
Each entry of ``MODELS`` then replaces the named parts (vision path, language model, action expert) by those
of another checkpoint; the parts it does not name stay the checkpoint's.

On ``frames_per_episode`` evenly spaced frames of every episode of ``root``, each model predicts ``seeds``
chunks (flow seeds 0.., the same noise for every model and step text) under several step texts, everything
else of the prompt and the observation unchanged:

    own      the frame's recorded step
    home     the checkpoint's ``eval_home_subtask``
    later    another step of the same episode whose next stretch starts after the frame
    earlier  another step of the same episode that was last carried out before the frame
    none     no step clause

Each text comes with the arm pose where that step ends in the recording (last frame of the stretch; ``home``
for the home step), so the viewer can say whether a chunk moves toward it. ``subtask_swap_view.py`` draws the file.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import torch
from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import (
    _dataset_subtask_indices,
    build_episode_index,
    get_subtask_str,
    probe_frame_inputs,
    register_config_choices,
    sample_episodes_evenly,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

from backbone_cross import INTERNAL, PARTS, jpeg, parts
from run_text_probe import label

# name -> {part: checkpoint}; a part not named stays the loaded checkpoint's
MODELS = {
    "internal": {},
    "base vision + base language model, internal action expert": {"vision": "base", "language": "base"},
    "internal vision + language model, base action expert": {"expert": "base"},
}


@dataclass
class SwapConfig(TrainRLServerPipelineConfig):
    checkpoint: str = INTERNAL
    root: str = "outputs/data/rebot_2026-10-14-val-v1"
    frames_per_episode: int = 20
    seeds: int = 4
    depth: bool = True
    # arm pose (deg) the home step ends in: the pose episodes 2, 3, 5, 6, 7 of the default root start from, rounded
    home: str = "0,-5,-10,10,0,0,-5"
    out: str = "outputs/analysis/subtask_swap_2026-11-16/chunks.pt"


def stretches(dataset) -> dict[int, list[list]]:
    """{episode: [[step, first index, last index]]} in recording order."""
    steps = _dataset_subtask_indices(dataset).tolist()
    names = {step: get_subtask_str(dataset, step) for step in set(steps)}
    out = {}
    for episode, indices in build_episode_index(dataset).items():
        runs = out[episode] = []
        for index in indices:
            if runs and runs[-1][0] == names[steps[index]]:
                runs[-1][2] = index
            else:
                runs.append([names[steps[index]], index, index])
    return out


def conditions(index: int, runs: list[list], home: str) -> list[tuple]:
    """[(kind, step text, index of the frame where that step ends or None)] for the frame at ``index``."""
    own = next(run for run in runs if run[1] <= index <= run[2])
    out = [("own", own[0], own[2])]
    if home != own[0]:
        out.append(("home", home, None))
    for step in dict.fromkeys(run[0] for run in runs):
        if step not in (own[0], home):
            same = [run for run in runs if run[0] == step]
            later = [run for run in same if run[1] > index]
            out.append(("later", step, later[0][2]) if later else ("earlier", step, same[-1][2]))
    return out + [("none", "", None)]


def run(cfg: SwapConfig) -> None:
    init_logging()
    assert not os.path.exists(cfg.out), f"{cfg.out} exists; pass another --out"
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    base_path, tokenizer = cfg.policy.base_path, cfg.policy.discrete_action_tokenizer
    cfg.policy = PreTrainedConfig.from_pretrained(cfg.checkpoint)
    # the saved config has the paths of the day it was written
    cfg.policy.base_path, cfg.policy.discrete_action_tokenizer, cfg.policy.pretrained_path = base_path, tokenizer, cfg.checkpoint

    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.root)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)
    policy = adapter.policy

    sources = dict.fromkeys([cfg.checkpoint] + [path for swap in MODELS.values() for path in swap.values()])
    weights = {path: parts(path, base_path) for path in sources}
    own = set(policy.state_dict())
    for path, state in weights.items():
        for part, prefix in PARTS.items():
            assert set(state[part]) == {name for name in own if name.startswith(prefix)}, (path, part)

    samples = sample_episodes_evenly(dataset, cfg.frames_per_episode, None, 0, stride=cfg.policy.image_stride)
    frames = [
        probe_frame_inputs(dataset, cfg, index, cfg.policy.chunk_size, with_depth=cfg.depth, with_gripper_event_targets=False)
        for _, _, index in samples
    ]
    runs = stretches(dataset)
    states = torch.tensor(dataset.hf_dataset.select_columns(["observation.state"]).with_format(None)[:]["observation.state"])
    home = torch.tensor([float(x) for x in cfg.home.split(",")])
    nowhere = torch.full_like(home, float("nan"))
    swaps = [conditions(f["global_idx"], runs[f["episode_idx"]], cfg.policy.eval_home_subtask) for f in frames]
    result = {
        "checkpoint": label(cfg.checkpoint), "models": list(MODELS), "root": cfg.root, "depth": cfg.depth, "fps": cfg.env.fps,
        "joints": dataset.meta.features["action"]["names"], "home": home,
        "episode": [f["episode_idx"] for f in frames], "frame": [f["frame_idx"] for f in frames],
        "index": [f["global_idx"] for f in frames], "task": [f["task"] for f in frames],
        "kinds": [[kind for kind, _, _ in swap] for swap in swaps], "steps": [[step for _, step, _ in swap] for swap in swaps],
        # arm pose where each step ends, [texts, joints] per frame; nan = no such pose (no step clause)
        "ends": [torch.stack([home if kind == "home" else nowhere if end is None else states[end] for kind, _, end in swap]) for swap in swaps],
        "state": torch.stack([f["state"].float().cpu() for f in frames]),
        "recorded": torch.stack([f["gt_actions"].float().cpu() for f in frames]),  # [frames, steps, joints], degrees
        "images": [{key.removeprefix("observation.images."): jpeg(value) for key, value in f["obs"].items()
                    if key.startswith("observation.images.") and value.dim() == 4} for f in frames],
        "chunks": {},  # model -> per frame [texts, seeds, steps, action width], degrees
    }

    for name, swap in MODELS.items():
        start = time.time()
        state = {}
        for part in PARTS:
            state |= weights[swap.get(part, cfg.checkpoint)][part]
        missing, unexpected = policy.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        result["chunks"][name] = [
            torch.stack([
                torch.stack([
                    adapter.predict_action_chunk_in_mode(
                        f["obs"], f["task"], inference_action_mode="continuous", state=f["state"], subtask=step or None,
                        metadata=f["metadata"], generator=torch.Generator(device=device).manual_seed(seed),
                    )[0].float().cpu()
                    for seed in range(cfg.seeds)
                ])
                for step in steps
            ])
            for f, steps in zip(frames, result["steps"])
        ]
        torch.save(result, cfg.out)
        print(f"{name}: {sum(len(steps) for steps in result['steps'])} step texts on {len(frames)} frames x {cfg.seeds} seeds, {time.time() - start:.0f} s", flush=True)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: SwapConfig):
    run(cfg)


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()
