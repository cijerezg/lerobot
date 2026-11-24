"""Does swapping the vision path change the discrete action output more than the continuous one?

    uv run python lerobot/adhocs/vision_swap/discrete_swap.py --config_path=config_rl_validate.yaml

The checkpoint ``model`` is loaded whole with its own saved policy config. On every frame it runs as it is (the
reference) and once per condition; a condition changes only what the language model receives at the image
positions (the output of the vision backbone, one vector per image patch position):

    base      the tokens of the vision path of ``vision`` (computed with its weights, then put in place)
    random    the reference tokens + per token a random direction of the length of (base - reference)
    next      the camera images of the frame ``next_seconds`` later in the same episode (earlier at its end)
    other     the camera images of the same slot of the next episode

State, depth, task, step and every other clause of the prompt are the frame's own in all of them. Per frame and
condition the file keeps:

    continuous  chunks of ``seeds`` flow seeds (same noise in every condition), degrees
    discrete    the chunk decoded from the generated action tokens, degrees (NaN when they do not decode)
    losses      the training forward on the recorded chunk (dropout off, flow timesteps and noise fixed per
                frame): cross-entropy and top-1 per recorded action token, flow loss

``discrete_swap_report.py`` reads the file.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "text_probe"))

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.objective import flow_timestep_grid
from lerobot.probes.utils import probe_frame_inputs, register_config_choices, sample_episodes_evenly
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from backbone_cross import INTERNAL, parts
from run_text_probe import label

CONDITIONS = ["base", "random", "next", "other"]


@dataclass
class SwapConfig(TrainRLServerPipelineConfig):
    model: str = INTERNAL  # loaded whole: its language model and action expert are the ones in every condition
    vision: str = "base"  # the other vision path; "base" = policy.base_path
    root: str = "outputs/data/rebot_2026-10-14-val-v1"
    frames_per_episode: int = 10
    seeds: int = 4
    next_seconds: float = 1.0
    out: str = "outputs/analysis/vision_swap_2026-11-19/discrete_swap.pt"


class image_tokens:
    """``fn(tokens) -> new tokens or None`` on the output of the vision backbone. ``seen`` keeps the output of
    every call as the model computed it."""

    def __init__(self, policy, fn=None):
        self.vision, self.fn, self.seen = policy._backbone().vision_backbone, fn, []

    def __enter__(self):
        def hook(_module, _inputs, output):
            self.seen.append(output.detach())
            if self.fn is None:
                return None
            new = self.fn(output)
            assert new.shape == output.shape, (new.shape, output.shape)
            return new.to(output.dtype)

        self.handle = self.vision.register_forward_hook(hook)
        return self

    def __exit__(self, *_):
        self.handle.remove()


def with_images(own: dict, other: dict) -> dict:
    """``own``'s observation with the camera images of ``other`` (depth stays ``own``'s)."""
    assert own.keys() == other.keys(), (sorted(own.keys() ^ other.keys()))
    return {key: other[key] if key.startswith("observation.images.") else value for key, value in own.items()}


def run(cfg: SwapConfig) -> None:
    init_logging()
    assert not os.path.exists(cfg.out), f"{cfg.out} exists; pass another --out"
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    base_path, tokenizer = cfg.policy.base_path, cfg.policy.discrete_action_tokenizer
    cfg.policy = PreTrainedConfig.from_pretrained(cfg.model)
    # the saved config has the paths of the day it was written
    cfg.policy.base_path, cfg.policy.discrete_action_tokenizer, cfg.policy.pretrained_path = base_path, tokenizer, cfg.model

    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.root)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)
    policy = adapter.policy
    own_vision, other_vision = parts(cfg.model, base_path)["vision"], parts(cfg.vision, base_path)["vision"]
    assert set(own_vision) == set(other_vision)
    grid = torch.from_numpy(flow_timestep_grid(cfg.policy, int(cfg.policy.num_flow_timesteps))).float()

    def frame(index):
        return probe_frame_inputs(dataset, cfg, index, cfg.policy.chunk_size)

    def chunk(f, obs, mode, seed, fn):
        with image_tokens(policy, fn) as vision:
            out, _, _ = adapter.predict_action_chunk_in_mode(
                obs, f["task"], inference_action_mode=mode, state=f["state"], subtask=f["subtask"] or None,
                metadata=f["metadata"], generator=torch.Generator(device=device).manual_seed(seed),
            )
        assert len(vision.seen) == 1, f"vision backbone ran {len(vision.seen)} times in one forward"
        return out.float().cpu(), vision.seen[0]

    def losses(f, obs, fn):
        with image_tokens(policy, fn) as vision:
            out = adapter.training_losses({**f, "obs": obs}, flow_timesteps=grid, flow_noise_seed=int(f["global_idx"]))
        assert len(vision.seen) == 1, f"vision backbone ran {len(vision.seen)} times in the training forward"
        flow = out["loss_flow_joint"] if out["loss_flow_joint"] is not None else out["loss_flow"]
        return {"ce": torch.from_numpy(out["discrete_token_ce"]).flatten(), "top1": torch.from_numpy(out["discrete_token_top1"]).flatten(),
                "ce_loss": out["loss_discrete_ce"], "flow": flow, "flow_by_timestep": torch.from_numpy(out["loss_flow_by_timestep"])}

    samples = sample_episodes_evenly(dataset, cfg.frames_per_episode, None, 0, stride=cfg.policy.image_stride)
    frames = [frame(index) for _, _, index in samples]
    shift = round(cfg.next_seconds * cfg.env.fps / cfg.policy.image_stride) * cfg.policy.image_stride
    later = []
    for f in frames:
        candidate = frame(f["global_idx"] + shift) if f["global_idx"] + shift < dataset.num_frames else None
        if candidate is None or candidate["episode_idx"] != f["episode_idx"]:
            candidate = frame(f["global_idx"] - shift)
        assert candidate["episode_idx"] == f["episode_idx"]
        later.append(candidate)

    # the other vision path's tokens of every frame, with its weights in place
    policy.load_state_dict(other_vision, strict=False)
    swapped = [chunk(f, f["obs"], "continuous", 0, None) for f in frames]
    policy.load_state_dict(own_vision, strict=False)

    result = {
        "model": label(cfg.model), "vision": label(cfg.vision), "root": cfg.root, "conditions": CONDITIONS, "seeds": cfg.seeds,
        "next_seconds": cfg.next_seconds, "fps": cfg.env.fps, "joints": dataset.meta.features["action"]["names"], "flow_timesteps": grid,
        "episode": [f["episode_idx"] for f in frames], "frame": [f["frame_idx"] for f in frames], "index": [f["global_idx"] for f in frames],
        "state": torch.stack([f["state"].float().cpu() for f in frames]),
        "recorded": torch.stack([f["gt_actions"].float().cpu() for f in frames]), "frames": [],
    }
    start = time.time()
    for at, f in enumerate(frames):
        _, a = chunk(f, f["obs"], "continuous", 0, None)
        b = swapped[at][1].to(a.dtype)
        delta = b - a
        generator = torch.Generator(device=device).manual_seed(at)
        direction = torch.nn.functional.normalize(torch.randn(a.shape, generator=generator, device=device), dim=-1)
        noised = a + (direction * delta.float().norm(dim=-1, keepdim=True)).to(a.dtype)
        other = frames[(at + cfg.frames_per_episode) % len(frames)]
        conditions = {
            "reference": (f["obs"], None),
            "base": (f["obs"], lambda _tokens: b),
            "random": (f["obs"], lambda _tokens: noised),
            "next": (with_images(f["obs"], later[at]["obs"]), None),
            "other": (with_images(f["obs"], other["obs"]), None),
        }
        entry = {"other_episode": other["episode_idx"], "weights_chunk": swapped[at][0], "continuous": {}, "discrete": {}, "losses": {},
                 "token_cosine": float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).median())}
        for name, (obs, fn) in conditions.items():
            entry["continuous"][name] = torch.stack([chunk(f, obs, "continuous", seed, fn)[0] for seed in range(cfg.seeds)])
            try:
                entry["discrete"][name] = chunk(f, obs, "discrete", 0, fn)[0]
            except RuntimeError as error:  # the generated tokens do not decode to a full chunk: kept as NaN
                print(f"frame {at + 1}, {name}: {error}", flush=True)
                entry["discrete"][name] = torch.full_like(entry["continuous"][name][0], float("nan"))
            entry["losses"][name] = losses(f, obs, fn)
        result["frames"].append(entry)
        if at == 0:
            print("image tokens", tuple(a.shape), "| recorded action tokens", len(entry["losses"]["reference"]["ce"]), "| discrete chunk", tuple(entry["discrete"]["reference"].shape), flush=True)
            print("base tokens put in place against base weights, largest chunk difference (deg)", float((entry["continuous"]["base"][0] - entry["weights_chunk"]).abs().max()), flush=True)
        print(f"frame {at + 1} of {len(frames)}, {time.time() - start:.0f} s", flush=True)
        if at % 10 == 9 or at == len(frames) - 1:
            torch.save(result, cfg.out)
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
