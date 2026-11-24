"""Backbone crossing: one action expert on the vision path and the language model of other checkpoints.

    uv run python lerobot/adhocs/text_probe/backbone_cross.py --config_path=config_rl_validate.yaml

The checkpoint ``expert`` is loaded whole, with its own saved policy config and processors (its prompt format,
its normalization stats, its depth tokens); ``config_rl_validate.yaml`` only supplies the rest of the pipeline
config. Then, for every pair (vision path, language model) of ``visions`` x ``languages``, the weights under
``model.model.vision_backbone.`` (ViT, pooling, projector) and under ``model.model.transformer.`` (token
embedding, the 36 blocks, final norm) are replaced by those of the named checkpoints, read from their files.
Everything else stays the expert checkpoint's: the action expert, the depth modules, the state projector.

Each cell predicts ``seeds`` chunks (flow seeds 0.., the same noise in every cell) on ``frames_per_episode``
evenly spaced frames of every episode of ``root``, through the deployed path (continuous mode, the frame's own
task and step). The file ``out`` holds the chunks in degrees and in the model's normalized space, the recorded
chunk, the arm state and the camera images; ``backbone_cross_view.py`` draws it.
"""
from __future__ import annotations

import glob
import io
import os
import sys
import time
from dataclasses import dataclass

import torch
from PIL import Image
from safetensors.torch import load_file

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import probe_frame_inputs, register_config_choices, sample_episodes_evenly
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from run_text_probe import label

TUBE = "outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/001200/pretrained_model"
INTERNAL = "outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model"
PARTS = {"vision": "model.model.vision_backbone.", "language": "model.model.transformer.", "expert": "model.model.action_expert."}


@dataclass
class CrossConfig(TrainRLServerPipelineConfig):
    expert: str = INTERNAL  # loaded whole; its action expert is the one in every cell
    visions: str = f"base,{INTERNAL}"  # comma list; "base" = policy.base_path
    languages: str = f"base,{TUBE},{INTERNAL}"
    root: str = "outputs/data/rebot_2026-10-14-val-v1"
    frames_per_episode: int = 20
    seeds: int = 4
    depth: bool = True  # false: the wrist depth is marked absent, so the expert checkpoint's depth tokens leave the prompt
    out: str = "outputs/analysis/backbone_cross_2026-11-16/chunks.pt"


def parts(path: str, base_path: str) -> dict[str, dict[str, torch.Tensor]]:
    """{part: weights} of one checkpoint, under the names the loaded policy uses."""
    if path == "base":
        # the released files: several shards, names without the policy wrapper's leading "model."
        files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
        state = {"model." + name: value for file in files for name, value in load_file(file).items()}
    else:
        state = load_file(os.path.join(path, "model.safetensors"))
    return {part: {name: value for name, value in state.items() if name.startswith(prefix)} for part, prefix in PARTS.items()}


def jpeg(image: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((image[0].permute(1, 2, 0) * 255).round().byte().numpy()).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def run(cfg: CrossConfig) -> None:
    init_logging()
    assert not os.path.exists(cfg.out), f"{cfg.out} exists; pass another --out"
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    base_path, tokenizer = cfg.policy.base_path, cfg.policy.discrete_action_tokenizer
    cfg.policy = PreTrainedConfig.from_pretrained(cfg.expert)
    # the saved config has the paths of the day it was written
    cfg.policy.base_path, cfg.policy.discrete_action_tokenizer, cfg.policy.pretrained_path = base_path, tokenizer, cfg.expert

    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.root)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)
    policy = adapter.policy

    visions, languages = cfg.visions.split(","), cfg.languages.split(",")
    weights = {path: parts(path, base_path) for path in dict.fromkeys(visions + languages)}
    own = set(policy.state_dict())
    for path, state in weights.items():
        for part, prefix in PARTS.items():
            assert set(state[part]) == {name for name in own if name.startswith(prefix)}, (path, part)

    samples = sample_episodes_evenly(dataset, cfg.frames_per_episode, None, 0, stride=cfg.policy.image_stride)
    frames = [
        probe_frame_inputs(dataset, cfg, index, cfg.policy.chunk_size, with_depth=cfg.depth, with_gripper_event_targets=False)
        for _, _, index in samples
    ]
    result = {
        "expert": label(cfg.expert), "visions": [label(p) for p in visions], "languages": [label(p) for p in languages],
        "root": cfg.root, "depth": cfg.depth, "fps": cfg.env.fps, "joints": dataset.meta.features["action"]["names"],
        "episode": [f["episode_idx"] for f in frames], "frame": [f["frame_idx"] for f in frames],
        "index": [f["global_idx"] for f in frames], "task": [f["task"] for f in frames], "subtask": [f["subtask"] or "" for f in frames],
        "state": torch.stack([f["state"].float().cpu() for f in frames]),
        "recorded": torch.stack([f["gt_actions"].float().cpu() for f in frames]),  # [frames, steps, joints], degrees
        "recorded_norm": torch.stack([adapter.normalize_gt_actions(f["gt_actions"], f["state"]).float().cpu() for f in frames]),
        "images": [{key.removeprefix("observation.images."): jpeg(value) for key, value in f["obs"].items()
                    if key.startswith("observation.images.") and value.dim() == 4} for f in frames],
        "chunks": {}, "chunks_norm": {},  # (vision, language) -> [frames, seeds, steps, action width]
    }

    for language in languages:
        for vision in visions:
            start = time.time()
            missing, unexpected = policy.load_state_dict(weights[vision]["vision"] | weights[language]["language"], strict=False)
            assert not unexpected, unexpected
            chunks, norms = [], []
            for f in frames:
                both = [
                    adapter.predict_action_chunk_in_mode(
                        f["obs"], f["task"], inference_action_mode="continuous", state=f["state"], subtask=f["subtask"] or None,
                        metadata=f["metadata"], generator=torch.Generator(device=device).manual_seed(seed),
                    )[:2]
                    for seed in range(cfg.seeds)
                ]
                chunks.append(torch.stack([chunk.float().cpu() for chunk, _ in both]))
                norms.append(torch.stack([norm.float().cpu() for _, norm in both]))
            cell = (label(vision), label(language))
            result["chunks"][cell], result["chunks_norm"][cell] = torch.stack(chunks), torch.stack(norms)
            torch.save(result, cfg.out)
            print(f"vision {cell[0]} | language model {cell[1]}: {len(frames)} frames x {cfg.seeds} seeds, {time.time() - start:.0f} s", flush=True)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: CrossConfig):
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
