"""Which prompt positions carry the instruction to the action expert, and what the high-norm positions are worth to it.

    uv run python lerobot/adhocs/text_probe/layer_patch.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,<pretrained_model dir>,... --frames=100,300,... --clutter_frames=500,1500,... \
        --out=outputs/analysis/layer_cosine_2026-11-16/patch.pt

Per checkpoint, frame, instruction pair (a, b) and flow seed: the chunk under a, the chunk under b, and the chunk
under a with part of what the action expert reads (its projected keys and values, all 36 layers) taken from the
run under b: one kind of position at a time, or every position but only a band of layers. The language model
itself is never touched. Then the chunk under a with some positions removed from the action expert's mask: the
high-norm positions (L2 norm over 10 x the layer median at any layer; saved under the key ``massive``), and as
many random image positions.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import torch

from lerobot.configs import parser
from lerobot.probes.utils import register_config_choices
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from layer_analysis import FACTOR
from layer_sweep import INSTRUCTIONS, Sweep, SweepConfig
from run_text_probe import label

PAIRS = [("left", "right"), ("tubeleft", "tuberight")]
BANDS = {"layers 0-11": range(0, 12), "layers 12-17": range(12, 18), "layers 18-23": range(18, 24), "layers 24-29": range(24, 30), "layers 30-35": range(30, 36)}


@dataclass
class PatchConfig(SweepConfig):
    out: str = "outputs/analysis/layer_cosine_2026-11-16/patch.pt"
    seeds: int = 2


def chunk(sweep: Sweep, spec: dict, seed: int, hide: list[int] | None = None, patch: tuple | None = None, keep: list | None = None) -> torch.Tensor:
    """One action forward. ``patch`` = (context of another run, positions, layers): the action expert's keys and
    values at those positions and layers come from the other run. ``keep`` receives this run's own context."""
    policy = sweep.policy
    expert = policy._backbone()._require_action_expert()
    frame, obs, metadata = sweep.action_inputs(spec["image"], spec["style"])
    task, subtask = sweep.words(spec)
    prepare = expert.prepare_context

    def prepared(**kwargs):
        if hide:
            mask = kwargs.get("encoder_attention_mask")
            positions = kwargs["encoder_kv_states"][0][0].shape[1]
            mask = torch.ones(kwargs["batch_size"], positions, dtype=torch.bool, device=kwargs["device"]) if mask is None else mask.clone()
            mask[:, hide] = False
            kwargs["encoder_attention_mask"] = mask
        context = prepare(**kwargs)
        if keep is not None and not keep:
            keep.append([(k.clone(), v.clone()) for k, v in context.kv_contexts])
        if patch is not None:
            other, positions, layers = patch
            swapped = []
            for index, (k, v) in enumerate(context.kv_contexts):
                if index in layers:
                    k, v = k.clone(), v.clone()
                    k[:, positions], v[:, positions] = other[index][0][:, positions], other[index][1][:, positions]
                swapped.append((k, v))
            context.kv_contexts = swapped
        return context

    expert.prepare_context = prepared
    try:
        out, _, _ = sweep.adapter.predict_action_chunk_in_mode(
            obs, task, inference_action_mode="continuous", state=frame["state"], subtask=subtask,
            metadata=metadata, generator=torch.Generator(device=sweep.device).manual_seed(seed),
        )
    finally:
        del expert.prepare_context
    return out


def run(cfg: PatchConfig) -> None:
    init_logging()
    sweep = Sweep(cfg, get_safe_torch_device(try_device=cfg.policy.device))
    result = {}
    for path in cfg.checkpoints.split(","):
        name = label(path).replace("/", "__")
        start = time.time()
        sweep.load(path)
        for image in sweep.robot:
            for a, b in PAIRS:
                spec_a = {"image": image, "style": "step", "text": INSTRUCTIONS[a]}
                spec_b = {"image": image, "style": "step", "text": INSTRUCTIONS[b]}
                first = sweep.named(sweep.act(spec_a))
                groups = first["groups"]
                kinds = {kind: [i for i, g in enumerate(groups) if g == kind] for kind in dict.fromkeys(groups) if kind not in ("image", "masked")}
                after = max(kinds["subtask"]) + 1
                sets = {
                    "subtask": kinds["subtask"], "state": kinds.get("state", []), "velocity": kinds.get("velocity", []),
                    "torque + tip": kinds.get("gripper_torque", []) + kinds.get("tip_target", []), "question": kinds["question"][:-1] if groups[-1] == "question" else kinds["question"],
                    "tail": [i for i in range(max(kinds["question"]) + 1, len(groups))] or [len(groups) - 1],
                    "all after the instruction": list(range(after, len(groups))), "all": list(range(len(groups))),
                }
                attended = [i for i, g in enumerate(groups) if g != "masked"]
                norm = first["encoder"].float().norm(dim=-1)  # [layers, positions]
                big = (norm > FACTOR * norm[:, attended].median(dim=1, keepdim=True).values).any(dim=0)
                big = [i for i in attended if big[i]]
                images = [i for i, g in enumerate(groups) if g == "image" and i not in big]
                pick = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))[: len(big)].tolist()
                hides = {"massive": big, "massive, image only": [i for i in big if groups[i] == "image"], "random image": [images[i] for i in pick]}
                for seed in range(cfg.seeds):
                    keep_a, keep_b = [], []
                    entry = {"a": chunk(sweep, spec_a, seed, keep=keep_a), "b": chunk(sweep, spec_b, seed, keep=keep_b), "patch": {}, "hide": {},
                             "counts": {k: len(v) for k, v in (sets | hides).items()}}
                    if keep_a[0][0][0].shape == keep_b[0][0][0].shape:
                        for key, positions in sets.items():
                            if positions:
                                entry["patch"][key] = chunk(sweep, spec_a, seed, patch=(keep_b[0], positions, range(36)))
                        for key, layers in BANDS.items():
                            entry["patch"][key] = chunk(sweep, spec_a, seed, patch=(keep_b[0], sets["all"], layers))
                    for key, positions in hides.items():
                        entry["hide"][key] = chunk(sweep, spec_a, seed, hide=positions)
                    result[(name, image, a, b, seed)] = entry
            print(name, image, f"{time.time() - start:.0f} s", flush=True)
        sweep.free()
        torch.save(result, cfg.out)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: PatchConfig):
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
