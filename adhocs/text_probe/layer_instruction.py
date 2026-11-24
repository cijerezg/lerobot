"""Where the instruction travels inside the language model, and whether the action expert can still respond to it.

    uv run python lerobot/adhocs/text_probe/layer_instruction.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,<pretrained_model dir>,... --frames=... --clutter_frames=... \
        --out=outputs/analysis/layer_cosine_2026-11-16/instruction.pt

Per checkpoint, frame, instruction pair (a, b) and flow seed, always starting from the run under a:

``residual``: the output of ONE language-model block is replaced by the run under b at some positions (the
instruction tokens, or every position after the instruction); every later block and the action expert then
read the edited stream. Says by which block the instruction has left its own tokens.

``scale``: the keys and values the action expert reads from the language model (before its own projection) are
set to a + alpha x (b - a) at every layer and position. alpha = 1 is the run under b. Says whether the expert
responds when the difference is larger than the language model makes it.

``--stage=donors`` (own output file): the action expert of one checkpoint on the keys and values of ANOTHER
checkpoint's language model, same frame and pair (checkpoints with the same prompt length only): its own under
a plus the other's (b - a), and the other's under a and under b. Says whether this checkpoint's expert responds
to the instruction as another checkpoint's language model writes it.

Also kept per layer: the relative size of the difference between the two runs, in the residual stream and in
the raw values, per group of position.
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
from layer_edit import edited, replace
from layer_sweep import INSTRUCTIONS, Sweep, SweepConfig
from run_text_probe import label

PAIRS = [("left", "right"), ("tubeleft", "tuberight")]
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32]
ALPHAS = [0.5, 2.0, 4.0, 8.0]


@dataclass
class InstructionConfig(SweepConfig):
    out: str = "outputs/analysis/layer_cosine_2026-11-16/instruction.pt"
    seeds: int = 2
    stage: str = "own"  # own | donors (give donors its own ``out``)


def chunk(sweep: Sweep, spec: dict, seed: int, keep: list | None = None, change=None) -> torch.Tensor:
    """One action forward. ``keep`` receives the language model's keys and values as the expert gets them,
    [(k, v)] per layer; ``change`` maps that list to the one the expert is given instead."""
    expert = sweep.policy._backbone()._require_action_expert()
    frame, obs, metadata = sweep.action_inputs(spec["image"], spec["style"])
    task, subtask = sweep.words(spec)
    prepare = expert.prepare_context

    def prepared(**kwargs):
        if keep is not None and not keep:
            keep.append([(k.detach(), v.detach()) for k, v in kwargs["encoder_kv_states"]])
        if change is not None:
            kwargs["encoder_kv_states"] = change(kwargs["encoder_kv_states"])
        return prepare(**kwargs)

    expert.prepare_context = prepared
    try:
        out, _, _ = sweep.adapter.predict_action_chunk_in_mode(
            obs, task, inference_action_mode="continuous", state=frame["state"], subtask=subtask,
            metadata=metadata, generator=torch.Generator(device=sweep.device).manual_seed(seed),
        )
    finally:
        del expert.prepare_context
    return out


def one_layer(layer: int, positions: list[int], other: torch.Tensor):
    mask = torch.zeros(other.shape[:2], dtype=torch.bool)
    mask[layer, positions] = True
    return replace(mask, other)


def relative(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|b - a| / |a| per position, [..., S]"""
    return (b.float() - a.float()).norm(dim=-1) / a.float().norm(dim=-1).clamp_min(1e-6)


def own(sweep: Sweep, differences: dict, name: str) -> dict:
    result = {}
    for image in sweep.robot:
        for a, b in PAIRS:
            spec_a = {"image": image, "style": "step", "text": INSTRUCTIONS[a]}
            spec_b = {"image": image, "style": "step", "text": INSTRUCTIONS[b]}
            run_a, run_b = sweep.named(sweep.act(spec_a)), sweep.named(sweep.act(spec_b))
            groups = run_a["groups"]
            if groups != run_b["groups"]:
                continue
            subtask = [i for i, g in enumerate(groups) if g == "subtask"]
            after = list(range(max(subtask) + 1, len(groups)))
            members = {g: [i for i, x in enumerate(groups) if x == g] for g in dict.fromkeys(groups) if g not in ("masked", "image")}
            members["image after the instruction"] = [i for i in after if groups[i] == "image"]
            for seed in range(sweep.cfg.seeds):
                keep_a, keep_b = [], []
                entry = {"a": chunk(sweep, spec_a, seed, keep=keep_a), "b": chunk(sweep, spec_b, seed, keep=keep_b), "residual": {}, "scale": {}}
                raw_a, raw_b = keep_a[0], keep_b[0]
                for kind, positions in (("subtask", subtask), ("after", after)):
                    for layer in LAYERS:
                        with edited(sweep.policy, len(groups), one_layer(layer, positions, run_b["encoder"])):
                            entry["residual"][(kind, layer)] = chunk(sweep, spec_a, seed)
                for alpha in ALPHAS:
                    entry["scale"][alpha] = chunk(sweep, spec_a, seed, change=lambda _kv, alpha=alpha: [
                        (ka + alpha * (kb - ka), va + alpha * (vb - va)) for (ka, va), (kb, vb) in zip(raw_a, raw_b)])
                if seed == 0:
                    stream = relative(run_a["encoder"], run_b["encoder"])  # [layers, S]
                    values = torch.stack([relative(va.reshape(-1, va.shape[-1]), vb.reshape(-1, vb.shape[-1])) for (_, va), (_, vb) in zip(raw_a, raw_b)]).cpu()
                    entry["stream"] = {g: stream[:, index].mean(dim=1) for g, index in members.items() if index}
                    entry["values"] = {g: values[:, index].mean(dim=1) for g, index in members.items() if index}
                    differences[(name, image, a, b)] = [((kb - ka).cpu(), (vb - va).cpu()) for (ka, va), (kb, vb) in zip(raw_a, raw_b)]
                result[(image, a, b, seed)] = entry
    return result


def raw(sweep: Sweep) -> dict:
    """The language model's keys and values under both instructions of every pair, on the CPU."""
    kept = {}
    for image in sweep.robot:
        for a, b in PAIRS:
            keep_a, keep_b = [], []
            chunk(sweep, {"image": image, "style": "step", "text": INSTRUCTIONS[a]}, 0, keep=keep_a)
            chunk(sweep, {"image": image, "style": "step", "text": INSTRUCTIONS[b]}, 0, keep=keep_b)
            kept[(image, a, b)] = ([(k.cpu(), v.cpu()) for k, v in keep_a[0]], [(k.cpu(), v.cpu()) for k, v in keep_b[0]])
    return kept


def donors(sweep: Sweep, kept: dict, name: str) -> dict:
    """This checkpoint's action expert on another checkpoint's keys and values (same prompt length only):
    ``difference`` = own keys and values under a + the other's (b - a); ``a`` / ``b`` = the other's under a / b."""
    result = {}
    for (image, a, b), (own_a, _) in kept[name].items():
        spec_a = {"image": image, "style": "step", "text": INSTRUCTIONS[a]}
        spec_b = {"image": image, "style": "step", "text": INSTRUCTIONS[b]}
        for seed in range(sweep.cfg.seeds):
            entry = {"a": chunk(sweep, spec_a, seed), "b": chunk(sweep, spec_b, seed), "donor": {}}
            for other, runs in kept.items():
                if other == name or runs[(image, a, b)][0][0][0].shape != own_a[0][0].shape:
                    continue
                other_a, other_b = runs[(image, a, b)]
                given = lambda kv, states: [(k.to(own.device), v.to(own.device)) for (k, v), (own, _) in zip(states, kv)]  # noqa: E731
                entry["donor"][other] = {
                    "difference": chunk(sweep, spec_a, seed, change=lambda kv: [
                        (k + (kb - ka).to(k.device), v + (vb - va).to(v.device)) for (k, v), (ka, va), (kb, vb) in zip(kv, other_a, other_b)]),
                    "a": chunk(sweep, spec_a, seed, change=lambda kv: given(kv, other_a)),
                    "b": chunk(sweep, spec_a, seed, change=lambda kv: given(kv, other_b)),
                }
            result[(image, a, b, seed)] = entry
    return result


def run(cfg: InstructionConfig) -> None:
    init_logging()
    sweep = Sweep(cfg, get_safe_torch_device(try_device=cfg.policy.device))
    result, kept = {}, {}
    paths = cfg.checkpoints.split(",")
    for path in paths:
        name = label(path).replace("/", "__")
        start = time.time()
        sweep.load(path)
        if cfg.stage == "own":
            result[name] = own(sweep, {}, name)
            torch.save(result, cfg.out)
        else:
            kept[name] = raw(sweep)
        sweep.free()
        print(name, f"{time.time() - start:.0f} s", flush=True)
    if cfg.stage == "donors":
        for path in paths:
            name = label(path).replace("/", "__")
            sweep.load(path)
            result[name] = donors(sweep, kept, name)
            sweep.free()
            torch.save(result, cfg.out)
            print(name, "donors", flush=True)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: InstructionConfig):
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
