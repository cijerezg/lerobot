"""The positions with a large L2 norm: where the norm comes from, who attends to them, what they are worth.

    uv run python lerobot/adhocs/text_probe/layer_norms.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,<pretrained_model dir>,... --stage=trace --out=outputs/analysis/layer_cosine_2026-11-16/norms_trace.pt

A position has a high norm at a layer when the L2 norm of that block's output there is over 10 x the median over
the attended positions of the layer (``layer_edit.high_norm``).

``trace`` (per checkpoint and prompt: action prompts on the robot frames, "Describe this image." on the photos,
text-only prompts): the norm of every block's output, of what its attention and its MLP add, the language
model's attention from the text positions after the last image to each class of position, the probability of
the action tokens when each block's output is decoded through the final norm and lm_head at every position, and
for image prompts the token norms after every vision block, after pooling and after the projector.

``chunks`` (per checkpoint, frame and flow seed): the action chunk when the language model is edited (norm
clamped to the layer median at every block; the block updates that create a high norm undone), and when the
action expert alone is edited (a fraction of the high-norm positions hidden from its cross-attention; their
keys and values taken from another frame).
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
from layer_edit import attention, block_writes, clamp, edited, high_norm, updates
from layer_patch import chunk
from layer_sweep import TEXT_PROMPTS, Sweep, SweepConfig
from run_text_probe import label

LENS_LAYERS = [0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 34, 35]
FRACTIONS = [0.25, 0.5, 0.75, 1.0]


@dataclass
class NormsConfig(SweepConfig):
    out: str = "outputs/analysis/layer_cosine_2026-11-16/norms_trace.pt"
    stage: str = "trace"  # trace | chunks
    seeds: int = 2


def classes(run: dict) -> tuple[torch.Tensor, dict[str, list[int]]]:
    """(high [layers, S], {class: positions}). Classes of the high-norm positions: ``from block 0`` (over the
    threshold at the first block's output), ``later, image`` and ``later, text``; every other attended position
    goes by its group."""
    groups = run["groups"]
    high = high_norm(run["encoder"], groups)
    ever, first = high.any(dim=0), high[0]
    members: dict[str, list[int]] = {}
    for at, group in enumerate(groups):
        if group == "masked":
            continue
        if first[at]:
            name = "high norm from block 0"
        elif ever[at]:
            name = "high norm later, image" if group == "image" else "high norm later, text"
        else:
            name = group
        members.setdefault(name, []).append(at)
    return high, members


@torch.no_grad()
def decoded(sweep: Sweep, run: dict) -> dict:
    """Probability of the action tokens, and whether the most likely token is one, at every position when the
    output of the blocks ``LENS_LAYERS`` goes through the final norm and lm_head."""
    hf, config = sweep.policy._hf_model(), sweep.policy.model.config
    actions = slice(int(config.action_output_token_id), int(config.action_token_start_id) + int(config.num_action_tokens))
    mass, top = [], []
    for layer in LENS_LAYERS:
        hidden = run["encoder"][layer].to(sweep.device)
        prob = hf.lm_head(hf.model.transformer.ln_f(hidden.to(hf.lm_head.weight.dtype))).float().softmax(dim=-1)
        mass.append(prob[:, actions].sum(dim=-1).cpu())
        best = prob.argmax(dim=-1)
        top.append(((best >= actions.start) & (best < actions.stop)).cpu())
    return {"action_mass": torch.stack(mass), "action_top": torch.stack(top)}


def vision_norms(sweep: Sweep):
    """Hooks on the vision path: token norms after every vision block, the pooling and the projector."""
    vision = sweep.policy._backbone().vision_backbone
    kept = {}

    def keep(name):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            kept.setdefault(name, hidden.detach().float().reshape(-1, hidden.shape[-1]).norm(dim=-1).cpu())

        return hook

    handles = [block.register_forward_hook(keep(f"vit {i:02d}")) for i, block in enumerate(vision.image_vit.transformer.resblocks)]
    handles += [vision.image_pooling_2d.register_forward_hook(keep("pooled")), vision.image_projector.register_forward_hook(keep("projected"))]
    return kept, handles


def trace(sweep: Sweep, spec: dict) -> dict:
    forward = sweep.act if spec["kind"] == "action" else sweep.chat
    first = sweep.named(forward(spec))
    groups, positions = first["groups"], len(first["groups"])
    last_image = max((i for i, g in enumerate(groups) if g == "image"), default=-1)
    queries = [i for i, g in enumerate(groups) if i > last_image and g != "masked"]
    kept, handles = vision_norms(sweep)
    try:
        with updates(sweep.policy, positions) as added, attention(sweep.policy, positions, queries) as seen:
            run = sweep.named(forward(spec))
    finally:
        for handle in handles:
            handle.remove()
    high, members = classes(run)
    mass = seen.mass.float()  # [layers, queries, S]
    return {
        "spec": spec, "groups": groups, "tokens": run["tokens"], "norm": run["encoder"].float().norm(dim=-1), "high": high,
        "attn": added.attn, "mlp": added.mlp, "members": members, "queries": queries,
        "mass": {name: mass[:, :, index].sum(dim=-1) for name, index in members.items()},  # [layers, queries]
        "vision": kept, "embedding_note": "norm[0] is the output of block 0; 'projected' is what enters block 0 at the image positions",
    } | decoded(sweep, run)


def traces(sweep: Sweep) -> dict:
    specs = [{"name": f"act.{image}", "kind": "action", "image": image, "style": "step", "text": None} for image in sweep.robot]
    specs += [{"name": f"chat.{photo}", "kind": "chat", "image": photo, "text": "Describe this image."} for photo in sweep.photos]
    specs += [{"name": f"text.{key}", "kind": "chat", "image": None, "text": TEXT_PROMPTS[key]} for key in ("france", "times", "tubeplan")]
    return {spec["name"]: trace(sweep, spec) for spec in specs}


def chunks(sweep: Sweep) -> dict:
    """Chunks under edits, per frame and seed. Language-model edits change what every later block and the action
    expert read; the expert-only edits leave the language model alone."""
    result, names = {}, list(sweep.robot)
    runs = {image: sweep.named(sweep.act({"image": image, "style": "step", "text": None})) for image in names}
    for at, image in enumerate(names):
        spec = {"image": image, "style": "step", "text": None}
        run = runs[image]
        same = [n for n in names if len(runs[n]["groups"]) == len(run["groups"])]  # the other frame needs the same prompt length
        other = {"image": same[(same.index(image) + 1) % len(same)], "style": "step", "text": None}
        groups, positions = run["groups"], len(run["groups"])
        attended = torch.tensor([g != "masked" for g in groups])
        high, members = classes(run)
        ever = high.any(dim=0)
        image_positions = torch.tensor([g == "image" for g in groups])
        big = [i for i in range(positions) if ever[i]]
        ordinary = [i for i, g in enumerate(groups) if g == "image" and not ever[i]]
        order = torch.randperm(len(big), generator=torch.Generator().manual_seed(0)).tolist()
        pick = torch.randperm(len(ordinary), generator=torch.Generator().manual_seed(0)).tolist()
        for seed in range(sweep.cfg.seeds):
            keep, keep_other = [], []
            entry = {"clean": chunk(sweep, spec, seed, keep=keep), "count": len(big), "classes": {k: len(v) for k, v in members.items()}}
            entry["other seed"] = chunk(sweep, spec, seed + 100)
            chunk(sweep, other, seed, keep=keep_other)
            edits = {
                "lm: norm clamped": clamp(attended), "lm: writes undone": block_writes(attended),
                "lm: writes undone, image positions only": block_writes(attended, only=image_positions),
                "lm: writes undone, other positions only": block_writes(attended, only=~image_positions),
            }
            for name, fn in edits.items():
                with edited(sweep.policy, positions, fn):
                    entry[name] = chunk(sweep, spec, seed)
            for fraction in FRACTIONS:
                count = round(fraction * len(big))
                entry[f"expert: {fraction:.2f} of the high-norm positions hidden"] = chunk(sweep, spec, seed, hide=[big[i] for i in order[:count]])
                entry[f"expert: as many ordinary image positions hidden ({fraction:.2f})"] = chunk(sweep, spec, seed, hide=[ordinary[i] for i in pick[:count]])
            if other["image"] != image:
                entry["expert: high-norm keys and values from another frame"] = chunk(sweep, spec, seed, patch=(keep_other[0], big, range(36)))
                entry["expert: as many ordinary image keys and values from another frame"] = chunk(sweep, spec, seed, patch=(keep_other[0], [ordinary[i] for i in pick[: len(big)]], range(36)))
            result[(image, seed)] = entry
    return result


def run(cfg: NormsConfig) -> None:
    init_logging()
    sweep = Sweep(cfg, get_safe_torch_device(try_device=cfg.policy.device))
    result = {}
    for path in cfg.checkpoints.split(","):
        name = label(path).replace("/", "__")
        start = time.time()
        sweep.load(path)
        result[name] = traces(sweep) if cfg.stage == "trace" else chunks(sweep)
        sweep.free()
        torch.save(result, cfg.out)
        print(name, f"{time.time() - start:.0f} s", flush=True)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: NormsConfig):
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
