"""Where in the vision path a finetuned checkpoint's image tokens leave the base model's.

    uv run python lerobot/adhocs/text_probe/layer_vision.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,<pretrained_model dir>,... --out=outputs/analysis/layer_cosine_2026-11-16/vision.pt

The first checkpoint is the reference. One image per forward (the photos and the top camera of the robot frames).
Stages: the output of every vision transformer block, the pooled features (image_pooling_2d), the projected
features (image_projector, what the language model receives as image tokens). Per stage and image, against the
reference: raw cosine of matched tokens (mean and median), relative L2 distance, the same cosine for shuffled
tokens, and the mean token length in each checkpoint.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

from lerobot.configs import parser
from lerobot.probes.utils import register_config_choices
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from layer_sweep import Sweep, SweepConfig
from run_text_probe import label


@dataclass
class VisionConfig(SweepConfig):
    out: str = "outputs/analysis/layer_cosine_2026-11-16/vision.pt"


@torch.no_grad()
def stages(sweep: Sweep, image: str) -> dict[str, torch.Tensor]:
    """{stage: [tokens, width]} for one image."""
    vision = sweep.policy._backbone().vision_backbone
    kept = {}

    def keep(name):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            kept.setdefault(name, hidden.detach().float().reshape(-1, hidden.shape[-1]).cpu())

        return hook

    handles = [block.register_forward_hook(keep(f"vit {i:02d}")) for i, block in enumerate(vision.image_vit.transformer.resblocks)]
    handles += [vision.image_pooling_2d.register_forward_hook(keep("pooled")), vision.image_projector.register_forward_hook(keep("projected"))]
    try:
        sweep.chat({"image": image, "text": "Describe the image."})
    finally:
        for handle in handles:
            handle.remove()
    return kept


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    shuffled = torch.nn.functional.cosine_similarity(a, b.roll(len(b) // 2 + 1, dims=0), dim=-1)
    return {
        "cos": float(cos.mean()), "cos_median": float(cos.median()), "null_cos": float(shuffled.mean()),
        "rel": float(((a - b).norm(dim=-1) / ((a.norm(dim=-1) + b.norm(dim=-1)) / 2)).mean()),
        "norm_a": float(a.norm(dim=-1).mean()), "norm_b": float(b.norm(dim=-1).mean()),
    }


def run(cfg: VisionConfig) -> None:
    init_logging()
    sweep = Sweep(cfg, get_safe_torch_device(try_device=cfg.policy.device))
    reference, result = None, {}
    for path in cfg.checkpoints.split(","):
        name = label(path).replace("/", "__")
        sweep.load(path)
        kept = {image: stages(sweep, image) for image in sweep.images}
        sweep.free()
        if reference is None:
            reference = kept
            continue
        for image, run_stages in kept.items():
            for stage, hidden in run_stages.items():
                result[(name, image, stage)] = compare(reference[image][stage], hidden)
        print(name, flush=True)
        torch.save(result, cfg.out)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: VisionConfig):
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
