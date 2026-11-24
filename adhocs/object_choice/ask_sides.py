"""Object-choice scenes, the question side: does the vision encoder + language model of a cell know where each object is?

    uv run python lerobot/adhocs/object_choice/ask_sides.py --config_path=config_rl_validate.yaml

Every placement's top-camera start image (scripts/sim_builders/object_choice/renders) is shown to each cell as a plain chat turn (as
text_probe: no robot clauses, action tokens forbidden, greedy). Per object two questions with a one-word answer: its side in the
IMAGE and its side from the ROBOT's point of view (the top camera faces the robot, so the two answers are opposite); the truth is
the scene's objects.json. One open question ("What objects are on the table?") is stored unscored. The action expert is never
called, so the base_vision_lm cell answers as the released model does.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "text_probe"))
from run_text_probe import answer  # noqa: E402

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import register_config_choices
from lerobot.robots.rebot_b601_sim.tasks import _CHOICE_PAIRS, _TWIN_BASKET, SCENES_DIR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

ROOT = "outputs/analysis/object_choice_2026-11-17"
RENDERS = "lerobot/scripts/sim_builders/object_choice/renders"
QUESTIONS = {
    "image": "In this image, is {noun} on the left side or on the right side of the image? Answer with one word: left or right.",
    "robot": "This image shows a robot arm at a table, seen from a camera that faces the robot. From the robot's own point of view, "
             "is {noun} on the robot's left or on the robot's right? Answer with one word: left or right.",
}
OPEN = "What objects are on the table?"


@dataclass
class AskSidesConfig(TrainRLServerPipelineConfig):
    out_dir: str = f"{ROOT}/ask_sides"
    # comma list of name=pretrained_model dir
    cells: str = (f"base_vision_lm={ROOT}/checkpoints/base_vision_lm/pretrained_model,base_vision={ROOT}/checkpoints/base_vision/pretrained_model,"
                  "internal=outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model")
    max_new_tokens: int = 48


def side(text: str) -> str | None:
    """left / right if the answer names exactly one of them."""
    found = [word for word in ("left", "right") if word in text.lower()]
    return found[0] if len(found) == 1 else None


def run(cfg: AskSidesConfig) -> None:
    init_logging()
    os.makedirs(cfg.out_dir)  # an existing folder stops the script
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)  # only for the policy's construction
    items = []  # (pair, placement, image, body or None, kind, question, truth)
    for pair, bodies in _CHOICE_PAIRS.items():
        for k in range(1, 7):
            image = np.asarray(Image.open(os.path.join(RENDERS, f"choice_{pair}_p{k}_ext.png")).convert("RGB"))
            sides = json.loads((SCENES_DIR / f"choice_{pair}_ask_pill_p{k}" / "objects.json").read_text())["sides"]
            items.append((pair, k, image, None, "open", OPEN, None))
            for body in bodies:
                for kind, question in QUESTIONS.items():
                    items.append((pair, k, image, body, kind, question.format(noun=_TWIN_BASKET[body][0]), sides[body][f"side_{kind}"]))
    rows = []
    for cell, path in (entry.split("=") for entry in cfg.cells.split(",")):
        cfg.policy.pretrained_path = path
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
        adapter._set_probe_cuda_graph_enabled(False)
        processor = next(s for s in adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)).processor
        for pair, k, image, body, kind, question, truth in items:
            text, action_mass, _ = answer(adapter.policy, processor, [question], image, cfg.max_new_tokens, mask_action_tokens=True)
            rows.append(dict(cell=cell, pair=pair, placement=k, object=body, kind=kind, question=question, answer=text, said=side(text) if truth else None,
                             truth=truth, action_mass_first=round(action_mass[0], 4)))
            logging.info(f"[ask_sides] {cell} {pair} p{k} {kind} {body}: {text!r} (truth {truth})")
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    with open(os.path.join(cfg.out_dir, "answers.json"), "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    logging.info(f"[ask_sides] wrote {cfg.out_dir}/answers.json; tables: lerobot/adhocs/object_choice/summary.py")


@parser.wrap()
def cli(cfg: AskSidesConfig):
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
