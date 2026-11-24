"""Text probe: can the language model inside MolmoAct2 still answer plain questions?

    uv run python lerobot/adhocs/text_probe/run_text_probe.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/000700/pretrained_model \
        --out_dir=outputs/analysis/text_probe/tube-ckpt700_2026-11-15

Each question goes in as a plain chat turn (``<|im_start|>user ... <|im_start|>assistant``, no robot
clauses, no action output token) and the answer is greedy-decoded through ``lm_head`` until
``<|im_end|>`` or ``max_new_tokens``. The action expert is never called. ``TEXT_QUESTIONS`` run without
an image, ordered from very common to rarer; ``IMAGE_QUESTIONS`` run with one top-camera frame of the
validation set. ``base`` in ``checkpoints`` is the foundation checkpoint (``policy.base_path``).
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

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import register_config_choices
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

TEXT_QUESTIONS = (
    "Hi, how are you?",
    "What is your name?",
    "What is 2 + 2?",
    "What color is the sky on a clear day?",
    "What is the capital of France?",
    "How many days are in a week?",
    "What is the opposite of hot?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in the solar system?",
    "What is the boiling point of water in degrees Celsius?",
    "Translate 'good morning' into Spanish.",
    "Who painted the Mona Lisa?",
    "In what year did the Second World War end?",
    "What is the chemical symbol for gold?",
    "What is 17 times 23?",
    "What is the capital of Australia?",
    "Write a Python function that returns the sum of a list.",
    "What is the derivative of x^3?",
    "Who proved Fermat's Last Theorem?",
    "What is the capital of Burkina Faso?",
)
IMAGE_QUESTIONS = (
    "Describe this image.",
    "What objects are on the table?",
    "What color is the tube?",
    "Point to the robot gripper.",
)
# suite "long": each entry is the list of user turns of one conversation (a single entry = one long prompt)
LONG_TEXT = (
    ["Alice, Bob and Carol each own exactly one pet: a cat, a dog and a parrot. Alice is allergic to fur. "
     "Bob does not own the dog. Who owns which pet? Explain your reasoning step by step."],
    ["A train leaves city A at 9:00 travelling at 80 km/h toward city B, which is 420 km away. A second train "
     "leaves city B at 10:00 travelling at 100 km/h toward city A. At what time do they meet, and how far from "
     "city A? Show the calculation."],
    ["Read this passage and answer the questions after it.\n\n"
     "In March 1911 the survey ship Halcyon left the port of Varnes with a crew of 34 under Captain Ingrid Solheim. "
     "Its orders were to chart the Meridian Shoals, a chain of sandbars that had wrecked eleven ships in the "
     "previous decade. Nineteen days out, a storm tore away the mizzen mast and flooded the forward hold, spoiling "
     "roughly a third of the provisions. Solheim put the crew on half rations and, instead of turning back, "
     "anchored in the lee of Tern Island for six days while the carpenter, a man named Oduya, built a replacement "
     "mast from two spare yards. The survey was finished in July, four weeks late, and the charts the Halcyon "
     "brought home stayed in use until 1958.\n\n"
     "1. How many crew were aboard, and who commanded the ship?\n2. What did the storm damage?\n"
     "3. Who repaired the mast, where, and from what?\n4. Summarize the passage in one sentence."],
    ["Write a Python class LRUCache with get(key) and put(key, value) that both run in O(1), with a fixed "
     "capacity given at construction. Then explain in a few sentences why both operations are O(1)."],
    ["This binary search has a bug. Find it, explain what goes wrong on which inputs, and give the fixed code.\n\n"
     "def search(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
     "        if a[mid] == x:\n            return mid\n        if a[mid] < x:\n            lo = mid\n"
     "        else:\n            hi = mid\n    return -1"],
    ["Explain the difference between behavior cloning and offline reinforcement learning for robot manipulation, "
     "and explain where advantage-weighted regression sits between the two. Keep it under 200 words."],
    ["Prove that the sum of the first n odd numbers equals n squared."],
    ["Write a four-line poem about a robot arm in which the lines start with the letters T, U, B, E in that order. "
     "Then translate the poem into French. Then state how many words the English version has."],
    ["A robot arm with a two-finger gripper stands at a table. Three small tubes lie flat on the table and three "
     "pegs stand upright. Write a step-by-step plan to put each tube on a peg, and for each step name one thing "
     "that could go wrong and how the robot could notice it."],
    ["My name is Carlos and I am working on a robot that slides small tubes onto pegs. Please remember the number 7421; "
     "I will ask for it later.",
     "What are three common failure modes of peg insertion tasks?",
     "Expand on the second one and give a concrete fix.",
     "Write a short Python function that flags that failure from a list of gripper torque readings.",
     "What number did I ask you to remember, and what is my name?",
     "Summarize our whole conversation in three bullet points."],
    ["Start with the number 12 and double it. What do you get?",
     "Subtract 5 from that.",
     "Multiply the result by 3.",
     "Is that number prime? Explain.",
     "List every intermediate value we computed, in order."],
)
LONG_IMAGE = (
    ["Describe the scene in detail. Count the small white objects and the transparent objects, and say where each "
     "one is relative to the robot."],
    ["What do you see in this image?",
     "How many small white objects are on the table?",
     "Which of them is closest to the robot gripper?",
     "If the robot had to pick one of them up first, which one should it take and why?",
     "Point to that object."],
)
# suite "switch": (question, with image, text already placed in the assistant turn), ordered from plain chat
# to the wording of the training prompt. The last rung ends on the action output token, as in training; the
# state / velocity / torque clauses of the real prompt are left out.
ROBOT_QUESTION = "The task is to put the tube in the peg. Given these, what action should the robot take to complete the task?"
SWITCH = (
    ("Hi, how are you?", False, ""),
    ("What is the capital of France?", False, ""),
    ("What is 17 times 23?", False, ""),
    ("How would a robot arm put a small tube on a peg?", False, ""),
    (ROBOT_QUESTION, False, ""),
    ("Describe this image.", True, ""),
    ("What objects are on the table?", True, ""),
    ("How many small white objects are on the table?", True, ""),
    ("Is the robot gripper open or closed?", True, ""),
    ("What should the robot do next?", True, ""),
    ("The task is to put the tube in the peg. What should the robot do next? Answer in one sentence.", True, ""),
    (ROBOT_QUESTION, True, ""),
    (ROBOT_QUESTION, True, "<action_output>"),
)
# suite "images": every question on the validation frame and on every image of image_dir (non-robot photos)
IMAGES_QUESTIONS = (
    "Describe this image.",
    "What objects are in this image?",
    "What should the robot do next?",
    ROBOT_QUESTION,
)
IMAGE_KEY = "observation.images.top"


@dataclass
class TextProbeConfig(TrainRLServerPipelineConfig):
    out_dir: str = "outputs/analysis/text_probe/tube-ckpt700_2026-11-15"
    checkpoints: str = "base"  # comma list; "base" = policy.base_path, anything else a pretrained_model dir
    frame: int = 600  # validation-set frame shown with IMAGE_QUESTIONS
    max_new_tokens: int = 96
    suite: str = "basic"  # "basic" = TEXT_QUESTIONS + IMAGE_QUESTIONS, "long" = LONG_TEXT + LONG_IMAGE (use --max_new_tokens=512), "switch" = SWITCH, "images" = IMAGES_QUESTIONS on image_dir
    image_dir: str = "outputs/data/coco_val2017_sample"  # suite "images": folder of non-robot photos
    mask_action_tokens: bool = False  # decode with the action tokens forbidden: the text the model gives when it cannot switch


@torch.no_grad()
def answer(policy, processor, turns: list[str], image, max_new_tokens: int, prefix: str = "", mask_action_tokens: bool = False):
    """turns: user, assistant, user, ... ending on the user turn to answer. Returns the answer and, per decoded
    position, the probability on the action tokens (action output / start / end + the FAST bins), and the five
    most likely tokens at the first answer position with their probabilities."""
    text = "<|image|>" if image is not None else ""
    for i, turn in enumerate(turns):
        text += f"<|im_start|>{'assistant' if i % 2 else 'user'}\n{turn}<|im_end|>\n"
    text += "<|im_start|>assistant\n" + prefix
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    device = next(policy.model.parameters()).device
    inputs = policy._model_inputs({key: value.to(device) for key, value in inputs.items()})
    inputs = policy._drop_trivial_attention_mask(inputs)
    output = policy.model(**inputs, use_cache=True)
    attention_mask = inputs.get("attention_mask")
    config = policy.model.config
    eos = int(config.eos_token_id)
    actions = slice(int(config.action_output_token_id), int(config.action_token_start_id) + int(config.num_action_tokens))
    tokens, action_mass = [], []
    top = output.logits[0, -1].float().softmax(dim=-1).topk(5)  # the five most likely first tokens
    first_top = [(processor.tokenizer.decode([int(i)]), float(p)) for p, i in zip(top.values, top.indices)]
    for _ in range(max_new_tokens):
        logits = output.logits[:, -1].float()
        action_mass.append(float(logits.softmax(dim=-1)[0, actions].sum()))
        if mask_action_tokens:
            logits[:, actions] = -torch.inf
        token = logits.argmax(dim=-1)
        if int(token) == eos:
            break
        tokens.append(int(token))
        output, attention_mask = policy.model._consume_generation_tokens(
            token, past_key_values=output.past_key_values, attention_mask=attention_mask
        )
    # special tokens stay in: a finetuned checkpoint that answers with action tokens should show it
    return processor.tokenizer.decode(tokens, skip_special_tokens=False).strip(), action_mass, first_top


def label(checkpoint: str) -> str:
    """outputs/runs/<run>/checkpoints/<step>/pretrained_model -> <run>/<step>"""
    parts = checkpoint.rstrip("/").split("/")
    return checkpoint if checkpoint == "base" else f"{parts[-4]}/{parts[-2]}"


def run(cfg: TextProbeConfig) -> None:
    init_logging()
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)
    image = (dataset[cfg.frame][IMAGE_KEY].permute(1, 2, 0) * 255).round().byte().numpy()
    Image.fromarray(image).save(os.path.join(cfg.out_dir, "frame.png"))
    if cfg.suite == "basic":
        conversations = [([q], None, "") for q in TEXT_QUESTIONS] + [([q], image, "") for q in IMAGE_QUESTIONS]
    elif cfg.suite == "long":
        conversations = [(c, None, "") for c in LONG_TEXT] + [(c, image, "") for c in LONG_IMAGE]
    elif cfg.suite == "images":
        photos = {"robot frame": image}
        for name in sorted(os.listdir(cfg.image_dir)):
            photos[name] = np.asarray(Image.open(os.path.join(cfg.image_dir, name)).convert("RGB"))
        conversations = [([q], photo, "") for photo in photos.values() for q in IMAGES_QUESTIONS]
    else:
        conversations = [([q], image if with_image else None, prefix) for q, with_image, prefix in SWITCH]
    # one row per user turn; a later turn of a conversation is marked with its turn number
    questions = [
        ((q if len(c) == 1 else f"[turn {t + 1}] {q}") + (f" [assistant starts with {prefix}]" if prefix else ""), img)
        for c, img, prefix in conversations
        for t, q in enumerate(c)
    ]
    if cfg.suite == "images":  # the table row names the image; the model sees only the question
        questions = [(f"[{name}] {q}", photo) for name, photo in photos.items() for q in IMAGES_QUESTIONS]

    checkpoints = [label(c) for c in cfg.checkpoints.split(",")]
    answers, masses = {}, {}
    for checkpoint, path in zip(checkpoints, cfg.checkpoints.split(",")):
        cfg.policy.pretrained_path = None if path == "base" else path
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
        adapter._set_probe_cuda_graph_enabled(False)
        processor = next(s for s in adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)).processor
        answers[checkpoint], masses[checkpoint] = [], []
        for user_turns, img, prefix in conversations:
            turns = []
            for question in user_turns:
                turns.append(question)
                text, action_mass, _ = answer(
                    adapter.policy, processor, turns, img, cfg.max_new_tokens, prefix, cfg.mask_action_tokens
                )
                turns.append(text)  # the model's own answer is the history of the next turn
                answers[checkpoint].append(text)
                masses[checkpoint].append(action_mass)
                logging.info(f"[text_probe] {checkpoint} | {question} | first {action_mass[0]:.3f} max {max(action_mass):.3f} -> {text!r}")
        del adapter
        gc.collect()
        torch.cuda.empty_cache()

    rows = [
        {
            "question": q,
            "image": img is not None,
            **{c: answers[c][i] for c in checkpoints},
            "action_mass": {c: masses[c][i] for c in checkpoints},
        }
        for i, (q, img) in enumerate(questions)
    ]
    with open(os.path.join(cfg.out_dir, "answers.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    def cell(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = ["| # | image | question | " + " | ".join(checkpoints) + " |", "|---|---|---|" + "---|" * len(checkpoints)]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {'yes' if row['image'] else ''} | {cell(row['question'])} | "
            + " | ".join(cell(row[c]) for c in checkpoints) + " |"
        )
    with open(os.path.join(cfg.out_dir, "answers.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # probability on the action tokens: at the first answer position / the largest along the answer
    lines = lines[:2]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {'yes' if row['image'] else ''} | {cell(row['question'])} | "
            + " | ".join(f"{row['action_mass'][c][0]:.3f} / {max(row['action_mass'][c]):.3f}" for c in checkpoints) + " |"
        )
    with open(os.path.join(cfg.out_dir, "action_mass.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    logging.info(f"[text_probe] wrote {cfg.out_dir}/answers.md ({len(rows)} questions x {len(checkpoints)} checkpoints)")


@parser.wrap()
def cli(cfg: TextProbeConfig):
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
