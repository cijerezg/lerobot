"""Scored language and image questions per checkpoint, plain and with the language model or the vision side edited.

    uv run python lerobot/adhocs/text_probe/layer_ability.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,<pretrained_model dir>,... --photos=20 --out=outputs/analysis/layer_cosine_2026-11-16/ability.pt

Three question sets, all as plain chat turns (no robot clauses):

``yes / no``: "Is there a <thing> in this image? Answer yes or no." on the photos of ``image_dir``. Scored on
the logits of the first answer position: yes when the best "yes" token is above the best "no" token. The action
tokens play no part in the score; their probability at that position is kept next to it.

``open``: one question per photo with a one-word answer; ``facts``: text-only questions. Both decoded greedily
with the action tokens forbidden, and scored by a pattern on the answer. The probability of the action tokens
at the first answer position (without the mask) is kept.

Conditions (the answers of a checkpoint under each): ``plain``; the four edits of the high-norm positions inside
the language model (``layer_edit``), with a control that undoes as many block updates at ordinary positions;
``vision from <other>``: the vision tower, pooling and projector of another checkpoint under this checkpoint's
language model (``base`` under every finetuned one, every finetuned one under ``base``; ``base`` must be first
in ``checkpoints``). With the swapped vision side the action chunk on the robot frames is kept too.

The truth of the photo questions was written by hand from the 20 photos of ``outputs/data/coco_val2017_sample``.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass

import torch

from lerobot.configs import parser
from lerobot.probes.utils import register_config_choices
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from layer_edit import _over, block_writes, clamp, edited, high_norm, mean_token
from layer_patch import chunk
from layer_sweep import Sweep, SweepConfig
from run_text_probe import label

# photo id -> (things in the photo, things not in it)
PRESENT = {
    139: (["television", "chair", "vase"], ["dog", "cat", "bicycle"]),
    285: (["bear"], ["car", "person", "fish"]),
    632: (["bed", "book"], ["television", "cat"]),
    776: (["teddy bear"], ["cat", "car", "person"]),
    785: (["person", "pair of skis"], ["dog", "car"]),
    802: (["refrigerator", "stove"], ["person", "cat"]),
    872: (["person", "baseball glove"], ["car", "dog"]),
    885: (["tennis racket", "person"], ["dog", "car"]),
    1000: (["tennis racket", "person"], ["dog", "car"]),
    1268: (["bird", "bridge"], ["dog", "horse"]),
    1296: (["person", "phone"], ["dog", "car"]),
    1425: (["sandwich", "plate"], ["pizza", "dog"]),
    1532: (["car", "road sign"], ["train", "airplane"]),
    1584: (["bus", "tree"], ["airplane", "dog"]),
    1675: (["cat", "laptop"], ["dog", "person"]),
    1761: (["airplane", "bridge"], ["dog", "helicopter"]),
    1818: (["zebra"], ["person", "car", "giraffe"]),
    1993: (["bed", "chair"], ["television", "person"]),
    2006: (["bus", "person"], ["airplane", "dog"]),
    2149: (["apple", "bowl"], ["banana", "person"]),
}
OPEN = [
    (285, "What animal is in this image?", r"bear"), (785, "What sport is this?", r"ski"), (785, "What color is the person's jacket?", r"red"),
    (802, "What color is the refrigerator?", r"white"), (872, "What sport is being played?", r"baseball"), (885, "What sport is being played?", r"tennis"),
    (1268, "What kind of bird is in the water?", r"swan"), (1425, "What food is on the plate?", r"sandwich"), (1584, "What color is the bus?", r"red"),
    (1675, "What animal is in this image?", r"cat"), (1761, "How many airplanes are in this image?", r"two|\b2\b"), (1818, "What animal is in this image?", r"zebra"),
    (2006, "What color is the bus?", r"purple|violet"), (2149, "What fruit is in the bowl?", r"apple"), (2149, "What color is the fruit?", r"green"),
    (632, "What color is the blanket on the bed?", r"blue"), (139, "What room is this?", r"living|dining"), (802, "What room is this?", r"kitchen"),
    (1993, "What room is this?", r"bedroom"), (776, "What kind of toys are in this image?", r"bear|teddy"),
]
FACTS = [
    ("What is the capital of France?", r"paris"), ("What is 2 + 2?", r"\b4\b|four"), ("What color is the sky on a clear day?", r"blue"),
    ("How many days are in a week?", r"\b7\b|seven"), ("What is the opposite of hot?", r"cold"), ("Who wrote Romeo and Juliet?", r"shakespeare"),
    ("What is the largest planet in the solar system?", r"jupiter"), ("What is the boiling point of water in degrees Celsius?", r"\b100\b"),
    ("Translate 'good morning' into Spanish.", r"buenos"), ("Who painted the Mona Lisa?", r"vinci|leonardo"),
    ("In what year did the Second World War end?", r"1945"), ("What is the chemical symbol for gold?", r"\bau\b"), ("What is 17 times 23?", r"\b391\b"),
    ("What is the capital of Australia?", r"canberra"), ("What is the derivative of x^3?", r"3\s*\*?\s*x\s*(\^|\*\*)\s*\{?2|3x²"),
    ("Who proved Fermat's Last Theorem?", r"wiles"), ("What is the capital of Burkina Faso?", r"ouagadougou"), ("What is the capital of Japan?", r"tokyo"),
    ("How many legs does a spider have?", r"\b8\b|eight"), ("What is H2O commonly called?", r"water"), ("What is the largest ocean on Earth?", r"pacific"),
    ("Who wrote the novel 1984?", r"orwell"), ("What is the square root of 144?", r"\b12\b|twelve"), ("How many continents are there?", r"\b7\b|seven"),
    ("Which planet is closest to the Sun?", r"mercury"), ("What language is spoken in Brazil?", r"portuguese"),
    ("Who was the first president of the United States?", r"washington"), ("What is the chemical symbol for sodium?", r"\bna\b"),
    ("What is 9 times 8?", r"\b72\b"), ("What is the capital of Italy?", r"rome"), ("What is the plural of mouse?", r"mice"),
    ("Who developed the theory of general relativity?", r"einstein"), ("What color do you get when you mix blue and yellow?", r"green"),
    ("What is the tallest mountain on Earth?", r"everest"), ("How many minutes are in an hour?", r"\b60\b|sixty"),
]


@dataclass
class AbilityConfig(SweepConfig):
    out: str = "outputs/analysis/layer_cosine_2026-11-16/ability.pt"
    photos: int = 20
    seeds: int = 2
    edits: bool = True  # false: only ``plain`` and the vision swaps
    max_new_tokens: int = 16


def ordinary_writes(attended: torch.Tensor):
    """Control for ``block_writes``: at every block, the update is undone at as many positions as are over the
    threshold there, drawn among the attended positions that are not."""

    def fn(layer, before, after):
        over, _ = _over(after, attended.to(after.device), 10.0)
        row = over.reshape(-1, over.shape[-1])[0]
        rest = (attended.to(after.device) & ~row).nonzero().flatten()
        pick = rest[torch.randperm(len(rest), generator=torch.Generator().manual_seed(layer))[: int(row.sum())].to(rest.device)]
        chosen = torch.zeros_like(row)
        chosen[pick] = True
        return torch.where(chosen.reshape(*[1] * (after.dim() - 2), -1, 1), before, after)

    return fn


def edits(attended: torch.Tensor, image: torch.Tensor) -> dict:
    out = {
        "norm clamped to the median, every block": clamp(attended),
        "block updates that create a high norm undone": block_writes(attended),
        "same, image positions only": block_writes(attended, only=image),
        "same, text positions only": block_writes(attended, only=~image),
        "control: as many updates undone at ordinary positions": ordinary_writes(attended),
    }
    if bool(image.any()):
        out["high-norm image tokens of block 0 set to the mean image token"] = mean_token(attended, image)
    return out


class Asker:
    def __init__(self, sweep: Sweep):
        self.sweep, tokenizer = sweep, sweep.processor.tokenizer
        config = sweep.policy.model.config
        self.actions = slice(int(config.action_output_token_id), int(config.action_token_start_id) + int(config.num_action_tokens))
        self.eos = int(config.eos_token_id)
        first = lambda words: sorted({tokenizer.encode(word, add_special_tokens=False)[0] for word in words})  # noqa: E731
        self.yes, self.no = first(["Yes", "yes", " Yes", " yes"]), first(["No", "no", " No", " no"])
        self.patch = sweep.adapter._image_patch_token_id()

    @torch.no_grad()
    def ask(self, question: str, image, condition: str, new_tokens: int) -> dict:
        """First-position logits and the greedy answer with the action tokens forbidden."""
        policy, processor = self.sweep.policy, self.sweep.processor
        picture = self.sweep.images[image] if image is not None else None
        text = ("<|image|>" if picture is not None else "") + f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=text, images=picture, return_tensors="pt", padding=True)
        inputs = policy._model_inputs({key: value.to(self.sweep.device) for key, value in inputs.items()})
        ids, mask = inputs["input_ids"][0], inputs.get("attention_mask")
        attended = torch.ones(len(ids), dtype=torch.bool) if mask is None else mask[0].bool().cpu()
        inputs = policy._drop_trivial_attention_mask(inputs)
        attention_mask = inputs.get("attention_mask")
        fn = edits(attended, (ids == self.patch).cpu()).get(condition)
        if fn is None:
            output = policy.model(**inputs, use_cache=True)
        else:
            with edited(policy, len(ids), fn):
                output = policy.model(**inputs, use_cache=True)
        logits = output.logits[0, -1].float()
        prob = logits.softmax(dim=-1)
        out = {
            "yes": float(logits[self.yes].max() - logits[self.no].max()), "action_mass": float(prob[self.actions].sum()),
            "first": processor.tokenizer.decode([int(logits.argmax())]),
        }
        tokens = []
        for _ in range(new_tokens):
            logits = output.logits[:, -1].float()
            logits[:, self.actions] = -torch.inf
            token = logits.argmax(dim=-1)
            if int(token) == self.eos:
                break
            tokens.append(int(token))
            output, attention_mask = policy.model._consume_generation_tokens(token, past_key_values=output.past_key_values, attention_mask=attention_mask)
        out["answer"] = processor.tokenizer.decode(tokens, skip_special_tokens=False).strip()
        return out

    def questions(self, condition: str) -> dict:
        sweep, new = self.sweep, self.sweep.cfg.max_new_tokens
        photo = {int(name.split(".")[0]): name for name in sweep.photos}
        rows = {"yes / no": [], "open": [], "facts": []}
        for number, (inside, outside) in PRESENT.items():
            if number not in photo:
                continue
            for thing, truth in [(t, True) for t in inside] + [(t, False) for t in outside]:
                got = self.ask(f"Is there a {thing} in this image? Answer yes or no.", photo[number], condition, 0)
                rows["yes / no"].append(got | {"photo": number, "thing": thing, "truth": truth, "correct": (got["yes"] > 0) == truth})
        for number, question, pattern in OPEN:
            if number not in photo:
                continue
            got = self.ask(question + " Answer in one or two words.", photo[number], condition, new)
            rows["open"].append(got | {"photo": number, "question": question, "correct": re.search(pattern, got["answer"].lower()) is not None})
        if not condition.startswith("vision from") and "image" not in condition:
            for question, pattern in FACTS:
                got = self.ask(question, None, condition, new)
                rows["facts"].append(got | {"question": question, "correct": re.search(pattern, got["answer"].lower()) is not None})
        return rows


def vision(sweep: Sweep):
    return sweep.policy._backbone().vision_backbone


def cpu_state(module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def later_high(sweep: Sweep, spec: dict) -> list[int]:
    """Image positions that get a high norm after the first block."""
    run = sweep.named(sweep.act(spec))
    high = high_norm(run["encoder"], run["groups"])
    return [i for i, g in enumerate(run["groups"]) if g == "image" and bool(high[:, i].any()) and not bool(high[0, i])]


def swapped(sweep: Sweep, state: dict, other: str) -> dict:
    """Questions, the chunk on every robot frame and the image positions that get a high norm, with the vision
    side of another checkpoint."""
    own = cpu_state(vision(sweep))
    specs = [{"image": image, "style": "step", "text": None} for image in sweep.robot]
    chunks = {(spec["image"], seed): {"own": chunk(sweep, spec, seed), "other seed": chunk(sweep, spec, seed + 100)} for spec in specs for seed in range(sweep.cfg.seeds)}
    high = {spec["image"]: {"own": later_high(sweep, spec)} for spec in specs}
    vision(sweep).load_state_dict(state)
    try:
        rows = Asker(sweep).questions(f"vision from {other}")
        for spec in specs:
            high[spec["image"]]["swapped"] = later_high(sweep, spec)
            for seed in range(sweep.cfg.seeds):
                chunks[(spec["image"], seed)]["swapped"] = chunk(sweep, spec, seed)
    finally:
        vision(sweep).load_state_dict(own)
    return rows | {"chunks": chunks, "high": high}


def summary(name: str, condition: str, rows: dict) -> None:
    parts = []
    for kind in ("yes / no", "open", "facts"):
        if rows.get(kind):
            right = sum(r["correct"] for r in rows[kind])
            parts.append(f"{kind} {right}/{len(rows[kind])} (action tokens {sum(r['action_mass'] for r in rows[kind]) / len(rows[kind]):.2f})")
    print(f"{name} | {condition} | " + " | ".join(parts), flush=True)


def run(cfg: AbilityConfig) -> None:
    init_logging()
    sweep = Sweep(cfg, get_safe_torch_device(try_device=cfg.policy.device))
    result, states = {}, {}
    paths = cfg.checkpoints.split(",")
    for path in paths:
        name = label(path).replace("/", "__")
        start = time.time()
        sweep.load(path)
        asker = Asker(sweep)
        result[name] = {"plain": asker.questions("plain")}
        summary(name, "plain", result[name]["plain"])
        if cfg.edits:
            for condition in edits(torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)):
                result[name][condition] = asker.questions(condition)
                summary(name, condition, result[name][condition])
        states[name] = cpu_state(vision(sweep))
        if name != "base" and "base" in states:
            result[name]["vision from base"] = swapped(sweep, states["base"], "base")
            summary(name, "vision from base", result[name]["vision from base"])
        sweep.free()
        torch.save(result, cfg.out)
        print(name, f"{time.time() - start:.0f} s", flush=True)
    if paths[0] == "base" and len(paths) > 1:
        sweep.load("base")
        for name, state in states.items():
            if name != "base":
                result["base"][f"vision from {name}"] = swapped(sweep, state, name)
                summary("base", f"vision from {name}", result["base"][f"vision from {name}"])
        sweep.free()
        torch.save(result, cfg.out)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: AbilityConfig):
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
