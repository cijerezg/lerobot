"""Public question benchmarks on the original VLM, the released MolmoAct2 and our checkpoints, side by side.

    uv run python lerobot/adhocs/vlm_knowledge/run_benchmarks.py --config_path=config_rl_validate.yaml \
        --models=er,base,internal=<pretrained_model dir>,tube700=<pretrained_model dir> \
        --benchmarks=mmlu_redux,cv_bench,embspatial,vqa_v2 --grid=true

Question files come from ``prepare_benchmarks.py``. Every question is one plain chat turn (no robot clauses, no
action output token), the action expert is never called. ``er`` is Molmo2-ER (the VLM MolmoAct2 was trained
from): its vision encoder, language model and output rows loaded into the released MolmoAct2, whose extra token
rows (state, action, ...) Molmo2-ER does not have and can therefore never emit. ``base`` is ``policy.base_path``.

Each answer is read three ways from the same forward pass:
  free     greedy decoding over the whole vocabulary
  masked   greedy decoding with the action tokens forbidden
  letters  multiple choice only: the most likely of the option letters at the first position
``action_mass`` is the probability on the action tokens at the first answer position.

``--grid=true`` also runs every image benchmark with the vision encoder (tower, pooling, projector) of every
other model under each language model. One file per (language model, vision encoder, benchmark) under ``out``,
one line per question; a file that exists is skipped, so an interrupted run continues. ``summary.py`` reads them.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import register_config_choices
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

ER = "outputs/models/Molmo2-ER"
ER_ROWS = 151669  # token rows Molmo2-ER shares with MolmoAct2; MolmoAct2's later rows are its own new tokens
LETTERS = "ABCDEFGH"
MC_STYLES = {
    "letter": "{question}\n{options}\nAnswer with the option's letter from the given choices directly.",
    "molmo": "{question}\nOnly return the correct answer option.\n{options}",
}
NUMBERS = dict(zip("zero one two three four five six seven eight nine ten".split(), map(str, range(11))))


@dataclass
class BenchmarkConfig(TrainRLServerPipelineConfig):
    models: str = "er,base"  # comma list of er | base | <name>=<pretrained_model dir>
    benchmarks: str = "mmlu_redux,cv_bench,embspatial,vqa_v2"
    data: str = "outputs/data/vlm_benchmarks"
    out: str = "outputs/analysis/vlm_knowledge_2026-11-17/answers"
    grid: bool = False
    limit: int = 0  # > 0: that many questions per benchmark, the same seeded draw for every model
    crop_mode: str = "resize"  # resize = one crop, as MolmoAct2 and our runs see images; overlap-and-resize-c2 = Molmo2-ER's own
    mc_style: str = "letter"
    max_new_tokens: int = 10


def tensors(path: str, part: str) -> dict:
    """Tensors of one part ("vision_backbone." / "transformer." / "lm_head.") of a checkpoint folder, names
    relative to the part. Released folders hold shards, ours one model.safetensors with a leading "model."."""
    out = {}
    for file in sorted(glob.glob(os.path.join(path, "model*.safetensors"))):
        with safe_open(file, "pt") as f:
            for name in f.keys():
                if part in name and not name.split(part)[0].replace("model.", ""):
                    out[name.split(part, 1)[1]] = f.get_tensor(name)
    return out


def normalize(answer: str) -> str:
    """VQA v2 answer normalization (lower case, no punctuation, no articles, number words as digits)."""
    words = re.sub(r"[^\w\s']", " ", answer.lower().replace("\n", " ")).split()
    return " ".join(NUMBERS.get(w, w) for w in words if w not in ("a", "an", "the"))


class Model:
    def __init__(self, cfg: BenchmarkConfig, device, dataset, name: str, path: str | None):
        cfg.policy.pretrained_path = path
        self.adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
        self.adapter._set_probe_cuda_graph_enabled(False)
        self.processor = next(s for s in self.adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)).processor
        self.processor.image_processor.crop_mode = cfg.crop_mode
        self.policy, self.cfg, self.device, self.name = self.adapter.policy, cfg, device, name
        config, tokenizer = self.policy.model.config, self.processor.tokenizer
        self.actions = slice(int(config.action_output_token_id), int(config.action_token_start_id) + int(config.num_action_tokens))
        self.eos = int(config.eos_token_id)
        self.absent = None
        self.letters = [sorted({tokenizer.encode(form, add_special_tokens=False)[0] for form in (letter, " " + letter)}) for letter in LETTERS]
        if name == "er":
            self.load_language(ER)
            self.load_vision(ER)
            self.absent = slice(ER_ROWS, None)

    def load_vision(self, path: str) -> None:
        self.policy._backbone().vision_backbone.load_state_dict(tensors(path, "vision_backbone."))

    def load_language(self, path: str) -> None:
        transformer, head = self.policy._backbone().transformer, self.policy.model.lm_head
        state = tensors(path, "transformer.")
        rows = state.pop("wte.embedding")
        missing = transformer.load_state_dict(state, strict=False)
        # rotary_emb.inv_freq is a computed buffer, not in the files
        assert set(missing.missing_keys) == {"wte.embedding", "rotary_emb.inv_freq"} and not missing.unexpected_keys, missing
        with torch.no_grad():
            transformer.wte.embedding[:ER_ROWS] = rows[:ER_ROWS].to(transformer.wte.embedding)
            head.weight[:ER_ROWS] = tensors(path, "lm_head.")["weight"][:ER_ROWS].to(head.weight)

    def logits(self, output) -> torch.Tensor:
        logits = output.logits[:, -1].float()
        if self.absent is not None:
            logits[:, self.absent] = -torch.inf
        return logits

    @torch.no_grad()
    def ask(self, text: str, picture, new_tokens: int) -> dict:
        """One chat turn. ``new_tokens`` 0: first position only."""
        policy, processor = self.policy, self.processor
        text = ("<|image|>" if picture is not None else "") + f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=text, images=picture, return_tensors="pt", padding=True)
        inputs = policy._drop_trivial_attention_mask(policy._model_inputs({key: value.to(self.device) for key, value in inputs.items()}))
        start = policy.model(**inputs, use_cache=True)
        first = self.logits(start)
        out = {"action_mass": float(first.softmax(dim=-1)[0, self.actions].sum()), "first": first[0]}
        for mode in ("masked", "free"):
            output, attention_mask, tokens, switched = start, inputs.get("attention_mask"), [], False
            for _ in range(new_tokens):
                logits = self.logits(output)
                if mode == "masked":
                    switched |= self.actions.start <= int(logits.argmax()) < self.actions.stop
                    logits[:, self.actions] = -torch.inf
                token = logits.argmax(dim=-1)
                if int(token) == self.eos:
                    break
                tokens.append(int(token))
                output, attention_mask = policy.model._consume_generation_tokens(token, past_key_values=output.past_key_values, attention_mask=attention_mask)
            out[mode] = processor.tokenizer.decode(tokens, skip_special_tokens=False).strip()
            out[mode + "_action_tokens"] = sum(self.actions.start <= t < self.actions.stop for t in tokens)
            if mode == "masked" and not switched:  # no action token would have been picked: free decoding gives the same answer
                out |= {"free": out["masked"], "free_action_tokens": 0}
                break
        return out

    def question(self, row: dict, folder: str) -> dict:
        picture = np.asarray(Image.open(os.path.join(folder, row["image"])).convert("RGB")) if row["image"] else None
        kept = {"id": row["id"], "category": row["category"]}
        if row["options"] is None:
            got = self.ask("vqa2: " + row["question"], picture, self.cfg.max_new_tokens)
            humans = [normalize(a) for a in row["answer"]]
            score = lambda text: min(1.0, humans.count(normalize(text)) / 3)  # noqa: E731
            return kept | {
                "free": score(got["free"]), "masked": score(got["masked"]), "action_mass": got["action_mass"],
                "free_answer": got["free"], "masked_answer": got["masked"], "free_in_action_tokens": got["free_action_tokens"] > 0,
            }
        options = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, row["options"]))
        got = self.ask(MC_STYLES[self.cfg.mc_style].format(question=row["question"], options=options), picture, 0)
        first, right = got["first"], self.letters[row["answer"]]
        masked = first.clone()
        masked[self.actions] = -torch.inf
        letters = torch.stack([first[ids].max() for ids in self.letters[: len(row["options"])]])
        return kept | {
            "free": int(first.argmax()) in right, "masked": int(masked.argmax()) in right, "letters": int(letters.argmax()) == row["answer"],
            "action_mass": got["action_mass"], "free_answer": self.processor.tokenizer.decode([int(first.argmax())]),
            "masked_answer": self.processor.tokenizer.decode([int(masked.argmax())]),
            "free_in_action_tokens": self.actions.start <= int(first.argmax()) < self.actions.stop,
        }


def questions(cfg: BenchmarkConfig, benchmark: str) -> list[dict]:
    with open(os.path.join(cfg.data, benchmark, "questions.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    if cfg.limit:
        rows = random.Random(0).sample(rows, min(cfg.limit, len(rows)))
    return rows


def run(cfg: BenchmarkConfig) -> None:
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)
    paths = {"er": ER, "base": cfg.policy.base_path} | dict(m.split("=", 1) for m in cfg.models.split(",") if "=" in m)
    names = [m.split("=", 1)[0] for m in cfg.models.split(",")]
    for name in names:
        model = Model(cfg, device, dataset, name, None if name in ("er", "base") else paths[name])
        for vision in [name] + [other for other in names if other != name and cfg.grid]:
            loaded = False
            for benchmark in cfg.benchmarks.split(","):
                rows = questions(cfg, benchmark)
                file = os.path.join(cfg.out, f"{name}__vision-{vision}", benchmark + ".jsonl")
                if os.path.exists(file) or (vision != name and not rows[0]["image"]):
                    continue
                if not loaded:
                    model.load_vision(paths[vision])
                    loaded = True
                start, done = time.time(), []
                for row in rows:
                    done.append(model.question(row, os.path.join(cfg.data, benchmark)))
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in done)
                mean = lambda key: sum(r.get(key, float("nan")) for r in done) / len(done)  # noqa: E731
                print(f"{name} | vision {vision} | {benchmark} | n {len(done)} | free {mean('free'):.3f} | masked {mean('masked'):.3f} | letters {mean('letters'):.3f} | "
                      f"in action tokens {mean('free_in_action_tokens'):.3f} | action mass {mean('action_mass'):.3f} | {time.time() - start:.0f} s", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


@parser.wrap()
def cli(cfg: BenchmarkConfig):
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
