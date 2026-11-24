"""Vision path across depth: how far the language model carries a change of its image tokens.

    uv run python lerobot/adhocs/text_probe/vision_depth.py --config_path=config_rl_validate.yaml

The checkpoint ``model`` is loaded whole, with its own saved policy config (as in ``backbone_cross.py``). On every
frame it runs once as it is (the reference) and once per condition; a condition changes only the image tokens the
language model receives (the output of the vision backbone, one vector per image patch position):

    base      the tokens of the vision path of ``vision`` (computed with its weights, then put in place)
    random    the reference tokens + per token a random direction of the length of (base - reference)
    shuffled  the reference tokens + the differences (base - reference) of other patches (one permutation of
              the patches of the present cameras, the very long tokens left in place)
    next      the camera images of the frame ``next_seconds`` later in the same episode (earlier at its end)
    other     the camera images of the same slot of the next episode (in both the depth stays the frame's own)

State, task, step and every other clause of the prompt are the frame's own in all of them. Per condition and
layer, against the reference at the same position (median over the positions of a kind): the length of the
difference |x - a|, the length |a|, the spread |a - m| (m = the mean of the reference over frames: per layer and
kind of position in the language model, per layer and action position in the expert), the two ratios, the
cosine of (x, a) and of (x - m, a - m). Quantities: the language model's input embedding and every block's
output, the keys and values the action expert reads per layer, the expert's block output and cross-attention
output (first flow step, same noise). And the chunks of ``seeds`` flow seeds (same noise in every condition).
``vision_depth_report.py`` reads the file.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import torch

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import probe_frame_inputs, register_config_choices, sample_episodes_evenly
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from backbone_cross import INTERNAL, parts
from layer_capture import label_tokens
from layer_edit import FACTOR
from run_text_probe import label

CONDITIONS = ["base", "random", "shuffled", "next", "other"]


@dataclass
class DepthConfig(TrainRLServerPipelineConfig):
    model: str = INTERNAL  # loaded whole: its language model and action expert are the ones in every condition
    vision: str = "base"  # the other vision path; "base" = policy.base_path
    root: str = "outputs/data/rebot_2026-10-14-val-v1"
    frames_per_episode: int = 20
    seeds: int = 8
    next_seconds: float = 1.0
    out: str = "outputs/analysis/vision_depth_2026-11-18/depth2.pt"


class image_tokens:
    """``fn(tokens) -> new tokens or None`` on the output of the vision backbone, [image positions, width].
    ``seen`` keeps the output of every call as the model computed it."""

    def __init__(self, policy, fn=None):
        self.vision, self.fn, self.seen = policy._backbone().vision_backbone, fn, []

    def __enter__(self):
        def hook(_module, _inputs, output):
            self.seen.append(output.detach())
            return None if self.fn is None else self.fn(output)

        self.handle = self.vision.register_forward_hook(hook)
        return self

    def __exit__(self, *_):
        self.handle.remove()


class Runner:
    def __init__(self, adapter, device):
        self.adapter, self.policy, self.device = adapter, adapter.policy, device
        self.expert = self.policy._backbone()._require_action_expert()
        self.blocks = self.policy._backbone().transformer.blocks

    def chunk(self, f, obs, seed, fn=None):
        with image_tokens(self.policy, fn) as vision:
            chunk, norm, _ = self.adapter.predict_action_chunk_in_mode(
                obs, f["task"], inference_action_mode="continuous", state=f["state"], subtask=f["subtask"] or None,
                metadata=f["metadata"], generator=torch.Generator(device=self.device).manual_seed(seed),
            )
        assert len(vision.seen) == 1, f"vision backbone ran {len(vision.seen)} times in one forward"
        return chunk.float().cpu(), norm.float().cpu(), vision.seen[0]

    @torch.no_grad()
    def layers(self, f, obs, fn=None) -> dict:
        """One forward under flow seed 0 with everything kept on the device: ``hidden`` [37, S, D] (the input
        embedding, then every language-model block's output), ``keys`` / ``values`` [36, S, 768] (what the action
        expert's cross-attention reads), ``expert`` / ``cross`` [36, T, 768] (first flow step)."""
        policy, expert = self.policy, self.expert
        kept, seen = {"hidden": {}, "expert": {}, "cross": {}}, {}
        pack, prepare = policy._model_inputs, expert.prepare_context

        def first(store, index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                store.setdefault(index, hidden.detach().reshape(-1, *hidden.shape[-2:])[0])

            return hook

        def embedding(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            kept["hidden"].setdefault(-1, hidden.detach().reshape(-1, *hidden.shape[-2:])[0])

        def prepared(**kwargs):
            context = prepare(**kwargs)
            kept.setdefault("context", [(k[0].flatten(1), v[0].flatten(1)) for k, v in context.kv_contexts])
            return context

        handles = [self.blocks[0].register_forward_pre_hook(embedding, with_kwargs=True)]
        handles += [block.register_forward_hook(first(kept["hidden"], i)) for i, block in enumerate(self.blocks)]
        handles += [block.register_forward_hook(first(kept["expert"], i)) for i, block in enumerate(expert.blocks)]
        handles += [block.cross_attn.register_forward_hook(first(kept["cross"], i)) for i, block in enumerate(expert.blocks)]
        policy._model_inputs = lambda batch: seen.update(batch) or pack(batch)
        expert.prepare_context = prepared
        try:
            chunk, norm, tokens = self.chunk(f, obs, 0, fn)
        finally:
            del policy._model_inputs, expert.prepare_context
            for handle in handles:
                handle.remove()
        mask = seen.get("attention_mask")
        return {
            "chunk": chunk, "chunk_norm": norm, "tokens": tokens,
            "ids": seen["input_ids"][0].cpu(), "mask": None if mask is None else mask[0].cpu().bool(),
            "hidden": torch.stack([kept["hidden"][i] for i in sorted(kept["hidden"])]),
            "keys": torch.stack([k for k, _ in kept["context"]]), "values": torch.stack([v for _, v in kept["context"]]),
            "expert": torch.stack([kept["expert"][i] for i in sorted(kept["expert"])]),
            "cross": torch.stack([kept["cross"][i] for i in sorted(kept["cross"])]),
        }


STATS = ["difference", "length", "spread", "difference / length", "difference / spread", "cosine", "cosine, mean removed"]
LANGUAGE, EXPERT = ["hidden", "keys", "values"], ["expert", "cross"]


def kinds(adapter, text, run: dict) -> tuple[list[str], list[str], dict[str, torch.Tensor]]:
    """(tokens, groups, {kind: [S] bool}). Kinds: image (patches of the present cameras), depth (the depth
    tokens), text (every other attended position after the images), last (the last position)."""
    tokens, groups = label_tokens(adapter, text, {"ids": run["ids"], "mask": run["mask"], "encoder": run["hidden"][1:]})
    image = torch.tensor([g == "image" for g in groups])
    depth = torch.tensor([t == "<extra_1>" for t in tokens])
    attended = torch.tensor([g != "masked" for g in groups])
    after = torch.arange(len(groups)) > int(image.nonzero().max())
    last = torch.zeros(len(groups), dtype=torch.bool)
    last[-1] = True
    return tokens, groups, {"image": image, "depth": depth & attended, "text": attended & after & ~depth, "last": last}


class Means:
    """Mean vector of the reference over frames: per layer and kind for the language model (positions over
    ``FACTOR`` x the median length of their kind left out), per layer and action position for the expert."""

    def __init__(self):
        self.total, self.count = {}, {}

    def add(self, run: dict, masks: dict) -> None:
        for quantity in LANGUAGE:
            for kind, mask in masks.items():
                x = run[quantity][:, mask.to(run[quantity].device)].float()  # [L, n, D]
                length = x.norm(dim=-1)
                ordinary = length < FACTOR * length.median(dim=1, keepdim=True).values
                self.total[(quantity, kind)] = self.total.get((quantity, kind), 0) + (x * ordinary.unsqueeze(-1)).sum(dim=1)
                self.count[(quantity, kind)] = self.count.get((quantity, kind), 0) + ordinary.sum(dim=1, keepdim=True)
        for quantity in EXPERT:
            self.total[(quantity, "all")] = self.total.get((quantity, "all"), 0) + run[quantity].float()
            self.count[(quantity, "all")] = self.count.get((quantity, "all"), 0) + 1

    def mean(self, quantity: str, kind: str) -> torch.Tensor:
        """[L, 1, D] for the language model, [L, T, D] for the expert"""
        mean = self.total[(quantity, kind)] / self.count[(quantity, kind)]
        return mean.unsqueeze(1) if quantity in LANGUAGE else mean


def stats(x: torch.Tensor, a: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """``STATS`` per layer, [7, L]: each is the median over the positions [L, n, D] of
    |x - a|, |a|, |a - mean|, |x - a| / |a|, |x - a| / |a - mean|, cos(x, a), cos(x - mean, a - mean)."""
    x, a = x.float(), a.float()
    difference, length, spread = (x - a).norm(dim=-1), a.norm(dim=-1), (a - mean).norm(dim=-1)
    cosine = torch.nn.functional.cosine_similarity
    rows = [difference, length, spread, difference / length.clamp_min(1e-6), difference / spread.clamp_min(1e-6), cosine(x, a, dim=-1), cosine(x - mean, a - mean, dim=-1)]
    return torch.stack([row.median(dim=1).values for row in rows]).cpu()


def against(run: dict, reference: dict, masks: dict, means: Means) -> dict:
    """{quantity: {kind: [7, L]}}"""
    out = {quantity: {kind: stats(run[quantity][:, mask.to(run[quantity].device)], reference[quantity][:, mask.to(run[quantity].device)], means.mean(quantity, kind))
                      for kind, mask in masks.items() if mask.any()} for quantity in LANGUAGE}
    return out | {quantity: {"all": stats(run[quantity], reference[quantity], means.mean(quantity, "all"))} for quantity in EXPERT}


def with_images(own: dict, other: dict) -> dict:
    """``own``'s observation with the camera images of ``other`` (depth stays ``own``'s)."""
    assert own.keys() == other.keys(), (sorted(own.keys() ^ other.keys()))
    return {key: other[key] if key.startswith("observation.images.") else value for key, value in own.items()}


def run(cfg: DepthConfig) -> None:
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
    text = next(s for s in adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)).processor.tokenizer
    runner = Runner(adapter, device)
    own_vision, other_vision = parts(cfg.model, base_path)["vision"], parts(cfg.vision, base_path)["vision"]
    assert set(own_vision) == set(other_vision)

    def frame(index):
        return probe_frame_inputs(dataset, cfg, index, cfg.policy.chunk_size, with_gripper_event_targets=False)

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
    swapped = [runner.chunk(f, f["obs"], 0) for f in frames]
    policy.load_state_dict(own_vision, strict=False)

    means = Means()
    for f in frames:
        reference = runner.layers(f, f["obs"])
        means.add(reference, kinds(adapter, text, reference)[2])
        del reference

    result = {
        "model": label(cfg.model), "vision": label(cfg.vision), "root": cfg.root, "conditions": CONDITIONS, "seeds": cfg.seeds, "stats": STATS,
        "next_seconds": cfg.next_seconds, "fps": cfg.env.fps, "joints": dataset.meta.features["action"]["names"],
        "episode": [f["episode_idx"] for f in frames], "frame": [f["frame_idx"] for f in frames], "index": [f["global_idx"] for f in frames],
        "subtask": [f["subtask"] or "" for f in frames], "state": torch.stack([f["state"].float().cpu() for f in frames]),
        "recorded": torch.stack([f["gt_actions"].float().cpu() for f in frames]), "frames": [],
    }
    start = time.time()
    for at, f in enumerate(frames):
        reference = runner.layers(f, f["obs"])
        a, b = reference["tokens"], swapped[at][2].to(reference["tokens"].dtype)
        delta = b - a
        generator = torch.Generator(device=device).manual_seed(at)
        direction = torch.nn.functional.normalize(torch.randn(a.shape, generator=generator, device=device), dim=-1)
        # shuffled among the patches of the present cameras whose token is not one of the very long ones
        length = a.float().norm(dim=-1)
        seen = reference["mask"][reference["ids"] == adapter._image_patch_token_id()].to(device) if reference["mask"] is not None else torch.ones_like(length, dtype=torch.bool)
        ordinary = (seen & (length < FACTOR * length[seen].median())).nonzero().flatten()
        permutation = torch.arange(len(a), device=device)
        permutation[ordinary] = ordinary[torch.randperm(len(ordinary), generator=generator, device=device)]
        other = frames[(at + cfg.frames_per_episode) % len(frames)]
        conditions = {
            "base": (f["obs"], lambda _tokens: b),
            "random": (f["obs"], lambda _tokens: a + (direction * delta.float().norm(dim=-1, keepdim=True)).to(a.dtype)),
            "shuffled": (f["obs"], lambda _tokens: a + delta[permutation]),
            "next": (with_images(f["obs"], later[at]["obs"]), None),
            "other": (with_images(f["obs"], other["obs"]), None),
        }
        _, groups, masks = kinds(adapter, text, reference)
        entry = {
            "other_episode": other["episode_idx"], "next_frame": later[at]["frame_idx"], "ordinary": len(ordinary),
            "weights_chunk": swapped[at][0],  # seed 0 with the other vision path's weights in place: must equal the "base" condition
            "chunks": {"reference": [reference["chunk"]]}, "chunks_norm": {"reference": [reference["chunk_norm"]]}, "layers": {},
        }
        for name, (obs, fn) in conditions.items():
            run_ = runner.layers(f, obs, fn)
            assert run_["hidden"].shape == reference["hidden"].shape, (name, run_["hidden"].shape, reference["hidden"].shape)
            entry["layers"][name] = against(run_, reference, masks, means)
            entry["chunks"][name], entry["chunks_norm"][name] = [run_["chunk"]], [run_["chunk_norm"]]
            del run_
        for seed in range(1, cfg.seeds):
            for name, (obs, fn) in {"reference": (f["obs"], None), **conditions}.items():
                chunk, norm, _ = runner.chunk(f, obs, seed, fn)
                entry["chunks"][name].append(chunk)
                entry["chunks_norm"][name].append(norm)
        entry["chunks"] = {name: torch.stack(chunks) for name, chunks in entry["chunks"].items()}  # [seeds, steps, joints], degrees
        entry["chunks_norm"] = {name: torch.stack(chunks) for name, chunks in entry["chunks_norm"].items()}
        result["frames"].append(entry)
        del reference
        if at == 0:
            print("groups", {g: groups.count(g) for g in dict.fromkeys(groups)}, "| kinds", {k: int(m.sum()) for k, m in masks.items()}, "| image tokens", tuple(a.shape), flush=True)
            print("base tokens put in place against base weights, largest chunk difference (deg)", float((entry["chunks"]["base"][0] - entry["weights_chunk"]).abs().max()), flush=True)
        print(f"frame {at + 1} of {len(frames)}, {time.time() - start:.0f} s", flush=True)
        if at % 10 == 9 or at == len(frames) - 1:
            torch.save(result, cfg.out)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: DepthConfig):
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
