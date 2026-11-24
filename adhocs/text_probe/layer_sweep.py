"""Layer sweep: hidden states of every block for a fixed list of runs, one checkpoint at a time, saved to disk.

    uv run python lerobot/adhocs/text_probe/layer_sweep.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,<pretrained_model dir>,... --suite=full --store=outputs/cache/layer_cosine_2026-11-16

A run is one forward of one checkpoint on one prompt (chat or action prompt, as on the chat page's Layers tab).
For every run ``<store>/<checkpoint label>/<run name>.pt`` holds the ids, the kind of every position, the output
of all language-model blocks at every position and, for action prompts, what the action expert does with it:
its projected key / value context per layer, its blocks' output and cross-attention output (first flow step),
the cross-attention mass per prompt position, and the predicted chunk. ``layer_analysis.py`` reads the store.

Suites: ``verify`` (shapes and hooks on a few runs, nothing saved), ``full`` (every run of ``chat_runs`` and
``action_runs``, plus the chunks under other flow seeds, the chunks with one kind of token hidden from the
action expert, and the action tokens' own hidden states), ``series`` (a small subset, for the saved steps of
one training run).
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import probe_frame_inputs, register_config_choices
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.gripper_torque import GRIPPER_TORQUE_LEVEL_KEY, gripper_torque_levels
from lerobot.utils.motor_velocity import VELOCITY_KEY
from lerobot.utils.tip_target import TIP_TARGET_KEY
from lerobot.utils.utils import init_logging
from chat import ChatConfig
from layer_capture import capture, find, label_tokens
from run_text_probe import IMAGE_KEY, ROBOT_QUESTION, label

TUBE = ["robot_200", "robot_600", "robot_1200", "robot_2000"]
CLUTTER = ["clutter_1180", "clutter_2180", "clutter_3215", "clutter_4310", "clutter_6040"]
CHAT_PROMPTS = {"describe": "Describe this image.", "objects": "What objects are on the table?", "next": "What should the robot do next?"}
TEXT_PROMPTS = {
    "hi": "Hi, how are you?", "france": "What is the capital of France?", "times": "What is 17 times 23?",
    "tubeplan": "How would a robot arm put a small tube on a peg?", "robotq": ROBOT_QUESTION, "next": "What should the robot do next?",
}
INSTRUCTIONS = {
    "right": "move right", "left": "move left", "cup": "grasp the cup", "bottle": "grasp the purple bottle",
    "tubeleft": "grasp tube on the left", "tuberight": "grasp tube on the right",
}
HIDDEN_KINDS = ["image", "task", "subtask", "state", "velocity", "gripper_torque", "tip_target", "question", "tail"]


def chat_runs(photos: list[str], suite: str) -> list[dict]:
    if suite == "series":
        images, prompts, texts = ["robot_600", "clutter_3215", photos[0]], ["describe", "next"], ["france", "robotq"]
    else:
        images, prompts, texts = TUBE + CLUTTER + photos, list(CHAT_PROMPTS), list(TEXT_PROMPTS)
    runs = [{"name": f"chat.{image}.{p}", "kind": "chat", "image": image, "text": CHAT_PROMPTS[p]} for image in images for p in prompts]
    return runs + [{"name": f"chat.none.{t}", "kind": "chat", "image": None, "text": TEXT_PROMPTS[t]} for t in texts]


def action_runs(photos: list[str], suite: str) -> list[dict]:
    """style step: the text is the step (None = the frame's own step); task / original: the text is the task."""
    def one(image, style, key):
        return {"name": f"act.{style}.{image}.{key}", "kind": "action", "image": image, "style": style, "text": INSTRUCTIONS.get(key)}

    if suite == "series":
        return [one(image, "step", key) for image in ["robot_600", "robot_1200", "clutter_3215"] for key in ["own", "right", "left", "tubeleft", "tuberight"]]
    runs = [one(image, "step", key) for image in TUBE + CLUTTER for key in ["own", *INSTRUCTIONS]]
    runs += [one(image, "step", key) for image in photos[:3] + [None] for key in ["right", "left", "cup", "bottle"]]
    for style in ["task", "original"]:
        runs += [one(image, style, key) for image in ["robot_600", "robot_1200", "clutter_3215", "clutter_6040"] for key in ["right", "left", "cup", "bottle"]]
    return runs


@dataclass
class SweepConfig(ChatConfig):
    store: str = "outputs/cache/layer_cosine_2026-11-16"
    suite: str = "full"  # verify | full | series
    photos: int = 6  # first files of image_dir used as photos
    seeds: int = 3  # full: flow seeds per step-style action run (seed 0 is the captured one)


class Sweep:
    def __init__(self, cfg: SweepConfig, device):
        self.cfg, self.device = cfg, device
        self.dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)
        clutter = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.clutter_root)
        self.robot = {f"robot_{frame}": (self.dataset, int(frame)) for frame in cfg.frames.split(",")}
        self.robot |= {f"clutter_{frame}": (clutter, int(frame)) for frame in cfg.clutter_frames.split(",")}
        self.images = {name: (source[index][IMAGE_KEY].permute(1, 2, 0) * 255).round().byte().numpy() for name, (source, index) in self.robot.items()}
        self.photos = sorted(os.listdir(cfg.image_dir))[: cfg.photos]
        for name in self.photos:
            self.images[name] = np.asarray(Image.open(os.path.join(cfg.image_dir, name)).convert("RGB"))
        self.frames = {}

    def load(self, path: str):
        self.cfg.policy.pretrained_path = None if path == "base" else path
        self.adapter = ProbablePolicy.for_config(self.cfg, self.device, dataset=self.dataset)
        self.adapter._set_probe_cuda_graph_enabled(False)
        self.processor = next(s for s in self.adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)).processor
        self.policy = self.adapter.policy

    def free(self):
        del self.adapter, self.processor, self.policy
        gc.collect()
        torch.cuda.empty_cache()

    def action_inputs(self, name, style):
        """Same inputs as chat.py's action_inputs: frame, photo as the top camera, or no camera at all."""
        cfg, torque = self.cfg, self.cfg.policy.gripper_torque
        if name not in self.frames:
            source, index = self.robot.get(name, self.robot[cfg.body_frame])
            frame = probe_frame_inputs(source, cfg, index, cfg.policy.chunk_size, with_gripper_event_targets=False)
            frame["obs"].setdefault(f"probe_complementary.{VELOCITY_KEY}", torch.zeros(1, 7))
            frame["obs"].setdefault(
                f"probe_complementary.{GRIPPER_TORQUE_LEVEL_KEY}",
                torch.as_tensor(gripper_torque_levels(np.zeros(1), torque.levels, torque.low, torque.high)),
            )
            if name is None:
                for key in [key for key in frame["obs"] if key.startswith("observation.images.")]:
                    frame["obs"][key] = torch.zeros_like(frame["obs"][key])
                    frame["obs"][f"probe_complementary.camera_is_present.{key}"] = torch.tensor([False])
                frame["gt_actions"] = None
            elif name not in self.robot:
                top = frame["obs"]["observation.images.external_0"]
                photo = Image.fromarray(self.images[name])
                photo.thumbnail((top.shape[-1], top.shape[-2]))
                canvas = Image.new("RGB", (top.shape[-1], top.shape[-2]))
                canvas.paste(photo, ((canvas.width - photo.width) // 2, (canvas.height - photo.height) // 2))
                frame["obs"]["observation.images.external_0"] = torch.from_numpy(np.asarray(canvas)).permute(2, 0, 1)[None].float() / 255
                frame["gt_actions"] = None
            self.frames[name] = frame
        frame = self.frames[name]
        obs, metadata = frame["obs"], frame["metadata"]
        if style == "original":
            extra = [f"probe_complementary.{key}" for key in (VELOCITY_KEY, GRIPPER_TORQUE_LEVEL_KEY, TIP_TARGET_KEY)]
            obs, metadata = {key: value for key, value in obs.items() if key not in extra}, None
        return frame, obs, metadata

    def words(self, spec):
        """(task, step) of an action run."""
        frame, _, _ = self.action_inputs(spec["image"], spec["style"])
        if spec["style"] != "step":
            return spec["text"], None
        return frame["task"], spec["text"] if spec["text"] is not None else (frame["subtask"] or None)

    def chat(self, spec) -> dict:
        policy, processor, device = self.policy, self.processor, self.device

        def forward():
            picture = self.images[spec["image"]] if spec["image"] is not None else None
            text = ("<|image|>" if picture is not None else "") + f"<|im_start|>user\n{spec['text']}<|im_end|>\n<|im_start|>assistant\n"
            inputs = processor(text=text, images=picture, return_tensors="pt", padding=True)
            inputs = policy._model_inputs({key: value.to(device) for key, value in inputs.items()})
            forward.logits = policy.model(**policy._drop_trivial_attention_mask(inputs), use_cache=True).logits[0, -1].float().cpu()
            mask = inputs.get("attention_mask")
            return inputs["input_ids"][0], None if mask is None else mask[0]

        run = capture(self.adapter, forward)
        run["logits"] = forward.logits
        return run

    def act(self, spec, seed: int = 0, hide: list[int] | None = None, layers: bool = True, mode: str = "continuous") -> dict:
        """One action forward. ``hide``: prompt positions removed from the action expert's cross-attention mask
        (the language model still sees them). ``layers`` false: only the chunk is returned."""
        policy, adapter = self.policy, self.adapter
        expert = policy._backbone()._require_action_expert()
        frame, obs, metadata = self.action_inputs(spec["image"], spec["style"])
        task, subtask = self.words(spec)
        seen, kept = {}, {"mass": {}, "calls": {}, "cross": {}}
        pack, prepare = policy._model_inputs, expert.prepare_context

        def prepared(**kwargs):
            mask = kwargs.get("encoder_attention_mask")
            if hide:
                mask = torch.ones_like(seen["input_ids"], dtype=torch.bool) if mask is None else mask.clone()
                mask[:, hide] = False
                kwargs["encoder_attention_mask"] = mask
            context = prepare(**kwargs)
            if "context" not in kept:
                kept["context"] = [(k[0].flatten(1).cpu(), v[0].flatten(1).cpu()) for k, v in context.kv_contexts]
                kept["expert_mask"] = None if mask is None else mask[0].bool().cpu()
            return context

        def watch(index, attention):
            def attend(q, k, v, *, attn_mask=None):
                # q [B, T, H, d], k [B, S, H, d]: the weights scaled_dot_product_attention uses, mean over the action positions
                scores = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2) / q.shape[-1] ** 0.5
                if attn_mask is not None:
                    scores = scores + attn_mask.float()
                mass = scores.softmax(dim=-1)[0].mean(dim=1).cpu()  # [H, S]
                kept["calls"][index] = kept["calls"].get(index, 0) + 1
                kept["mass"].setdefault(index, []).append(mass)
                return attention(q, k, v, attn_mask=attn_mask)

            return attend

        def cross(index):
            def hook(_module, _inputs, output):
                kept["cross"].setdefault(index, output.detach()[0].cpu())

            return hook

        handles = []
        if layers:
            for index, block in enumerate(expert.blocks):
                block.cross_attn._attention = watch(index, block.cross_attn._attention)
                handles.append(block.cross_attn.register_forward_hook(cross(index)))
        policy._model_inputs = lambda batch: seen.update(batch) or pack(batch)
        expert.prepare_context = prepared

        def forward():
            forward.chunk, forward.norm, _ = adapter.predict_action_chunk_in_mode(
                obs, task, inference_action_mode=mode, state=frame["state"], subtask=subtask,
                metadata=metadata, generator=torch.Generator(device=self.device).manual_seed(seed),
            )
            mask = seen.get("attention_mask")
            return seen["input_ids"][0], None if mask is None else mask[0]

        try:
            if layers:
                run = capture(adapter, forward)
            else:
                forward()
                run = {}
        finally:
            del policy._model_inputs, expert.prepare_context
            for handle in handles:
                handle.remove()
            if layers:
                for block in expert.blocks:
                    del block.cross_attn._attention
        run |= {"chunk": forward.chunk, "chunk_norm": forward.norm, "task": task, "subtask": subtask}
        if layers:
            layers_ = sorted(kept["mass"])
            run["context_k"] = torch.stack([k for k, _ in kept["context"]])  # [L, S, 768] what cross-attention keys on
            run["context_v"] = torch.stack([v for _, v in kept["context"]])
            run["expert_mask"] = kept["expert_mask"]
            run["cross"] = torch.stack([kept["cross"][i] for i in layers_])  # [L, T, 768] cross-attention output, first flow step
            run["mass_first"] = torch.stack([kept["mass"][i][0] for i in layers_])  # [L, H, S]
            run["mass_mean"] = torch.stack([torch.stack(kept["mass"][i]).mean(dim=0) for i in layers_])
            run["flow_steps"] = kept["calls"][layers_[0]]
        return run

    @torch.no_grad()
    def lens(self, run) -> dict:
        """Logit lens at the last position: every block's output through the final norm and lm_head."""
        hf = self.policy._hf_model()
        config = self.policy.model.config
        hidden = run["encoder"][:, -1].to(self.device)
        head = hf.lm_head
        logits = head(hf.model.transformer.ln_f(hidden.to(head.weight.dtype))).float()
        prob = logits.softmax(dim=-1)
        actions = slice(int(config.action_output_token_id), int(config.action_token_start_id) + int(config.num_action_tokens))
        final = prob[-1].clamp_min(1e-12).log()
        return {
            "top": prob.topk(5).indices.cpu(),
            "top_prob": prob.topk(5).values.cpu(),
            "action_mass": prob[:, actions].sum(dim=-1).cpu(),
            "entropy": -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1).cpu(),
            "kl_to_final": (prob[-1] * (final - prob.clamp_min(1e-12).log())).sum(dim=-1).cpu(),
            "final_logits": logits[-1].cpu(),
        }

    @torch.no_grad()
    def stream(self, spec) -> dict:
        """Action tokens' own hidden states: the action prompt decoded through lm_head, every block's output kept
        at the prompt positions and at every token fed back in. ``fed`` are the ids of those extra positions."""
        policy, adapter = self.policy, self.adapter
        backbone = policy._backbone()
        frame, obs, metadata = self.action_inputs(spec["image"], spec["style"])
        task, subtask = self.words(spec)
        store, seen, generated = {}, {}, {}

        def keep(index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                store.setdefault(index, []).append(hidden.detach().reshape(-1, *hidden.shape[-2:])[0].cpu())

            return hook

        handles = [block.register_forward_hook(keep(i)) for i, block in enumerate(backbone.transformer.blocks)]
        pack, decode = policy._model_inputs, policy._decode_discrete_action_chunk
        policy._model_inputs = lambda batch: seen.update(batch) or pack(batch)
        policy._decode_discrete_action_chunk = lambda ids, **kwargs: generated.update(ids=ids) or decode(ids, **kwargs)
        error = ""
        try:
            adapter.predict_action_chunk_in_mode(obs, task, inference_action_mode="discrete", state=frame["state"], subtask=subtask, metadata=metadata)
        except (RuntimeError, ValueError) as failure:
            error = str(failure)
        finally:
            del policy._model_inputs, policy._decode_discrete_action_chunk
            for handle in handles:
                handle.remove()
        mask = seen.get("attention_mask")
        return {
            "ids": seen["input_ids"][0].cpu(),
            "mask": None if mask is None else mask[0].cpu().bool(),
            "generated": generated["ids"].reshape(-1).cpu() if "ids" in generated else None,
            "calls": [tuple(piece.shape) for piece in store[0]],
            "encoder": torch.stack([torch.cat(store[i]) for i in sorted(store)]),
            "error": error,
        }

    def named(self, run):
        tokenizer = self.processor.tokenizer
        run["tokens"], run["groups"] = label_tokens(self.adapter, tokenizer, run)
        # the tip target clause has no marker in the adapter's clause list: it would count as gripper torque
        start = find(run["ids"].tolist(), tokenizer.encode(" The target is at", add_special_tokens=False))
        if start is not None:
            for at in range(start, len(run["groups"])):
                if run["groups"][at] != "gripper_torque":
                    break
                run["groups"][at] = "tip_target"
        return run


def verify(sweep: Sweep) -> None:
    tokenizer = sweep.processor.tokenizer
    spec = {"name": "v", "kind": "chat", "image": "robot_600", "text": "Describe this image."}
    a, b = sweep.named(sweep.chat(spec)), sweep.named(sweep.chat(spec))
    print("chat ids", tuple(a["ids"].shape), "encoder", tuple(a["encoder"].shape), a["encoder"].dtype, "expert", a["expert"], "mask", None if a["mask"] is None else int(a["mask"].sum()))
    assert a["encoder"].shape[1] == len(a["ids"]), "positions != prompt length"
    print("chat same run twice, largest difference", float((a["encoder"].float() - b["encoder"].float()).abs().max()))
    lens = sweep.lens(a)
    print("lens: last block through ln_f + lm_head against the model's logits, largest difference", float((lens["final_logits"] - a["logits"]).abs().max()),
          "| argmax", tokenizer.decode([int(lens["final_logits"].argmax())]), "/", tokenizer.decode([int(a["logits"].argmax())]))
    print("chat groups", {g: a["groups"].count(g) for g in dict.fromkeys(a["groups"])})
    for style in ["step", "task", "original"]:
        spec = {"name": "v", "kind": "action", "image": "robot_600", "style": style, "text": "move right"}
        run = sweep.named(sweep.act(spec))
        again = sweep.act(spec)
        print(f"\naction {style}: ids", tuple(run["ids"].shape), "encoder", tuple(run["encoder"].shape), run["encoder"].dtype, "expert", tuple(run["expert"].shape), run["expert"].dtype)
        assert run["encoder"].shape[1] == len(run["ids"]), "positions != prompt length"
        assert run["expert"].shape[0] == len(sweep.policy._backbone()._require_action_expert().blocks), "an action expert hook did not fire"
        print("  context", tuple(run["context_k"].shape), "cross", tuple(run["cross"].shape), "mass", tuple(run["mass_first"].shape), "flow steps", run["flow_steps"],
              "mass sums to", float(run["mass_first"][0, 0].sum()), "expert mask", None if run["expert_mask"] is None else int(run["expert_mask"].sum()))
        print("  groups", {g: run["groups"].count(g) for g in dict.fromkeys(run["groups"])})
        print("  text after the images:", repr(tokenizer.decode(run["ids"][[i for i, g in enumerate(run["groups"]) if g not in ("image", "masked")]].tolist())))
        print("  chunk", tuple(run["chunk"].shape), "same seed twice", float((run["chunk"] - again["chunk"]).abs().max()), "encoder twice", float((run["encoder"].float() - again["encoder"].float()).abs().max()),
              "| other seed", float((run["chunk"] - sweep.act(spec, seed=1, layers=False)["chunk"]).abs().max()))
        hidden = [i for i, g in enumerate(run["groups"]) if g == "image"]
        print("  chunk with the image hidden from the action expert, largest change", float((run["chunk"] - sweep.act(spec, hide=hidden, layers=False)["chunk"]).abs().max()))
    spec = {"name": "v", "kind": "action", "image": "robot_600", "style": "step", "text": None}
    stream = sweep.stream(spec)
    print("\nstream: prompt", len(stream["ids"]), "encoder", tuple(stream["encoder"].shape), "calls", stream["calls"][:3], "...", len(stream["calls"]), "error", stream["error"] or "none")
    if stream["generated"] is not None:
        print("  generated", len(stream["generated"]), repr(tokenizer.decode(stream["generated"][:12].tolist())))


def run(cfg: SweepConfig) -> None:
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    sweep = Sweep(cfg, device)
    for path in cfg.checkpoints.split(","):
        name = label(path).replace("/", "__")
        start = time.time()
        sweep.load(path)
        if cfg.suite == "verify":
            verify(sweep)
            sweep.free()
            continue
        folder = os.path.join(cfg.store, name)
        os.makedirs(folder, exist_ok=True)
        chats, actions = chat_runs(sweep.photos, cfg.suite), action_runs(sweep.photos, cfg.suite)
        for spec in chats:
            run = sweep.named(sweep.chat(spec))
            run["lens"] = sweep.lens(run)
            torch.save(run | {"spec": spec}, os.path.join(folder, spec["name"] + ".pt"))
        extras = {}
        for spec in actions:
            run = sweep.named(sweep.act(spec))
            run["lens"] = sweep.lens(run)
            torch.save(run | {"spec": spec}, os.path.join(folder, spec["name"] + ".pt"))
            if spec["style"] != "step":
                continue
            extra = {"seeds": {0: run["chunk"]} | {seed: sweep.act(spec, seed=seed, layers=False)["chunk"] for seed in range(1, cfg.seeds)}}
            if spec["image"] in sweep.robot and spec["name"].rsplit(".", 1)[1] in ("own", "right", "tubeleft"):
                kinds = {kind: [i for i, g in enumerate(run["groups"]) if g == kind] for kind in HIDDEN_KINDS}
                question_end = max(kinds["question"])
                kinds["tail"] = [i for i in range(question_end + 1, len(run["groups"]))]
                kinds["all text"] = [i for i, g in enumerate(run["groups"]) if g not in ("image", "masked")]
                extra["hidden"] = {kind: sweep.act(spec, hide=positions, layers=False)["chunk"] for kind, positions in kinds.items() if positions}
                extra["hidden_counts"] = {kind: len(positions) for kind, positions in kinds.items()}
            extras[spec["name"]] = extra
        torch.save(extras, os.path.join(folder, "chunks.pt"))
        streams = [s for s in actions if s["style"] == "step" and s["name"].endswith(".own")]
        for spec in streams:
            torch.save(sweep.stream(spec) | {"spec": spec}, os.path.join(folder, spec["name"].replace("act.", "stream.") + ".pt"))
        sweep.free()
        logging.info(f"[layer_sweep] {name}: {len(chats)} chat + {len(actions)} action runs in {time.time() - start:.0f} s -> {folder}")


@parser.wrap()
def cli(cfg: SweepConfig):
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
