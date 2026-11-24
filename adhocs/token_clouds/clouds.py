"""Token clouds: move the tokens of one layer by hand and watch the later layers follow, over an episode or
over the checkpoints of a run.

    uv run python lerobot/adhocs/token_clouds/clouds.py --config_path=config_rl_validate.yaml

then open http://localhost:8773. The checkpoint ``model`` is loaded whole and stays on the GPU. A scene is a
list of steps with one or more items each, an item being a frame under a checkpoint: a few frames side by side
(one step), or one episode (one frame per step, ``RATE`` per second of footage, at most ``episode_frames``). Its points are the attended prompt
positions of the chosen kinds (``layers.KINDS``); the same image patch, or the same token in order within its
kind, is the same point at every step. At hidden index ``l`` (0 = the input of the first language-model block,
``l`` = the output of block ``l``) the point of a position with vector ``x`` is

    c = (x - mu_l) P_l^T / s_l

with ``mu_l`` the mean and ``P_l`` [3, D] the first three principal directions of the vectors at that index
(over the frames of the scene, or ``FIT`` steps spread over the episode), turned by the 3 x 3 rotation that
brings the cloud closest to the cloud of index ``l - 1``, and ``s_l`` the 95th percentile of the distance from
the mean. Positions longer than ``FAR`` x the median length are left out of the fit and marked far (the page
hides them unless asked). The expert's block outputs (action positions, first flow step) get the same.

A move is a new place ``c'`` for some points of the edit index ``l`` at one step. The input of block ``l``
becomes

    x' = x + s_l (c' - c) P_l

at those positions: the change stays inside the three shown directions. The forward runs on from there under
flow seed 0 and the page gets the points of every hidden index and per hidden index
``mean |x' - x| / mean |x - mu|`` over the points that are not far; a full move also gives the expert's points
and the tip path of the chunk (cm, filtered as deployed), the other stops before the action expert (about a
third of the time).

The checkpoints ``others`` (training steps of the run of ``model``, and "base") stay on the GPU next to the
weights of ``model``: one policy, whose parameters are pointed at the tensors of the checkpoint a forward is for.
A parameter a checkpoint's file does not have (base has no tip adapter) is the one of ``model``; the prompt, the
normalization and the flow seed are those of ``model`` for every checkpoint. A third kind of scene is one frame
through training: one step per checkpoint, the directions fitted on all of them. ``clouds.html`` is the page.
"""
from __future__ import annotations

import glob
import io
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image
from safetensors import safe_open
from werkzeug.exceptions import HTTPException

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "text_probe"))
sys.path.insert(0, os.path.join(here, "..", "vision_swap"))

from discrete_swap import SwapConfig
from layers import KINDS, Model
from run_text_probe import label

from lerobot.configs import PreTrainedConfig, parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.modeling_molmoact2 import register_action_attention_probing
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import _snap_position, build_episode_index, register_config_choices, sample_episodes_evenly
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

GRID = 14  # image patch positions per side of one camera slot
FAR = 4  # a point is far at a hidden index when its vector is longer than FAR x the median of the fitted ones
STILL = 5e-4  # a point this near its recorded place (in units of the cloud) is not moved
RATE = 3  # steps of an episode per second of footage
FIT = 6  # steps of an episode the directions are fitted on
SHOWN, FAR_AWAY, ABSENT = 0, 1, 2  # what a point is at one step and hidden index


class Stop(Exception):
    """Ends a forward before the action expert."""


@dataclass
class CloudsConfig(SwapConfig):
    model: str = "outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/001200/pretrained_model"
    # more checkpoints, about 10 GB of GPU each: "base", training steps of the run of ``model``, and name=path of any other
    others: str = "base,100,200,400,700,internal=outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model"
    root: str = "outputs/data/rebot_val_2026-11-15-tube-merged-v1"
    frames_per_episode: int = 4
    episode_frames: int = 120  # most steps of one episode; a longer episode gets fewer steps per second
    port: int = 8773


def directions(x: torch.Tensor, fit: torch.Tensor, before: torch.Tensor | None) -> dict:
    """``mu`` [D], ``P`` [3, D], ``s`` and the points [n, 3] of the vectors x [n, D]; fitted on the rows ``fit``,
    turned towards the points ``before`` of the index below."""
    mu = x[fit].mean(0)
    _, singular, rows = torch.linalg.svd(x[fit] - mu, full_matrices=False)
    P = rows[:3]
    points = (x - mu) @ P.T
    if before is not None:  # orthogonal Procrustes on the fitted rows
        u, _, vt = torch.linalg.svd(points[fit].T @ before[fit])
        turn = u @ vt
        P, points = turn.T @ P, points @ turn
    s = torch.quantile(points[fit].norm(dim=-1), 0.95).clamp_min(1e-6)
    return {"mu": mu, "P": P, "s": s, "points": points / s, "share": float(singular[:3].square().sum() / singular.square().sum()),
            "spread": (x[fit] - mu).norm(dim=-1).mean()}


def project(d: dict, x: torch.Tensor, start: int = 0) -> torch.Tensor:
    """The points [l, n, 3] of the vectors x [l, n, D] of the indices ``start`` and up."""
    return torch.einsum("lnd,lkd->lnk", x.float() - d["mu"][start:, None], d["P"][start:]) / d["s"][start:, None, None]


class Clouds(Model):
    def __init__(self, cfg: CloudsConfig):
        self.cfg, self.device = cfg, get_safe_torch_device(try_device=cfg.policy.device)
        base_path, tokenizer = cfg.policy.base_path, cfg.policy.discrete_action_tokenizer
        cfg.policy = PreTrainedConfig.from_pretrained(cfg.model)
        # the saved config has the paths of the day it was written
        cfg.policy.base_path, cfg.policy.discrete_action_tokenizer, cfg.policy.pretrained_path = base_path, tokenizer, cfg.model

        self.dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.root)
        self.adapter = ProbablePolicy.for_config(cfg, self.device, dataset=self.dataset)
        self.adapter._set_probe_cuda_graph_enabled(False)
        self.policy = self.adapter.policy
        self.text = next(s for s in self.adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)).processor.tokenizer
        self.expert = self.policy._backbone()._require_action_expert()
        self.blocks = self.policy._backbone().transformer.blocks
        register_action_attention_probing(self.expert)
        self.kinematics, self.joints = RebotKinematics(), len(self.dataset.meta.features["action"]["names"])
        self.cameras = list(self.adapter._configured_image_keys())  # one image slot each, in this order

        self.episodes = build_episode_index(self.dataset)
        samples = sample_episodes_evenly(self.dataset, cfg.frames_per_episode, None, 0, stride=cfg.policy.image_stride)
        self.offered = [index for _, _, index in samples]  # the frames to pick from; a frame is named by its dataset index
        self.places, self.shots, self.kept, self.current, self.made = {}, {}, {}, None, 0

        # the weights of every checkpoint under the names of the policy's parameters, in training order
        self.params = dict(self.policy.named_parameters())
        checkpoints = os.path.dirname(os.path.dirname(cfg.model))
        self.main = str(int(os.path.basename(os.path.dirname(cfg.model))))
        own = {key: p.data for key, p in self.params.items()}
        self.weights, self.using = {}, self.main
        paths = dict(other.split("=") for other in cfg.others.split(",") if "=" in other)  # the checkpoints of other runs, after the steps
        names = {other for other in cfg.others.split(",") if other and "=" not in other} | {self.main}
        for name in sorted(names, key=lambda name: -1 if name == "base" else int(name)) + list(paths):
            if name == self.main:
                self.weights[name] = own
                continue
            if name == "base":  # the released files: several shards, names without the policy wrapper's leading "model."
                files, prefix = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors"))), "model."
            else:
                folder = paths[name] if name in paths else os.path.join(checkpoints, f"{int(name):06d}", "pretrained_model")
                files, prefix = [os.path.join(folder, "model.safetensors")], ""
            found, new = dict(own), 0
            for file in files:
                with safe_open(file, framework="pt") as f:
                    for key in f.keys():
                        if prefix + key in own:
                            value = f.get_tensor(key).to(self.device, own[prefix + key].dtype)
                            if not torch.equal(value, own[prefix + key]):  # a parameter that did not train is kept once
                                found[prefix + key], new = value, new + value.numel() * value.element_size()
            self.weights[name] = found
            logging.info(f"[clouds] checkpoint {name}: {new / 1e9:.1f} GB of its own")
        self.progress, self.loading = [0, 0], threading.Lock()

    def shot(self, at: int) -> dict:
        with self.loading:
            if at not in self.shots:
                self.shots[at] = self.frame(at)
            return self.shots[at]

    def use(self, name: str) -> None:
        """Points the policy's parameters at the weights of a checkpoint."""
        if name != self.using:
            for key, p in self.params.items():
                p.data = self.weights[name][key]
            self.using = name

    def steps_of(self, episode: int) -> list:
        """The frames of an episode's steps: ``RATE`` per second on the image grid, fewer when that is more than
        ``episode_frames``."""
        indices, stride = self.episodes[episode], self.cfg.policy.image_stride
        gap = max(1, round(self.cfg.env.fps / RATE / stride)) * stride
        while len(indices[::gap]) > self.cfg.episode_frames:
            gap += stride
        return [_snap_position(self.dataset, indices, position, stride)[2] for position in range(0, len(indices), gap)]

    def own(self, item: tuple) -> dict:
        """What the runs of an item (frame, checkpoint) need: the layout of its positions, its packed batch, its
        image tokens; and per attended position the expert's cross-attention weight (mean over blocks, heads and
        action positions)."""
        if item not in self.kept:
            self.use(item[1])
            f, held, make = self.shot(item[0]), {}, self.adapter._make_batch
            self.adapter._make_batch = lambda *args, **kwargs: held.setdefault("batch", make(*args, **kwargs))
            try:
                run = self.capture(f, None, 0)
            finally:
                del self.adapter._make_batch
            layout = self.positions(run)
            self.kept[item] = {"layout": layout, "batch": held["batch"], "tokens": run["tokens"], "attention": run["attention"][:, layout["index"]].mean(0)}
        return self.kept[item]

    def colors(self, item: tuple) -> list:
        """Per attended position of an item the mean colour of its image patch, None where it has none."""
        f, layout = self.shot(item[0]), self.own(item)["layout"]
        cells = {}
        for slot, key in enumerate(self.cameras):
            if key in f["obs"]:
                cells[slot] = torch.nn.functional.adaptive_avg_pool2d(f["obs"][key][0].float().clamp(0, 1), GRID).flatten(1).T.cpu().numpy()
        out = []
        for number in layout["image"]:
            slot, cell = divmod(number, GRID * GRID)
            out.append(None if number < 0 or slot not in cells else cells[slot][cell].tolist())
        return out

    @torch.no_grad()
    def forward(self, item: tuple, level: int, index: torch.Tensor | None, change: torch.Tensor | None, full: bool) -> dict:
        """One forward of an item (frame, checkpoint) under flow seed 0 with ``change`` [len(index), D] added to the input of
        block ``level`` at the sequence positions ``index``. ``hidden`` [37 - level, S, D]: the changed input of
        block ``level``, then every block's output from there. With ``full`` also ``expert`` [36, T, 768] (first
        flow step) and ``tip`` [T, 3]; without it the forward stops where the action expert would start. The
        vision backbone is not run again: its recorded output is put in place."""
        self.use(item[1])
        own, f, kept, done = self.kept[item], self.shot(item[0]), {"hidden": {}, "expert": {}}, []
        vision, expert = self.policy._backbone().vision_backbone, self.expert

        def edit(_module, args, kwargs):
            if done:
                return None
            done.append(True)
            hidden = args[0] if args else kwargs["hidden_states"]
            if change is not None:
                hidden = hidden.clone()
                hidden[..., index, :] += change.to(hidden.dtype)
            kept["hidden"][level] = hidden.detach().reshape(-1, *hidden.shape[-2:])[0]
            return ((hidden, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": hidden})

        def first(store, at_index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                store.setdefault(at_index, hidden.detach().reshape(-1, *hidden.shape[-2:])[0])

            return hook

        def stop(**_kwargs):
            raise Stop

        handles = [self.blocks[level].register_forward_pre_hook(edit, with_kwargs=True)]
        handles += [block.register_forward_hook(first(kept["hidden"], i + 1)) for i, block in enumerate(self.blocks) if i >= level]
        handles += [block.register_forward_hook(first(kept["expert"], i)) for i, block in enumerate(expert.blocks)]
        self.adapter._make_batch = lambda *args, **kwargs: own["batch"]
        vision.forward = lambda *args, **kwargs: own["tokens"]
        if not full:
            expert.prepare_context = stop
        out = {}
        try:
            chunk, _, _ = self.adapter.predict_action_chunk_in_mode(
                f["obs"], f["task"], inference_action_mode="continuous", state=f["state"], subtask=f["subtask"] or None,
                metadata=f["metadata"], generator=torch.Generator(device=self.device).manual_seed(0),
            )
            out = {"expert": torch.stack([kept["expert"][i] for i in sorted(kept["expert"])]), "tip": self.tips(chunk.float().cpu())}
        except Stop:
            pass
        finally:
            del self.adapter._make_batch, vision.forward
            if not full:
                del expert.prepare_context
            for handle in handles:
                handle.remove()
        return out | {"hidden": torch.stack([kept["hidden"][i] for i in range(level, len(self.blocks) + 1)])}

    @torch.no_grad()
    def scene(self, steps: tuple, kinds: tuple, fit: tuple) -> dict:
        """The scene of ``steps`` (per step the items shown together) with the positions of ``kinds`` as points
        and the directions fitted on the steps ``fit``. One scene is kept. ``per`` [step][item]: the item, its
        sequence positions ``index``, their points ``slot``, their vectors as recorded ``base`` [37, n, D] and
        ``far`` [37, n], its expert points and tip path. ``cloud`` [steps, 37, points, 3]. ``blob``: everything
        the page draws, float32 (see ``clouds.html``)."""
        if self.current is not None and self.current["key"] == (steps, kinds):
            return self.current
        used = {item for step in steps for item in step}
        self.current = None  # the vectors of the scene before go first
        self.shots = {at: f for at, f in self.shots.items() if at in {item[0] for item in used} or at in self.offered}
        self.kept = {item: k for item, k in self.kept.items() if item in used}
        self.progress = [0, len(steps)]
        wanted = torch.tensor(kinds, device=self.device)

        def run(at):  # at: an item
            own = self.own(at)
            rows = torch.isin(own["layout"]["kind"], wanted).nonzero().flatten()
            index, out = own["layout"]["index"][rows], self.forward(at, 0, None, None, True)
            self.progress[0] += 1
            return {"at": at, "rows": rows, "index": index, "base": out["hidden"][:, index], "expert": out["expert"], "tip": out["tip"]}

        runs = {t: [run(at) for at in steps[t]] for t in fit}
        hidden = torch.cat([r["base"] for t in fit for r in runs[t]], dim=1).float()
        expert = torch.cat([r["expert"] for t in fit for r in runs[t]], dim=1).float()
        length = hidden.norm(dim=-1)
        middle = length.median(dim=1).values
        far, found, before = length > FAR * middle[:, None], [], None
        for x, out in zip(hidden, far):
            found.append(directions(x, ~out, before))
            before = found[-1]["points"]
        language = {k: torch.stack([d[k] for d in found]) for k in ("mu", "P", "s", "spread")} | {"share": [d["share"] for d in found]}
        found, before = [], None
        for x in expert:
            found.append(directions(x, torch.ones(len(x), dtype=torch.bool, device=x.device), before))
            before = found[-1]["points"]
        action = {k: torch.stack([d[k] for d in found]) for k in ("mu", "P", "s")} | {"share": [d["share"] for d in found]}
        del hidden, expert, length, found, before

        slots, points, per = {}, [], []
        for t, step in enumerate(steps):
            per.append(runs.pop(t) if t in runs else [run(at) for at in step])
            for order, r in enumerate(per[-1]):
                layout, colors, met, slot = self.kept[r["at"]]["layout"], self.colors(r["at"]), {}, []
                kind = layout["kind"].tolist()
                for i in r["rows"].tolist():
                    number = layout["image"][i]
                    key = (order, kind[i], number if number >= 0 else -1 - met.get(kind[i], 0))  # the same point at every step
                    met[kind[i]] = met.get(kind[i], 0) + 1
                    if key not in slots:
                        slots[key] = len(slots)
                        points.append({"frame": order, "kind": kind[i], "image": number})
                    slot.append(slots[key])
                rows = r["rows"].tolist()
                r |= {
                    "slot": torch.tensor(slot, device=self.device), "far": r["base"].float().norm(dim=-1) > FAR * middle[:, None],
                    "points": project(language, r["base"]), "action": project(action, r["expert"]),
                    "tokens": [layout["tokens"][i] for i in rows], "rgb": [colors[i] or [-1, -1, -1] for i in rows],
                    "attention": self.kept[r["at"]]["attention"][r["rows"]].float(),
                }
                del r["expert"]

        count, hidden_indices, steps_ahead = len(points), len(self.blocks) + 1, per[0][0]["tip"].shape[0]
        cloud = torch.zeros(len(steps), hidden_indices, count, 3, device=self.device)
        what = torch.full((len(steps), hidden_indices, count), float(ABSENT), device=self.device)
        rgb, attention = torch.full((len(steps), count, 3), -1.0), torch.zeros(len(steps), count)
        tokens = [[""] * count for _ in steps]
        for t, step in enumerate(per):
            for r in step:
                cloud[t][:, r["slot"]], what[t][:, r["slot"]] = r["points"], r["far"].float() * FAR_AWAY
                rgb[t, r["slot"].cpu()], attention[t, r["slot"].cpu()] = torch.tensor(r["rgb"], dtype=torch.float32), r["attention"].cpu()
                for slot, token in zip(r["slot"].tolist(), r["tokens"]):
                    tokens[t][slot] = token
                del r["points"], r["tokens"], r["rgb"], r["attention"]
        acts = torch.stack([torch.cat([r["action"] for r in step], dim=1) for step in per])  # [steps, 36, frames x T, 3]
        tips = np.stack([np.concatenate([r["tip"] for r in step]) for step in per])
        blob = np.concatenate([np.asarray(part, dtype="<f4").ravel() for part in (
            cloud.cpu().numpy(), what.cpu().numpy(), acts.cpu().numpy(), tips, rgb.numpy(), attention.numpy())]).tobytes()

        self.made += 1
        shots = [[(r["at"], self.shot(r["at"][0])) for r in step] for step in per]
        frames = [[{"index": item[0], "checkpoint": item[1], "episode": int(f["episode_idx"]), "frame": int(f["frame_idx"]),
                    "step": f["subtask"] or f["task"]} for item, f in step] for step in shots]
        meta = {
            "id": self.made, "steps": len(steps), "points": points, "tokens": tokens, "frames": frames, "ahead": steps_ahead,
            "share": [round(x, 4) for x in language["share"]], "action_share": [round(x, 4) for x in action["share"]],
        }
        self.current = {"key": (steps, kinds), "meta": meta, "blob": blob, "per": per, "cloud": cloud, "language": language, "action": action}
        return self.current


def serve(cfg: CloudsConfig) -> None:
    init_logging()
    model = Clouds(cfg)
    app = Flask(__name__)
    busy = threading.Lock()  # the model runs one request at a time

    def one_at_a_time(view):
        def locked(*args, **kwargs):
            with busy:
                return view(*args, **kwargs)

        locked.__name__ = view.__name__
        return locked

    @app.errorhandler(Exception)
    def failed(error):
        if isinstance(error, HTTPException):
            return error
        logging.exception("[clouds] request failed")
        return jsonify(error=f"{type(error).__name__}: {error}"), 500

    @app.get("/")
    def page():
        return send_file(os.path.join(here, "clouds.html"))

    @app.get("/frames")
    def listing():
        offered = [model.shot(at) for at in model.offered]
        present = [key for key in model.cameras if key in offered[0]["obs"]]
        return jsonify(
            model=label(cfg.model), root=cfg.root, checkpoints=list(model.weights), main=model.main, kinds=KINDS, grid=GRID, blocks=len(model.blocks), expert_blocks=len(model.expert.blocks),
            cameras=[{"key": key, "slot": model.cameras.index(key)} for key in present], fps=cfg.env.fps,
            frames=[{"index": at, "episode": int(f["episode_idx"]), "frame": int(f["frame_idx"]), "step": f["subtask"] or f["task"]} for at, f in zip(model.offered, offered)],
            episodes=[{"episode": episode, "index": indices[0], "seconds": round(len(indices) / cfg.env.fps, 1)} for episode, indices in sorted(model.episodes.items())],
        )

    @app.get("/image/<int:at>/<path:key>")
    def image(at, key):
        buffer = io.BytesIO()
        Image.fromarray((model.shot(at)["obs"][key][0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()).save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg", max_age=3600)

    @app.get("/progress")
    def progress():
        return jsonify(done=model.progress[0], total=model.progress[1])

    @app.get("/scene")
    @one_at_a_time
    def scene():
        """What the scene is: ``frames=`` dataset indices shown side by side or ``episode=``, both under the
        checkpoint ``model``, or ``training=`` one dataset index under every checkpoint. The numbers come from
        ``/blob``. A scene through training also has ``drift`` [step][other step][37]: per hidden index
        ``mean |x - x_other| / mean |x - mu|`` over the points far at neither."""
        kinds = tuple(sorted(int(k) for k in request.args["kinds"].split(",")))
        if "training" in request.args:
            steps = tuple(((int(request.args["training"]), name),) for name in model.weights)
            fit = tuple(sorted({int(t) for t in np.linspace(0, len(steps) - 1, FIT)}))
        elif "episode" in request.args:
            steps = tuple(((at, model.main),) for at in model.steps_of(int(request.args["episode"])))
            fit = tuple(sorted({int(t) for t in np.linspace(0, len(steps) - 1, FIT)}))
        else:
            steps, fit = (tuple((int(at), model.main) for at in request.args["frames"].split(",")),), (0,)
        s = model.scene(steps, kinds, fit)
        if "training" in request.args and "drift" not in s["meta"]:
            runs = [step[0] for step in s["per"]]
            s["meta"]["drift"] = [[[round(float(x), 4) for x in (((a["base"] - b["base"]).float().norm(dim=-1) * ~(a["far"] | b["far"])).sum(1)
                                    / (~(a["far"] | b["far"])).sum(1).clamp_min(1) / s["language"]["spread"])] for b in runs] for a in runs]
        return jsonify(s["meta"])

    @app.get("/blob")
    def blob():
        """float32, in order: the points [steps, 37, points, 3], what each is [steps, 37, points] (``SHOWN``,
        ``FAR_AWAY``, ``ABSENT``), the expert's points [steps, 36, actions, 3], the tip paths [steps, actions, 3]
        (cm), the patch colours [steps, points, 3] (-1: none), the expert's attention [steps, points]."""
        s = model.current
        assert s is not None and s["meta"]["id"] == int(request.args["id"]), "another scene was loaded since"
        return app.response_class(s["blob"], mimetype="application/octet-stream")

    @app.post("/move")
    @one_at_a_time
    def move():
        """Body: float32 [points, 3], the place of every point at hidden index ``level`` of step ``step``; a point
        less than ``STILL`` from its recorded place counts as not moved. Answer: float32, in order the points of
        every hidden index [37, points, 3], the change per hidden index [37] and, with ``full=1``, the expert's
        points [36, actions, 3] and the tip paths [actions, 3]."""
        start = time.perf_counter()
        s = model.current
        assert s is not None and s["meta"]["id"] == int(request.args["id"]), "another scene was loaded since"
        t, level, full = int(request.args["step"]), int(request.args["level"]), request.args["full"] == "1"
        language, recorded = s["language"], s["cloud"][t]
        places = torch.frombuffer(bytearray(request.get_data()), dtype=torch.float32).reshape(-1, 3).to(model.device)
        out, total, count, acts, tips = recorded.clone(), torch.zeros(len(recorded), device=model.device), 0, [], []
        for r in s["per"][t]:
            shift = places[r["slot"]] - recorded[level, r["slot"]]
            shift[shift.norm(dim=-1) < STILL] = 0
            near = ~r["far"]
            count = count + near.sum(1)
            if not bool(shift.any()):
                acts.append(r["action"])
                tips.append(r["tip"])
                continue
            new = model.forward(r["at"], level, r["index"], (shift * language["s"][level]) @ language["P"][level], full)
            hidden = new["hidden"][:, r["index"]]
            out[level:, r["slot"]] = project(language, hidden, level)
            total[level:] += ((hidden - r["base"][level:]).float().norm(dim=-1) * near[level:]).sum(1)
            if full:
                acts.append(project(s["action"], new["expert"]))
                tips.append(new["tip"])
        parts = [out, total / count.clamp_min(1) / language["spread"]]
        if full:
            parts += [torch.cat(acts, dim=1), torch.from_numpy(np.concatenate(tips))]
        body = torch.cat([part.flatten().float().cpu() for part in parts])
        answer = app.response_class(body.numpy().astype("<f4").tobytes(), mimetype="application/octet-stream")
        answer.headers["X-Ms"] = f"{1000 * (time.perf_counter() - start):.0f}"
        return answer

    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # no line per request
    logging.info(f"[clouds] page on http://localhost:{cfg.port}")
    # a browser keeps idle connections open, which stalls a server with one thread
    app.run(host="127.0.0.1", port=cfg.port, threaded=True)


@parser.wrap()
def cli(cfg: CloudsConfig):
    serve(cfg)


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
