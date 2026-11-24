"""Layer by layer page: two inputs at the image positions of one model, compared through its whole depth.

    uv run python lerobot/adhocs/vision_swap/layers.py --config_path=config_rl_validate.yaml

then open http://localhost:8772. The checkpoint ``model`` is loaded whole and stays on the GPU. An input is what
the language model receives at the image positions of a frame (the output of the vision backbone, one vector per
image patch position); everything else of the prompt is the frame's own. With ``a`` the frame's own tokens and
``t`` the tokens of a source, an input is ``a + fraction * (t - a)`` on the chosen image positions:

    own       the model's own tokens
    base      the tokens of the vision path of ``vision`` (computed with its weights at the start)
    next      the model's tokens of the camera images ``next_seconds`` later in the same episode
    other     the model's tokens of the camera images of the same slot of the next episode
    mean      per image position the mean of the own tokens over the frames of the page (no image content)
    random    per token a random direction of the length of (base - own) of that token

Positions: all image positions, the long ones (own token longer than ``LONG`` x the median) or the others.

One run is one forward under one flow seed. Kept: the input of the first language-model block and every block's
output at every position, the keys and values the action expert reads per layer, the expert's block output and
cross-attention output and its cross-attention weights (first flow step), the chunks of ``seeds`` flow seeds and
the log-probabilities of the training forward at the recorded action tokens. Two runs X and A are compared
position by position (``MEASURES``); ``m`` is the mean of the own-token runs over the frames of the page, per
layer and position: the same image patch, depth, state or end-of-prompt token; for the text clauses, whose length
varies, over the tokens of the clause; in the expert the same action position (all runs share the flow noise). ``layers.html`` is the page.
"""
from __future__ import annotations

import io
import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "text_probe"))

from discrete_swap import SwapConfig, image_tokens, with_images

from lerobot.configs import PreTrainedConfig, parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.molmoact2.modeling_molmoact2 import _MOLMOACT2_PROBING_CAPTURE, register_action_attention_probing
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.objective import flow_timestep_grid
from lerobot.probes.utils import probe_frame_inputs, register_config_choices, sample_episodes_evenly
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from backbone_cross import parts
from layer_capture import label_tokens
from run_text_probe import label

LONG = 10  # an image token is long when its length is over LONG x the median of the frame's image tokens
LANGUAGE, EXPERT = ["hidden", "keys", "values"], ["cross", "expert"]
MEASURES = ["ratio", "centered", "cosine", "distance", "length"]  # |x-a| / |a-m|, cos(x-m, a-m), cos(x, a), |x-a|, |a|
KINDS = ["image, ordinary", "image, long", "depth", "task", "step", "state", "quality clauses", "end of prompt", "other text"]
KIND_OF_GROUP = {"control_mode": "task", "task": "task", "subtask": "step", "state": "state", "metadata": "quality clauses", "question": "end of prompt"}
ALIGNED = {"depth", "state", "end of prompt"}  # clauses of the same length in every frame: m is per token
PLACES = 1024  # room for the means m
WEIGHTED, ACTION ="all positions, weighted by the expert's attention", "action positions"
OWN = ("own", "all", 1.0)


@dataclass
class LayersConfig(SwapConfig):
    frames_per_episode: int = 5
    seeds: int = 4
    kept_runs: int = 24  # runs kept on the GPU (about 0.3 GB each)
    port: int = 8772


class First(dict):
    """Keeps the first value written per key: the attention of the first flow step."""

    def __setitem__(self, key, value):
        if key not in self:
            super().__setitem__(key, value)


def measures(x: torch.Tensor, a: torch.Tensor, m: torch.Tensor) -> dict:
    """``MEASURES`` per layer and position, each [L, n], for x, a, m of [L, n, D]."""
    x, a, m = x.float(), a.float(), m.float()
    cosine = torch.nn.functional.cosine_similarity
    distance = (x - a).norm(dim=-1)
    return {"ratio": distance / (a - m).norm(dim=-1).clamp_min(1e-6), "centered": cosine(x - m, a - m, dim=-1), "cosine": cosine(x, a, dim=-1),
            "distance": distance, "length": a.norm(dim=-1)}


def rounded(x) -> list:
    return np.round(torch.as_tensor(x).float().cpu().numpy().astype(np.float64), 4).tolist()


class Model:
    def __init__(self, cfg: LayersConfig):
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
        self.grid = torch.from_numpy(flow_timestep_grid(cfg.policy, int(cfg.policy.num_flow_timesteps))).float()
        self.discrete_loss = self.policy._discrete_loss_from_backbone_outputs
        self.kinematics, self.joints = RebotKinematics(), len(self.dataset.meta.features["action"]["names"])
        self.cameras = list(self.adapter._configured_image_keys())  # one image slot each, in this order

        samples = sample_episodes_evenly(self.dataset, cfg.frames_per_episode, None, 0, stride=cfg.policy.image_stride)
        self.frames = [self.frame(index) for _, _, index in samples]
        shift = round(cfg.next_seconds * cfg.env.fps / cfg.policy.image_stride) * cfg.policy.image_stride
        self.later = []
        for f in self.frames:
            candidate = self.frame(f["global_idx"] + shift) if f["global_idx"] + shift < self.dataset.num_frames else None
            if candidate is None or candidate["episode_idx"] != f["episode_idx"]:
                candidate = self.frame(f["global_idx"] - shift)
            assert candidate["episode_idx"] == f["episode_idx"]
            self.later.append(candidate)
        self.other = [self.frames[(at + cfg.frames_per_episode) % len(self.frames)] for at in range(len(self.frames))]
        assert all(o["episode_idx"] != f["episode_idx"] for o, f in zip(self.other, self.frames))

        # the other vision path's tokens of every frame, with its weights in place
        own_vision, other_vision = parts(cfg.model, base_path)["vision"], parts(cfg.vision, base_path)["vision"]
        assert set(own_vision) == set(other_vision)
        self.policy.load_state_dict(other_vision, strict=False)
        swapped = [self.chunk(f, f["obs"], 0, None) for f in self.frames]
        self.policy.load_state_dict(own_vision, strict=False)
        self.base = [tokens for _, tokens in swapped]

        # one own-token run per frame: its tokens, its positions, and the sums for the means m
        self.runs, self.sources = OrderedDict(), {}
        self.own, self.layout, self.places, total, count = [], [], {}, {}, torch.zeros(PLACES, device=self.device)
        for at, f in enumerate(self.frames):
            run = self.capture(f, None, 0)
            self.own.append(run["tokens"])
            self.layout.append(self.positions(run))
            index, place = self.layout[at]["index"], self.layout[at]["place"]
            count.index_add_(0, place, torch.ones(len(place), device=self.device))
            for quantity in LANGUAGE:
                x = run[quantity][:, index].float()
                total.setdefault(quantity, torch.zeros(x.shape[0], PLACES, x.shape[2], device=self.device)).index_add_(1, place, x)
            for quantity in EXPERT:
                total[quantity] = total.get(quantity, 0) + run[quantity].float()
        self.means = {q: total[q] / count.clamp_min(1)[None, :, None] for q in LANGUAGE} | {q: total[q] / len(self.frames) for q in EXPERT}
        self.mean_tokens = torch.stack([a.float() for a in self.own]).mean(0)
        check = self.chunk(self.frames[0], self.frames[0]["obs"], 0, lambda _tokens: self.base[0])[0]
        logging.info(f"[layers] base tokens put in place against base weights, largest chunk difference (deg): {float((check - swapped[0][0]).abs().max())}")
        logging.info(f"[layers] {len(self.frames)} frames | positions per kind on frame 0: { {k: int((self.layout[0]['kind'] == i).sum()) for i, k in enumerate(KINDS)} }")

    def frame(self, index):
        return probe_frame_inputs(self.dataset, self.cfg, index, self.cfg.policy.chunk_size)

    def chunk(self, f, obs, seed, fn):
        with image_tokens(self.policy, fn) as vision:
            out, _, _ = self.adapter.predict_action_chunk_in_mode(
                obs, f["task"], inference_action_mode="continuous", state=f["state"], subtask=f["subtask"] or None,
                metadata=f["metadata"], generator=torch.Generator(device=self.device).manual_seed(seed),
            )
        assert len(vision.seen) == 1, f"vision backbone ran {len(vision.seen)} times in one forward"
        return out.float().cpu(), vision.seen[0]

    @torch.no_grad()
    def capture(self, f, fn, seed) -> dict:
        """One forward under one flow seed, everything kept on the device: ``hidden`` [37, S, D] (the input of the
        first language-model block, then every block's output), ``keys`` / ``values`` [36, S, 768] (what the action
        expert's cross-attention reads), ``cross`` / ``expert`` [36, T, 768] and ``attention`` [36, S] (the
        cross-attention weights, mean over heads and action positions), the last three of the first flow step."""
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
        _MOLMOACT2_PROBING_CAPTURE.clear()
        _MOLMOACT2_PROBING_CAPTURE.update(enabled=True, requires_grad=False, cross_attn_by_layer=First(), self_attn_by_layer=First())
        try:
            chunk, tokens = self.chunk(f, f["obs"], seed, fn)
            weights = _MOLMOACT2_PROBING_CAPTURE["cross_attn_by_layer"]
            attention = torch.stack([weights[i][0].float().mean(dim=(0, 1)) for i in range(len(expert.blocks))]).to(self.device)
        finally:
            _MOLMOACT2_PROBING_CAPTURE.clear()
            del policy._model_inputs, expert.prepare_context
            for handle in handles:
                handle.remove()
        mask = seen.get("attention_mask")
        return {
            "chunk": chunk, "tokens": tokens, "attention": attention,
            "ids": seen["input_ids"][0].cpu(), "mask": None if mask is None else mask[0].cpu().bool(),
            "hidden": torch.stack([kept["hidden"][i] for i in sorted(kept["hidden"])]),
            "keys": torch.stack([k for k, _ in kept["context"]]), "values": torch.stack([v for _, v in kept["context"]]),
            "expert": torch.stack([kept["expert"][i] for i in sorted(kept["expert"])]),
            "cross": torch.stack([kept["cross"][i] for i in sorted(kept["cross"])]),
        }

    def positions(self, run: dict) -> dict:
        """The attended positions of a frame's prompt: ``index`` into the sequence, the token as text, the kind
        (index into ``KINDS``), for an image position the number of its image token (-1 otherwise), and the
        ``place`` whose mean over frames is its m: the same image patch, the same token of a clause of fixed
        length (``ALIGNED``), or the whole clause where the length varies with the frame."""
        tokens, groups = label_tokens(self.adapter, self.text, {"ids": run["ids"], "mask": run["mask"], "encoder": run["hidden"][1:]})
        length = run["tokens"].float().norm(dim=-1)
        long = (length > LONG * length.median()).cpu()
        patch = (run["ids"] == self.adapter._image_patch_token_id()).nonzero().flatten().tolist()
        assert len(patch) == len(long), (len(patch), len(long))
        number = dict(zip(patch, range(len(patch))))
        question = groups.index("question")
        index, kind, image, place, met = [], [], [], [], {}
        for at, group in enumerate(groups):
            if group == "masked":
                continue
            if group == "image":
                name = "image, long" if long[number[at]] else "image, ordinary"
            elif tokens[at] == "<extra_1>":
                name = "depth"
            else:
                name = "end of prompt" if at >= question else KIND_OF_GROUP.get(group, "other text")
            index.append(at)
            kind.append(KINDS.index(name))
            image.append(number.get(at, -1))
            key = ("image", number[at]) if group == "image" else (name, met.get(name, 0) if name in ALIGNED else 0)
            met[name] = met.get(name, 0) + 1
            place.append(self.places.setdefault(key, len(self.places)))
        assert len(self.places) <= PLACES, len(self.places)
        return {"index": torch.tensor(index, device=self.device), "kind": torch.tensor(kind, device=self.device), "image": image,
                "place": torch.tensor(place, device=self.device),
                "tokens": [tokens[at] for at in index], "long": long.to(self.device)}

    def source(self, at: int, name: str) -> torch.Tensor:
        """The tokens of a source on frame ``at``, [image positions, width]."""
        if name == "base":
            return self.base[at]
        if name == "mean":
            return self.mean_tokens
        if name == "random":
            direction = torch.nn.functional.normalize(torch.randn(self.own[at].shape, generator=torch.Generator(device=self.device).manual_seed(at), device=self.device), dim=-1)
            return self.own[at].float() + direction * (self.base[at].float() - self.own[at].float()).norm(dim=-1, keepdim=True)
        if (at, name) not in self.sources:
            f, images = self.frames[at], {"next": self.later, "other": self.other}[name][at]
            self.sources[(at, name)] = self.chunk(f, with_images(f["obs"], images["obs"]), 0, None)[1]
        return self.sources[(at, name)]

    def put(self, at: int, spec: tuple):
        """``fn`` for ``image_tokens``: the input ``spec`` = (source, positions, fraction) on frame ``at``."""
        source, where, fraction = spec
        if source == "own" or fraction == 0:
            return None
        a, long = self.own[at], self.layout[at]["long"]
        weight = {"all": torch.ones_like(long), "long": long, "ordinary": ~long}[where].float()[:, None]
        new = (a.float() + fraction * weight * (self.source(at, source).float() - a.float())).to(a.dtype)
        return lambda _tokens: new

    def log_probabilities(self, f, fn):
        """Of the training forward on the recorded chunk, at the recorded action token positions: [positions,
        vocabulary], and the recorded tokens."""
        kept = {}

        def keep(batch, outputs, **kwargs):
            kept["hidden"], kept["labels"] = outputs.last_hidden_state, batch["labels"]
            return self.discrete_loss(batch, outputs, **kwargs)

        self.policy._discrete_loss_from_backbone_outputs = keep
        try:
            with image_tokens(self.policy, fn):
                out = self.adapter.training_losses({**f, "obs": f["obs"]}, flow_timesteps=self.grid, flow_noise_seed=int(f["global_idx"]))
        finally:
            del self.policy._discrete_loss_from_backbone_outputs
        shifted = torch.nn.functional.pad(kept["labels"], (0, 1), value=-100)[..., 1:]
        logp = torch.nn.functional.linear(kept["hidden"][shifted != -100], self.policy.model.lm_head.weight).float().log_softmax(-1)
        labels = shifted[shifted != -100]
        ce = -logp.gather(1, labels[:, None])[:, 0].cpu()
        assert torch.allclose(ce, torch.from_numpy(out["discrete_token_ce"]).flatten(), atol=2e-2), "not the logits of the token loss"
        return logp, labels

    def run(self, at: int, spec: tuple, seed: int = 0) -> dict:
        key = (at, spec, seed)
        if key in self.runs:
            self.runs.move_to_end(key)
            return self.runs[key]
        f, fn = self.frames[at], self.put(at, spec)
        run = self.capture(f, fn, seed)
        if seed == 0:  # the other flow seeds and the discrete output
            run["chunks"] = torch.stack([run["chunk"]] + [self.chunk(f, f["obs"], s, fn)[0] for s in range(1, self.cfg.seeds)])
            run["tips"] = self.tips(run["chunks"])
            run["logp"], run["labels"] = self.log_probabilities(f, fn)
        self.runs[key] = run
        while len(self.runs) > self.cfg.kept_runs:
            self.runs.popitem(last=False)
        return run

    def tips(self, chunks: torch.Tensor) -> np.ndarray:
        """[..., steps, joints] degrees -> the tip path [..., steps, 3] in cm, filtered as deployed."""
        chunks = chunks.double().numpy()[..., : self.joints]
        smooth = np.moveaxis(apply_butterworth_filter(np.moveaxis(chunks, -2, 0)), 0, -2)
        return self.kinematics.ee_path(smooth.reshape(-1, self.joints)).reshape(*smooth.shape[:-1], 3) * 100

    def per_position(self, x: dict, a: dict, at: int, quantity: str) -> dict:
        """``MEASURES`` of run x against run a, [L, n]: n = the attended positions for the language model's
        quantities, the action positions for the expert's."""
        if quantity in EXPERT:
            return measures(x[quantity], a[quantity], self.means[quantity])
        layout = self.layout[at]
        return measures(x[quantity][:, layout["index"]], a[quantity][:, layout["index"]], self.means[quantity][:, layout["place"]])

    def pair(self, at: int, x_spec: tuple, a_spec: tuple, x_seed: int = 0) -> dict:
        """Run x against run a on one frame: per quantity and kind of position the median over positions of
        every measure per layer, and the two outputs."""
        x, a = self.run(at, x_spec, x_seed), self.run(at, a_spec)
        kind, curves = self.layout[at]["kind"], {}
        for quantity in LANGUAGE:
            per = self.per_position(x, a, at, quantity)
            curves[quantity] = {name: {m: rounded(v[:, kind == i].median(dim=1).values) for m, v in per.items()} for i, name in enumerate(KINDS) if (kind == i).any()}
            if quantity != "hidden":
                weight = a["attention"][:, self.layout[at]["index"]]
                curves[quantity][WEIGHTED] = {m: rounded((v * weight).sum(1) / weight.sum(1)) for m, v in per.items()}
        for quantity in EXPERT:
            curves[quantity] = {ACTION: {m: rounded(v.median(dim=1).values) for m, v in self.per_position(x, a, at, quantity).items()}}
        out = {"curves": curves}
        if x_seed == 0:
            recorded = a["labels"][:, None]
            out["output"] = {
                "cm": float(np.linalg.norm(x["tips"] - a["tips"], axis=-1).mean()),
                "kl": float((a["logp"].exp() * (a["logp"] - x["logp"])).sum(1).mean()),
                "ce": float(a["logp"].gather(1, recorded).mean() - x["logp"].gather(1, recorded).mean()),
            }
        return out


def spec(text: str) -> tuple:
    source, where, fraction = text.split("|")
    assert source in ("own", "base", "next", "other", "mean", "random") and where in ("all", "long", "ordinary"), text
    return (source, where, float(fraction))


def serve(cfg: LayersConfig) -> None:
    init_logging()
    model = Model(cfg)
    frames = model.frames
    recorded = [model.tips(f["gt_actions"].float().cpu()) for f in frames]
    app, answers = Flask(__name__), {}  # the answers of /compare, by query
    here = os.path.dirname(os.path.abspath(__file__))

    @app.errorhandler(Exception)
    def failed(error):
        logging.exception("[layers] request failed")
        return jsonify(error=f"{type(error).__name__}: {error}")

    @app.get("/")
    def page():
        return send_file(os.path.join(here, "layers.html"))

    @app.get("/plotly.js")
    def plotly():
        from plotly.offline import get_plotlyjs  # the copy shipped with the package: no CDN needed

        return app.response_class(get_plotlyjs(), mimetype="application/javascript")

    @app.get("/frames")
    def listing():
        present = [key for key in model.cameras if key in frames[0]["obs"]]
        return jsonify(
            model=label(cfg.model), vision=label(cfg.vision), root=cfg.root, seeds=cfg.seeds, next_seconds=cfg.next_seconds, kinds=KINDS,
            weighted=WEIGHTED, action=ACTION, long=LONG, cameras=[{"key": key, "slot": model.cameras.index(key)} for key in present],
            frames=[{"episode": int(f["episode_idx"]), "frame": int(f["frame_idx"]), "step": f["subtask"] or "", "task": f["task"],
                     "next": int(model.later[at]["frame_idx"]), "other": int(model.other[at]["episode_idx"])} for at, f in enumerate(frames)],
        )

    @app.get("/image/<int:at>/<source>/<path:key>")
    def image(at, source, key):
        f = {"own": frames, "next": model.later, "other": model.other}[source][at]
        buffer = io.BytesIO()
        Image.fromarray((f["obs"][key][0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()).save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg")

    @app.get("/compare")
    def compare():
        """Input b against input a on one frame, next to the scales of that frame: the camera images 1 s later
        and of another episode against the own tokens, and the own tokens under another flow seed."""
        if request.query_string in answers:
            return answers[request.query_string]
        at, a, b = int(request.args["frame"]), spec(request.args["a"]), spec(request.args["b"])
        pairs = {"b": model.pair(at, b, a)}
        if request.args.get("scales", "1") == "1":
            pairs |= {"next": model.pair(at, ("next", "all", 1.0), OWN), "other": model.pair(at, ("other", "all", 1.0), OWN), "seed": model.pair(at, OWN, OWN, 1)}
        run_a, run_b, layout = model.run(at, a), model.run(at, b), model.layout[at]
        seeds = [(i, j) for i in range(cfg.seeds) for j in range(cfg.seeds) if i < j]
        attention = run_a["attention"][:, layout["index"]]
        answers[request.query_string] = jsonify(
            pairs=pairs,
            attention={name: rounded(attention[:, layout["kind"] == i].sum(1)) for i, name in enumerate(KINDS) if (layout["kind"] == i).any()},
            counts={name: int((layout["kind"] == i).sum()) for i, name in enumerate(KINDS)},
            two_seeds=float(np.mean([np.linalg.norm(run_a["tips"][i] - run_a["tips"][j], axis=-1).mean() for i, j in seeds])),
            to_recorded={"a": float(np.linalg.norm(run_a["tips"] - recorded[at], axis=-1).mean()), "b": float(np.linalg.norm(run_b["tips"] - recorded[at], axis=-1).mean())},
            paths={"a": np.round(run_a["tips"], 2).tolist(), "b": np.round(run_b["tips"], 2).tolist(), "recorded": np.round(recorded[at], 2).tolist()},
        )
        return answers[request.query_string]

    @app.get("/map")
    def position_map():
        """One measure of input b against input a at every layer and position of one frame."""
        at, a, b = int(request.args["frame"]), spec(request.args["a"]), spec(request.args["b"])
        quantity, layout = request.args["quantity"], model.layout[at]
        per = model.per_position(model.run(at, b), model.run(at, a), at, quantity)[request.args["measure"]]
        if quantity in EXPERT:
            return jsonify(z=rounded(per), tokens=[f"action position {i + 1}" for i in range(per.shape[1])], kinds=[ACTION] * per.shape[1], image=[-1] * per.shape[1])
        return jsonify(z=rounded(per), tokens=layout["tokens"], kinds=[KINDS[i] for i in layout["kind"].tolist()], image=layout["image"])

    @app.get("/tensor")
    def tensor():
        """For scripts: one layer of one quantity of a run as .npy, [attended positions, width] (the expert's
        quantities: [action positions, width]); quantity ``attention``: [36, attended positions]."""
        at, quantity, layout = int(request.args["frame"]), request.args["quantity"], model.layout[int(request.args["frame"])]
        run = model.run(at, spec(request.args["a"]))
        if quantity == "attention":
            x = run["attention"][:, layout["index"]]
        else:
            x = run[quantity][int(request.args["layer"])]
            x = x if quantity in EXPERT else x[layout["index"]]
        buffer = io.BytesIO()
        np.save(buffer, x.float().cpu().numpy())
        buffer.seek(0)
        return send_file(buffer, mimetype="application/octet-stream")

    logging.info(f"[layers] page on http://localhost:{cfg.port}")
    app.run(host="127.0.0.1", port=cfg.port, threaded=False)  # one request at a time: one model on the GPU


@parser.wrap()
def cli(cfg: LayersConfig):
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
