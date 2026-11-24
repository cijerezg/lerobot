"""Numbers behind outputs/analysis/layer_cosine_2026-11-16/REPORT.md, computed from the store layer_sweep.py writes.

    uv run python lerobot/adhocs/text_probe/layer_analysis.py <stage> [--store=...] [--out=...]

Stages (each writes <out>/<stage>.pt, a dict of plain lists and arrays):
  cross   same input, two checkpoints (base / tube / internal), every run of the full suite
  prompt  same checkpoint, two inputs: two prompts on one image, two images under one prompt, two instructions
  series  the saved steps of the tube run against base, step against step, and the instruction pairs per step
  expert  where the action expert's cross-attention goes, chunks under hidden kinds of token and other flow seeds
  stream  action tokens' own hidden states against the prompt's
  lens    logit lens at the last position

Raw cosine everywhere. Next to it: the relative L2 distance |a - b| / mean(|a|, |b|), the same two numbers for
shuffled positions of the same kind (what "any two vectors of this kind" give), and how often a position's
nearest neighbour in the other run, among positions of its kind, is the same position.

Norm of a position = L2 norm of the vector a language-model block writes at that position (2560 numbers).
A position has a high norm at a layer when its norm is over FACTOR (10) x the median norm of the attended
positions of that layer. cross / prompt / series report the other (ordinary) positions, count the high-norm
ones apart, and keep the plain mean over all positions as raw_cos. expert / stream / lens do not split.
The saved files keep their first key names: ``massive`` = high-norm count, ``both`` / ``flip`` = high norm in
both runs / in one only.
"""
from __future__ import annotations

import glob
import os
import sys

import torch

from layer_capture import paired

STORE = "outputs/cache/layer_cosine_2026-11-16"
OUT = "outputs/analysis/layer_cosine_2026-11-16"
BASE, TUBE, INTERNAL = "base", "rebot-v2-tube-b128-20261114__001200", "rebot-internal-only-20261021__001200"
MAIN = {"base": BASE, "tube": TUBE, "internal": INTERNAL}
TUBE_FRAMES = ["robot_200", "robot_600", "robot_1200", "robot_2000"]
CLUTTER = ["clutter_1180", "clutter_2180", "clutter_3215", "clutter_4310", "clutter_6040"]
device = torch.device("cuda")


def load(checkpoint: str, name: str, store: str = STORE) -> dict:
    run = torch.load(os.path.join(store, checkpoint, name + ".pt"), weights_only=False)
    groups = list(run["groups"])
    groups[-1] = "last"  # the position the next token is read from
    run["groups"] = groups
    return run


def members(groups: list[str]) -> dict[str, list[int]]:
    out = {"all": [i for i, g in enumerate(groups) if g != "masked"]}
    for at, group in enumerate(groups):
        if group != "masked":
            out.setdefault(group, []).append(at)
    return out


FACTOR = 10.0  # a position has a high norm at a layer when its L2 norm is over 10 x the median norm of that layer


def high_norm(layer: torch.Tensor, attended: list[int]) -> torch.Tensor:
    """[positions] bool. ``layer`` is [positions, width], float, on the device."""
    norm = layer.norm(dim=-1)
    return norm > FACTOR * norm[attended].median()


@torch.no_grad()
def profile(hidden: torch.Tensor, groups: list[str]) -> dict:
    """One run alone, per layer. A few positions have an L2 norm 100 to 3000 times the rest (high-norm
    positions); they are counted per kind and left out of every other number. Per kind over the ordinary positions:
    mean norm, mean cosine between two different positions. Over all ordinary positions: participation ratio,
    share of the squared norm in the largest 1 and 3 coordinates, share of the squared norm in the common
    (mean) vector, cosine and relative step to the previous layer."""
    index = members(groups)
    kinds = list(index)
    out = {"norm": {k: [] for k in kinds}, "pair": {k: [] for k in kinds}, "massive": {k: [] for k in kinds}, "massive_norm": [], "median": [],
           "spread": [], "top1": [], "top3": [], "common": [], "step_cos": [], "step_rel": [], "massive_positions": []}
    previous = None
    for layer in hidden:
        full = layer.to(device).float()
        big = high_norm(full, index["all"])
        unit = torch.nn.functional.normalize(full, dim=-1)
        out["massive_positions"].append([i for i in index["all"] if big[i]])
        out["massive_norm"].append(float(full[big].norm(dim=-1).mean()) if big.any() else float("nan"))
        out["median"].append(float(full[index["all"]].norm(dim=-1).median()))
        for kind, at in index.items():
            ordinary = [i for i in at if not big[i]]
            out["massive"][kind].append(len(at) - len(ordinary))
            out["norm"][kind].append(float(full[ordinary].norm(dim=-1).mean()) if ordinary else float("nan"))
            rows = unit[ordinary]
            out["pair"][kind].append(float((rows.sum(dim=0).square().sum() - len(rows)) / (len(rows) * (len(rows) - 1))) if len(ordinary) > 1 else float("nan"))
        ordinary = [i for i in index["all"] if not big[i]]
        x = full[ordinary]
        gram = x @ x.T
        out["spread"].append(float(gram.diagonal().sum() ** 2 / gram.square().sum()))
        energy = x.square()
        ordered = energy.sort(dim=-1, descending=True).values
        out["top1"].append(float((ordered[:, 0] / energy.sum(dim=-1)).mean()))
        out["top3"].append(float((ordered[:, :3].sum(dim=-1) / energy.sum(dim=-1)).mean()))
        out["common"].append(float(x.mean(dim=0).square().sum() / energy.sum(dim=-1).mean()))
        if previous is not None:
            a, b = full[index["all"]], previous
            keep = ~(big[index["all"]] | previous_big)
            out["step_cos"].append(float(torch.nn.functional.cosine_similarity(a[keep], b[keep], dim=-1).mean()))
            out["step_rel"].append(float(((a[keep] - b[keep]).norm(dim=-1) / b[keep].norm(dim=-1)).mean()))
        previous, previous_big = full[index["all"]], big[index["all"]]
    return out


@torch.no_grad()
def compare(a: torch.Tensor, b: torch.Tensor, groups_a: list[str], groups_b: list[str], retrieval: bool = True) -> dict:
    """Two runs, positions matched by kind. A matched pair is ordinary when the position has a high norm in neither
    run; ``both`` counts the pairs with a high norm in both, ``flip`` in one only. Per layer and kind, over the ordinary pairs:
    cos / rel (matched positions), null_cos / null_rel (a position of one run against ANOTHER position of the
    same kind of the other run), hit (share of positions whose nearest neighbour by cosine in the other run,
    among its kind, is the matched one; kinds of 8 or more), centred (cosine after each run's mean over its ordinary
    positions is subtracted; side check only), mean_cos (cosine between the two runs' mean vectors of the kind).
    raw_cos is the plain mean over every matched pair, high norm or not. both / flip are counts, both_cos the
    cosine over the pairs with a high norm in both."""
    first, second = paired(groups_a, groups_b)
    kinds = [groups_a[i] for i in first]
    index = {"all": list(range(len(first)))}
    for at, kind in enumerate(kinds):
        index.setdefault(kind, []).append(at)
    keys = ["cos", "rel", "null_cos", "null_rel", "hit", "centred", "mean_cos", "raw_cos", "both", "flip", "both_cos"]
    out = {key: {kind: [] for kind in index} for key in keys}
    out["count"] = {kind: len(at) for kind, at in index.items()}
    attended_a = [i for i, g in enumerate(groups_a) if g != "masked"]
    attended_b = [i for i, g in enumerate(groups_b) if g != "masked"]
    nan = float("nan")
    for x, y in zip(a, b):
        x, y = x.to(device).float(), y.to(device).float()
        big_x, big_y = high_norm(x, attended_a), high_norm(y, attended_b)
        centre_x = x[[i for i in attended_a if not big_x[i]]].mean(dim=0)
        centre_y = y[[i for i in attended_b if not big_y[i]]].mean(dim=0)
        x, y, big_x, big_y = x[first], y[second], big_x[first], big_y[second]
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        rel = (x - y).norm(dim=-1) / ((x.norm(dim=-1) + y.norm(dim=-1)) / 2)
        centred = torch.nn.functional.cosine_similarity(x - centre_x, y - centre_y, dim=-1)
        ordinary_all = ~(big_x | big_y)
        for kind, at in index.items():
            at = torch.tensor(at, device=x.device)
            ordinary = at[ordinary_all[at]]
            both = at[(big_x & big_y)[at]]
            out["raw_cos"][kind].append(float(cos[at].mean()))
            out["both"][kind].append(len(both))
            out["flip"][kind].append(len(at) - len(both) - len(ordinary))
            out["both_cos"][kind].append(float(cos[both].mean()) if len(both) else nan)
            if len(ordinary) == 0:
                for key in ("cos", "rel", "centred", "null_cos", "null_rel", "hit", "mean_cos"):
                    out[key][kind].append(nan)
                continue
            out["cos"][kind].append(float(cos[ordinary].mean()))
            out["rel"][kind].append(float(rel[ordinary].mean()))
            out["centred"][kind].append(float(centred[ordinary].mean()))
            out["mean_cos"][kind].append(float(torch.nn.functional.cosine_similarity(x[ordinary].mean(dim=0), y[ordinary].mean(dim=0), dim=0)))
            if len(ordinary) < 2 or kind == "all":
                for key in ("null_cos", "null_rel", "hit"):
                    out[key][kind].append(nan)
                continue
            xs, ys = x[ordinary], y[ordinary]
            sims = torch.nn.functional.normalize(xs, dim=-1) @ torch.nn.functional.normalize(ys, dim=-1).T
            off = ~torch.eye(len(ordinary), dtype=torch.bool, device=sims.device)
            out["null_cos"][kind].append(float(sims[off].mean()))
            distance = torch.cdist(xs, ys) / ((xs.norm(dim=-1)[:, None] + ys.norm(dim=-1)[None]) / 2)
            out["null_rel"][kind].append(float(distance[off].mean()))
            out["hit"][kind].append(float((sims.argmax(dim=1) == torch.arange(len(ordinary), device=sims.device)).float().mean()) if retrieval and len(ordinary) >= 8 else nan)
    return out


@torch.no_grad()
def expert_side(a: dict, b: dict) -> dict:
    """Two action runs with the same flow noise: what reaches the action expert and what it does with it."""
    out = {}
    first, second = paired(a["groups"], b["groups"])
    kinds = [a["groups"][i] for i in first]
    index = {"all": list(range(len(first)))}
    for at, kind in enumerate(kinds):
        index.setdefault(kind, []).append(at)
    for key in ("context_k", "context_v"):
        x, y = a[key].to(device).float()[:, first], b[key].to(device).float()[:, second]
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        rel = (x - y).norm(dim=-1) / ((x.norm(dim=-1) + y.norm(dim=-1)) / 2)
        out[key + "_cos"] = {kind: cos[:, at].mean(dim=1).tolist() for kind, at in index.items()}
        out[key + "_rel"] = {kind: rel[:, at].mean(dim=1).tolist() for kind, at in index.items()}
    for key in ("expert", "cross"):
        x, y = a[key].to(device).float(), b[key].to(device).float()
        out[key + "_cos"] = torch.nn.functional.cosine_similarity(x, y, dim=-1).mean(dim=1).tolist()
        out[key + "_rel"] = ((x - y).norm(dim=-1) / ((x.norm(dim=-1) + y.norm(dim=-1)) / 2)).mean(dim=1).tolist()
    out["chunk_rms"] = float((a["chunk"] - b["chunk"]).square().mean().sqrt())
    out["chunk_norm_rms"] = float((a["chunk_norm"] - b["chunk_norm"]).square().mean().sqrt())
    return out


def names(checkpoint: str, pattern: str, store: str = STORE) -> list[str]:
    return sorted(os.path.basename(path)[:-3] for path in glob.glob(os.path.join(store, checkpoint, pattern + ".pt")))


def cross(store: str) -> dict:
    out = {"profile": {}, "pairs": {}, "expert": {}, "counts": {}}
    for name in names(BASE, "chat.*", store) + names(BASE, "act.*", store):
        runs = {key: load(folder, name, store) for key, folder in MAIN.items()}
        out["profile"][name] = {key: profile(run["encoder"], run["groups"]) for key, run in runs.items()}
        out["counts"][name] = {kind: len(at) for kind, at in members(runs["base"]["groups"]).items()}
        for a, b in (("base", "tube"), ("base", "internal"), ("tube", "internal")):
            out["pairs"][(name, a, b)] = compare(runs[a]["encoder"], runs[b]["encoder"], runs[a]["groups"], runs[b]["groups"])
            if name.startswith("act."):
                out["expert"][(name, a, b)] = expert_side(runs[a], runs[b])
        print(name, flush=True)
    return out


def prompt_pairs(checkpoint: str, store: str, small: bool = False) -> list[tuple[str, str, str]]:
    """(family, run a, run b) for one checkpoint, from the runs it has."""
    have = set(names(checkpoint, "*", store))
    chat_images = sorted({n.split(".")[1] for n in have if n.startswith("chat.") and n.split(".")[1] != "none"})
    photos = [i for i in chat_images if i.endswith(".jpg")]
    families = [[i for i in TUBE_FRAMES if i in chat_images], [i for i in CLUTTER if i in chat_images], photos]
    pairs = []
    for image in chat_images:
        for a, b in (("describe", "objects"), ("describe", "next")):
            pairs.append(("chat prompt", f"chat.{image}.{a}", f"chat.{image}.{b}"))
    for family in families:
        for a, b in zip(family, family[1:]):
            pairs += [("chat image, same kind", f"chat.{a}.{p}", f"chat.{b}.{p}") for p in ("describe", "next")]
    for a, b in zip(families[0] + families[1], photos + photos):
        pairs.append(("chat image, unrelated", f"chat.{a}.describe", f"chat.{b}.describe"))
    for a, b in (("hi", "france"), ("france", "times"), ("tubeplan", "robotq"), ("hi", "robotq")):
        pairs.append(("chat text only", f"chat.none.{a}", f"chat.none.{b}"))
    act_images = sorted({n.split(".", 2)[2].rsplit(".", 1)[0] for n in have if n.startswith("act.step.")})
    for image in act_images:
        for a, b in (("right", "left"), ("cup", "bottle"), ("tubeleft", "tuberight"), ("right", "cup"), ("own", "right")):
            pairs.append((f"instruction {a} / {b}", f"act.step.{image}.{a}", f"act.step.{image}.{b}"))
    for family in families[:2]:
        frames = [i for i in family if i in act_images]
        for a, b in zip(frames, frames[1:]):
            pairs += [("action frame, same instruction", f"act.step.{a}.right", f"act.step.{b}.right"), ("action frame, own step", f"act.step.{a}.own", f"act.step.{b}.own")]
    for a, b in zip(families[0], families[1]):
        pairs.append(("action frame, other scene", f"act.step.{a}.right", f"act.step.{b}.right"))
    for style in ("task", "original"):
        for image in sorted({n.split(".", 2)[2].rsplit(".", 1)[0] for n in have if n.startswith(f"act.{style}.")}):
            for a, b in (("right", "left"), ("cup", "bottle")):
                pairs.append((f"{style}: instruction {a} / {b}", f"act.{style}.{image}.{a}", f"act.{style}.{image}.{b}"))
    return [pair for pair in pairs if pair[1] in have and pair[2] in have]


def prompt(store: str) -> dict:
    out = {}
    for key, folder in MAIN.items():
        for family, a, b in prompt_pairs(folder, store):
            first, second = load(folder, a, store), load(folder, b, store)
            entry = {"family": family, "lm": compare(first["encoder"], second["encoder"], first["groups"], second["groups"])}
            if a.startswith("act."):
                entry["expert"] = expert_side(first, second)
            out[(key, a, b)] = entry
        print(key, flush=True)
    return out


def series(store: str) -> dict:
    steps = sorted(folder for folder in os.listdir(store) if folder.startswith("rebot-v2-tube"))
    out = {"steps": [int(folder.split("__")[1]) for folder in steps], "base": {}, "previous": {}, "final": {}, "instruction": {}, "profile": {}}
    runs = names(steps[0], "chat.*", store) + names(steps[0], "act.*", store)
    for name in runs:
        base = load(BASE, name, store)
        final = load(steps[-1], name, store)
        previous = base
        for folder, step in zip(steps, out["steps"]):
            run = load(folder, name, store)
            out["base"][(name, step)] = compare(base["encoder"], run["encoder"], base["groups"], run["groups"], retrieval=False)
            out["previous"][(name, step)] = compare(previous["encoder"], run["encoder"], previous["groups"], run["groups"], retrieval=False)
            out["final"][(name, step)] = compare(final["encoder"], run["encoder"], final["groups"], run["groups"], retrieval=False)
            out["profile"][(name, step)] = profile(run["encoder"], run["groups"])
            if name.startswith("act."):
                out["base"][(name, step)]["expert"] = expert_side(base, run)
            previous = run
        print(name, flush=True)
    images = sorted({n.split(".", 2)[2].rsplit(".", 1)[0] for n in runs if n.startswith("act.step.")})
    for folder, step in zip([BASE] + steps, [0] + out["steps"]):
        for image in images:
            for a, b in (("right", "left"), ("tubeleft", "tuberight"), ("own", "right")):
                first, second = load(folder, f"act.step.{image}.{a}", store), load(folder, f"act.step.{image}.{b}", store)
                out["instruction"][(step, image, a, b)] = {"lm": compare(first["encoder"], second["encoder"], first["groups"], second["groups"], retrieval=False), "expert": expert_side(first, second)}
        print(step, flush=True)
    return out


def expert(store: str) -> dict:
    """Cross-attention mass by kind of token (mean over heads, action positions and flow steps), per layer; and
    the chunks: other flow seeds, one kind hidden from the action expert."""
    out = {"mass": {}, "mass_per_token": {}, "chunks": {}, "top_position": {}}
    folders = {**MAIN, **{f"tube {int(f.split('__')[1])}": f for f in sorted(os.listdir(store)) if f.startswith("rebot-v2-tube") and f != TUBE}}
    for key, folder in folders.items():
        for name in names(folder, "act.*", store):
            run = load(folder, name, store)
            mass = run["mass_mean"].float().mean(dim=1)  # [L, S]
            index = members(run["groups"])
            out["mass"][(key, name)] = {kind: mass[:, at].sum(dim=1).tolist() for kind, at in index.items() if kind != "all"}
            out["mass_per_token"][(key, name)] = {kind: mass[:, at].mean(dim=1).tolist() for kind, at in index.items() if kind != "all"}
        chunks = torch.load(os.path.join(store, folder, "chunks.pt"), weights_only=False)
        for name, extra in chunks.items():
            seeds = extra["seeds"]
            entry = {"seed_rms": [float((seeds[0] - seeds[s]).square().mean().sqrt()) for s in seeds if s != 0], "chunk": seeds[0], "seeds": seeds}
            if "hidden" in extra:
                entry["hidden_rms"] = {kind: float((seeds[0] - chunk).square().mean().sqrt()) for kind, chunk in extra["hidden"].items()}
                entry["hidden_counts"] = extra["hidden_counts"]
            out["chunks"][(key, name)] = entry
        print(key, flush=True)
    return out


@torch.no_grad()
def stream(store: str) -> dict:
    """Mean cosine between the positions of one kind of the prompt and the action tokens fed back in, per layer."""
    out = {}
    for key, folder in MAIN.items():
        for name in names(folder, "stream.*", store):
            run = torch.load(os.path.join(store, folder, name + ".pt"), weights_only=False)
            prompt_run = load(folder, name.replace("stream.", "act."), store)
            groups = prompt_run["groups"]
            extra = run["encoder"].shape[1] - len(groups)
            if run["generated"] is None or extra < 4:
                continue
            fed = run["generated"][:extra].tolist()
            index = members(groups)
            entry = {"generated": len(run["generated"]), "error": run["error"], "to_action": {}, "action_pair": [], "action_norm": []}
            for layer in run["encoder"]:
                unit = torch.nn.functional.normalize(layer.to(device).float(), dim=-1)
                action = unit[len(groups) + 1 :]  # after <action_start>
                entry["action_pair"].append(float((action.sum(dim=0).square().sum() - len(action)) / (len(action) * (len(action) - 1))))
                entry["action_norm"].append(float(layer[len(groups) + 1 :].float().norm(dim=-1).mean()))
                for kind, at in index.items():
                    entry["to_action"].setdefault(kind, []).append(float((unit[at] @ action.T).mean()))
            out[(key, name)] = entry
        print(key, flush=True)
    return out


def lens(store: str) -> dict:
    out = {}
    folders = {**MAIN, **{f"tube {int(f.split('__')[1])}": f for f in sorted(os.listdir(store)) if f.startswith("rebot-v2-tube") and f != TUBE}}
    for key, folder in folders.items():
        for name in names(folder, "chat.*", store) + names(folder, "act.*", store):
            run = torch.load(os.path.join(store, folder, name + ".pt"), weights_only=False)["lens"]
            out[(key, name)] = {k: v for k, v in run.items() if k != "final_logits"}
        print(key, flush=True)
    return out


if __name__ == "__main__":
    stage = sys.argv[1]
    options = dict(arg.lstrip("-").split("=", 1) for arg in sys.argv[2:])
    store, out = options.get("store", STORE), options.get("out", OUT)
    os.makedirs(out, exist_ok=True)
    result = {"cross": cross, "prompt": prompt, "series": series, "expert": expert, "stream": stream, "lens": lens}[stage](store)
    torch.save(result, os.path.join(out, stage + ".pt"))
    print("wrote", os.path.join(out, stage + ".pt"))
