"""Figures of outputs/analysis/layer_cosine_2026-11-16/REPORT.md, from the files layer_analysis.py, layer_weights.py,
layer_patch.py and layer_vision.py write there. No model, no GPU.

    uv run python lerobot/adhocs/text_probe/layer_report.py
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from layer_analysis import BASE, FACTOR, OUT, load, members

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
COLOR = {"base": "#444444", "tube": "#1f77b4", "internal": "#d62728"}
LAYERS = np.arange(36)
CHAT_IMAGE = lambda name: name.startswith("chat.") and ".none." not in name  # noqa: E731
CHAT_TEXT = lambda name: name.startswith("chat.none")  # noqa: E731
ACTION = lambda name: name.startswith("act.step") and ".none." not in name  # noqa: E731


def read(name: str):
    return torch.load(os.path.join(OUT, name + ".pt"), weights_only=False)


def mean(rows) -> np.ndarray:
    return np.nanmean(np.array(rows, dtype=float), axis=0)


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def lengths(cross: dict) -> None:
    run = load(BASE, "act.step.robot_600.right")
    attended = members(run["groups"])["all"]
    norm = run["encoder"].float().norm(dim=-1)[:, attended].numpy()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.2]})
    for ax, layer in zip(axes, (0, 6, 16, 35)):
        median = np.median(norm[layer])
        big = norm[layer] > FACTOR * median
        ax.scatter(np.arange(len(attended))[~big], norm[layer][~big], s=2, color="#888888")
        ax.scatter(np.arange(len(attended))[big], norm[layer][big], s=10, color="#d62728")
        ax.axhline(median, color="k", lw=0.8)
        ax.axhline(FACTOR * median, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(3, 4e5)
        ax.set_title(f"layer {layer}: median {median:.0f},\n{int(big.sum())} over 10 x the median (red)")
        ax.set_xlabel("position in the prompt")
    axes[0].set_ylabel("L2 norm of the block output")
    names = [name for name in cross["profile"] if ACTION(name)]
    for key in ("base", "tube", "internal"):
        axes[4].plot(LAYERS, mean([cross["profile"][name][key]["massive"]["all"] for name in names]), color=COLOR[key], label=key)
    axes[4].set_title(f"positions with norm over 10 x the median,\nper prompt ({len(names)} action prompts)")
    axes[4].set_xlabel("layer")
    axes[4].legend()
    save(fig, "fig1_lengths.png")


def cross_checkpoint(cross: dict) -> None:
    panels = [("image tokens, action prompts", ACTION, "image"), ("question tokens, action prompts", ACTION, "question"), ("text tokens, prompts without image", CHAT_TEXT, "text")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for row, pair in enumerate((("base", "tube"), ("base", "internal"))):
        for ax, (title, select, kind) in zip(axes[row], panels):
            items = [v for (name, a, b), v in cross["pairs"].items() if (a, b) == pair and select(name) and kind in v["cos"]]
            ax.plot(LAYERS, mean([v["cos"][kind] for v in items]), color=COLOR[pair[1]], label="same position (high-norm positions left out)")
            ax.plot(LAYERS, mean([v["null_cos"][kind] for v in items]), color=COLOR[pair[1]], ls="--", label="another position of the kind")
            ax.plot(LAYERS, mean([v["raw_cos"][kind] for v in items]), color="#999999", lw=0.9, label="same position, all positions")
            ax.set_title(f"{pair[0]} vs {pair[1]}: {title}\n(n={len(items)} prompts)")
            ax.axvline(16, color="k", lw=0.5, ls=":")
    axes[0][0].legend()
    for ax in axes[1]:
        ax.set_xlabel("layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("raw cosine")
    save(fig, "fig2_cross_checkpoint.png")


def same_checkpoint(prompt: dict) -> None:
    panels = [("action frame, other scene", "question"), ("action frame, other scene", "last"), ("action frame, other scene", "subtask"), ("chat text only", "text")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for ax, (family, kind) in zip(axes, panels):
        for key in ("base", "tube", "internal"):
            items = [v["lm"] for (c, a, b), v in prompt.items() if c == key and v["family"] == family and kind in v["lm"]["cos"]]
            ax.plot(LAYERS, mean([v["cos"][kind] for v in items]), color=COLOR[key], label=key)
            if kind != "last":
                ax.plot(LAYERS, mean([v["null_cos"][kind] for v in items]), color=COLOR[key], ls="--", lw=0.8)
        ax.set_title(f"{family}:\n{kind} tokens (n={len(items)} pairs)")
        ax.set_xlabel("layer")
    axes[0].set_ylabel("raw cosine, two inputs, one checkpoint")
    axes[0].legend()
    save(fig, "fig3_same_checkpoint.png")


def series(data: dict) -> None:
    steps = data["steps"]
    names = sorted({key[0] for key in data["base"]})
    panels = [("image tokens (chat prompts)", CHAT_IMAGE, "image"), ("question tokens (action prompts)", ACTION, "question"), ("text tokens (prompts without image)", CHAT_TEXT, "text")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for column, (title, select, kind) in enumerate(panels):
        chosen = [name for name in names if select(name)]
        for row, which in enumerate(("base", "previous")):
            for layer, shade in zip((0, 10, 22, 34), ("#9ecae1", "#4292c6", "#08519c", "#08306b")):
                axes[row][column].plot(steps, [mean([data[which][(name, step)]["cos"][kind] for name in chosen])[layer] for step in steps], marker="o", ms=3, color=shade, label=f"layer {layer}")
            axes[row][column].set_title(f"{title}, \nstep vs {"base" if which == "base" else "the step before"} (n={len(chosen)} prompts)")
    axes[0][0].legend()
    for ax in axes[1]:
        ax.set_xlabel("training step of the tube run")
    for ax in axes[:, 0]:
        ax.set_ylabel("raw cosine (high-norm positions left out)")
    save(fig, "fig4_series.png")


def instruction(prompt: dict, expert: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    family = "instruction tubeleft / tuberight"
    for ax, key in zip(axes, ("base", "tube", "internal")):
        items = [v["lm"] for (c, a, b), v in prompt.items() if c == key and v["family"] == family]
        for kind, shade in (("subtask", "#2ca02c"), ("state", "#ff7f0e"), ("question", "#9467bd"), ("last", "#000000"), ("image", "#aaaaaa")):
            ax.plot(LAYERS, mean([v["cos"][kind] for v in items if kind in v["cos"]]), color=shade, label=kind)
        ax.set_title(f"{key}: tube on the left vs on the right\n(n={len(items)} frames)")
        ax.set_ylim(0.8, 1.005)
        ax.set_xlabel("layer of the language model")
    axes[0].set_ylabel("raw cosine between the two prompts")
    axes[0].legend()
    labels, values = ["other flow seed"], {}
    for key in ("base", "tube", "internal"):
        values[key] = [np.mean([x for (c, n), v in expert["chunks"].items() if c == key for x in v["seed_rms"]])]
    for fam, short in (("instruction right / left", "right / left"), ("instruction cup / bottle", "cup / bottle"), ("instruction tubeleft / tuberight", "tube left / right")):
        labels.append(short)
        for key in values:
            values[key].append(np.mean([v["expert"]["chunk_rms"] for (c, a, b), v in prompt.items() if c == key and v["family"] == fam]))
    for kind in ("subtask", "state", "image"):
        labels.append(f"{kind} hidden")
        for key in values:
            values[key].append(np.mean([v["hidden_rms"][kind] for (c, n), v in expert["chunks"].items() if c == key and "hidden_rms" in v and kind in v["hidden_rms"]]))
    width = 0.27
    for offset, key in enumerate(values):
        axes[3].bar(np.arange(len(labels)) + (offset - 1) * width, values[key], width, color=COLOR[key], label=key)
    axes[3].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axes[3].set_ylabel("chunk change, RMS (degrees)")
    axes[3].set_title("what moves the predicted chunk")
    axes[3].legend()
    save(fig, "fig5_instruction.png")


def flat(chunk: torch.Tensor) -> np.ndarray:
    return chunk.reshape(-1, chunk.shape[-1]).float().cpu().numpy()


def patch(data: dict) -> None:
    checkpoints = list(dict.fromkeys(key[0] for key in data))
    short = {name: "base" if name == "base" else "tube" if "tube" in name else "internal" for name in checkpoints}
    sets = ["subtask", "state", "velocity", "torque + tip", "question", "tail", "all after the instruction", "all"]
    bands = [key for key in next(iter(data.values()))["patch"] if key.startswith("layers")]
    pairs = sorted({key[2:4] for key in data})
    fig, axes = plt.subplots(len(pairs), 3, figsize=(15, 3.4 * len(pairs)))
    for row, pair in enumerate(pairs):
        for offset, name in enumerate(checkpoints):
            items = [v for key, v in data.items() if key[0] == name and key[2:4] == pair]
            share = {}
            for key in sets + bands:
                got = []
                for v in items:
                    if key in v["patch"]:
                        a, b, p = flat(v["a"]), flat(v["b"]), flat(v["patch"][key])
                        got.append(((p - a) * (b - a)).sum() / max(((b - a) ** 2).sum(), 1e-12))
                share[key] = np.median(got) if got else np.nan
            color = COLOR[short[name]]
            axes[row][0].bar(np.arange(len(sets)) + (offset - 1) * 0.27, [share[k] for k in sets], 0.27, color=color, label=f"{short[name]} (n={len(items)})")
            axes[row][1].bar(np.arange(len(bands)) + (offset - 1) * 0.27, [share[k] for k in bands], 0.27, color=color)
            hides = list(items[0]["hide"])
            axes[row][2].bar(np.arange(len(hides)) + (offset - 1) * 0.27, [np.mean([np.sqrt(((flat(v["a"]) - flat(v["hide"][k])) ** 2).mean()) for v in items]) for k in hides], 0.27, color=color)
        axes[row][0].set_xticks(np.arange(len(sets)), sets, rotation=35, ha="right")
        axes[row][1].set_xticks(np.arange(len(bands)), bands, rotation=35, ha="right")
        axes[row][2].set_xticks(np.arange(len(hides)), [h.replace("massive", "high-norm positions") for h in hides], rotation=35, ha="right")  # "massive" is the saved key
        axes[row][0].set_title(f"{pair[0]} -> {pair[1]}: positions patched, all layers")
        axes[row][1].set_title("all positions patched, one band of layers")
        axes[row][2].set_title("positions hidden from the action expert")
        axes[row][0].set_ylabel("share of the chunk change (median)")
        axes[row][2].set_ylabel("chunk change, RMS (degrees)")
    axes[0][0].legend()
    fig.tight_layout()
    save(fig, "fig6_patch.png")


def collapse(cross: dict, stream: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for ax, (title, select) in zip(axes[:3], (("action prompts", ACTION), ("chat prompts with image", CHAT_IMAGE), ("prompts without image", CHAT_TEXT))):
        names = [name for name in cross["profile"] if select(name)]
        for key in ("base", "tube", "internal"):
            ax.plot(LAYERS, mean([cross["profile"][name][key]["pair"]["all"] for name in names]), color=COLOR[key], label=key)
        ax.set_title(f"two different positions,\n{title} (n={len(names)})")
        ax.set_xlabel("layer")
    axes[0].set_ylabel("mean raw cosine")
    axes[0].legend()
    for key in ("base", "tube", "internal"):
        items = [v for (c, n), v in stream.items() if c == key]
        axes[3].plot(LAYERS, mean([v["to_action"]["subtask"] for v in items]), color=COLOR[key], label=f"{key}: instruction")
        axes[3].plot(LAYERS, mean([v["to_action"]["image"] for v in items]), color=COLOR[key], ls="--", lw=0.8, label=f"{key}: image")
    axes[3].set_title(f"prompt position vs generated action tokens\n(n={len(items)} frames)")
    axes[3].set_xlabel("layer")
    axes[3].legend(fontsize=7)
    save(fig, "fig7_collapse.png")


def weights(data: dict) -> None:
    groups = [("language model block", "model.transformer.blocks.{}", 36), ("action expert block", "model.action_expert.blocks.{}", 36), ("vision transformer block", "model.vision_backbone.image_vit.transformer.resblocks.{}", 27)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, (title, pattern, count) in zip(axes, groups):
        for name, shade in (("tube_100", "#9ecae1"), ("tube_600", "#4292c6"), ("tube_1200", COLOR["tube"]), ("internal_1200", COLOR["internal"])):
            parts = data[name]["parts"]
            ax.plot([i for i in range(count) if pattern.format(i) in parts], [100 * parts[pattern.format(i)] for i in range(count) if pattern.format(i) in parts], color=shade, label=name.replace("_", " step "))
        ax.set_title(title)
        ax.set_xlabel("block")
    axes[0].set_ylabel("|finetuned - base| / |base|, %")
    axes[0].legend()
    save(fig, "fig8_weights.png")


def vision(data: dict) -> None:
    checkpoints = list(dict.fromkeys(key[0] for key in data))
    stages = list(dict.fromkeys(key[2] for key in data))
    images = list(dict.fromkeys(key[1] for key in data))
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
    for name in checkpoints:
        step = name.split("__")[-1].lstrip("0")
        color = COLOR["internal"] if "internal" in name else {"100": "#9ecae1", "300": "#4292c6"}.get(step, COLOR["tube"])
        tag = ("internal " if "internal" in name else "tube ") + step
        axes[0].plot([np.mean([data[(name, image, stage)]["cos"] for image in images]) for stage in stages], color=color, marker="o", ms=3, label=tag)
        axes[1].plot([np.mean([data[(name, image, stage)]["rel"] for image in images]) for stage in stages], color=color, marker="o", ms=3, label=tag)
    for ax, title in zip(axes, ("raw cosine to base, same token", "relative L2 distance to base")):
        ax.set_xticks(range(len(stages)), [s.replace("vit ", "") for s in stages], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{title} ({len(images)} images)")
        ax.set_xlabel("vision transformer block, then pooled, then projected")
    axes[0].legend()
    save(fig, "fig9_vision.png")


if __name__ == "__main__":
    only = sys.argv[1:]
    cross, prompt = read("cross"), read("prompt")
    jobs = {
        "lengths": lambda: lengths(cross), "cross": lambda: cross_checkpoint(cross), "same": lambda: same_checkpoint(prompt),
        "series": lambda: series(read("series")), "instruction": lambda: instruction(prompt, read("expert")),
        "patch": lambda: patch(read("patch")), "collapse": lambda: collapse(cross, read("stream")),
        "weights": lambda: weights(read("weights")), "vision": lambda: vision(read("vision")),
    }
    for name, job in jobs.items():
        if not only or name in only:
            job()
