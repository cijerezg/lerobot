"""Tables and figures of ``vision_depth.py`` (no model).

    uv run python lerobot/adhocs/text_probe/vision_depth_report.py outputs/analysis/vision_depth_2026-11-18/depth2.pt

Output: the tip path of every chunk (filtered as deployed, forward kinematics, cm) against the reference chunk of
the same flow seed, next to four scales: two flow seeds of the reference, how far the reference chunk takes the
tip from the current pose, the reference against the recorded chunk, and the conditions that change the camera
images. Cosine of the output: between the tip paths taken from the current tip position, and between the chunks
in the normalized action space.

Layers: per condition and layer the stats ``vision_depth.py`` stores (median over the positions of a kind within
a frame), here the mean over frames. Writes ``<name>.md``, ``<name>_output.png``, ``<name>_layers.png`` and
``<name>_lengths.png`` next to the file.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
COLORS = {"base": "#d62728", "random": "#7f7f7f", "shuffled": "#bcbd22", "next": "#1f77b4", "other": "#17becf"}
NAMES = {"base": "base vision path", "random": "random direction, same size", "shuffled": "same change, shuffled over patches", "next": "camera images 1 s later", "other": "camera images of another episode"}
ROWS = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36]  # of hidden: 0 = the input embedding, n = output of block n
STEPS = [1, 5, 10, 30]  # 5 = the execution horizon of the deployed config


def table(title: str, columns: list[str], rows: list[tuple]) -> str:
    lines = [f"**{title}**", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """over the last two axes"""
    a, b = a.reshape(*a.shape[:-2], -1), b.reshape(*b.shape[:-2], -1)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def output(data: dict, folder: str, name: str) -> list[str]:
    conditions, frames = data["conditions"], data["frames"]
    joints, cells = len(data["joints"]), ["reference", *data["conditions"]]
    raw = np.stack([np.stack([f["chunks"][cell].double().numpy() for f in frames]) for cell in cells])[..., :joints]  # [cells, frames, seeds, steps, joints]
    norm = np.stack([np.stack([f["chunks_norm"][cell].double().numpy() for f in frames]) for cell in cells])[..., :joints]
    smooth = np.moveaxis(apply_butterworth_filter(np.moveaxis(raw, 3, 0)), 0, 3)
    kinematics = RebotKinematics()

    def tip(q):
        return kinematics.ee_path(q.reshape(-1, joints)).reshape(*q.shape[:-1], 3) * 100

    tips, now, recorded = tip(smooth), tip(data["state"].double().numpy()[:, :joints]), tip(data["recorded"].double().numpy()[..., :joints])
    seeds, count = tips.shape[2], len(frames)
    pairs = ~np.eye(seeds, dtype=bool)
    shift = np.linalg.norm(tips[1:] - tips[:1], axis=-1)  # [conditions, frames, seeds, steps]
    between = np.linalg.norm(tips[0][:, :, None] - tips[0][:, None], axis=-1)[:, pairs]  # [frames, pairs, steps]
    travel = np.linalg.norm(tips[0] - now[:, None, None], axis=-1)  # [frames, seeds, steps]
    to_recorded = np.linalg.norm(tips - recorded[None, :, None], axis=-1)  # [cells, frames, seeds, steps]
    paths = tips - now[None, :, None, None]  # tip path from the current tip position
    # per frame: [frames, steps] and the mean over the chunk [frames]
    curves = {"two flow seeds of the reference": between.mean(axis=1), **{c: shift[i].mean(axis=1) for i, c in enumerate(conditions)}}
    per_frame = {key: curve.mean(axis=1) for key, curve in curves.items()}
    variation = per_frame["two flow seeds of the reference"]

    out = []
    rows = [(NAMES.get(key, key), *[float(curve[:, s - 1].mean()) for s in STEPS], float(per_frame[key].mean()), float(np.median(per_frame[key])), float(np.percentile(per_frame[key], 90)),
             "" if key not in conditions else f"{int((per_frame[key] > variation).sum())} of {count}") for key, curve in curves.items()]
    rows.append(("reference: tip distance from the current pose", *[float(travel[..., s - 1].mean()) for s in STEPS], float(travel.mean()), float(np.median(travel.mean(axis=(1, 2)))), float(np.percentile(travel.mean(axis=(1, 2)), 90)), ""))
    rows.append(("reference against the recorded chunk", *[float(to_recorded[0][..., s - 1].mean()) for s in STEPS], float(to_recorded[0].mean()), float(np.median(to_recorded[0].mean(axis=(1, 2)))), float(np.percentile(to_recorded[0].mean(axis=(1, 2)), 90)), ""))
    out.append(table("tip distance to the reference chunk of the same flow seed (cm); mean over frames and seeds unless said", ["", *[f"step {s}" for s in STEPS], "mean over the chunk", "median frame", "90th percentile frame", "frames over their own seed variation"], rows))

    # the part of the shift that is the same in every flow seed, and the cosines
    rows = []
    reference_cos = np.stack([cosine(paths[0][:, i], paths[0][:, j]) for i in range(seeds) for j in range(seeds) if i != j], axis=1)  # [frames, pairs]
    reference_norm_cos = np.stack([cosine(norm[0][:, i], norm[0][:, j]) for i in range(seeds) for j in range(seeds) if i != j], axis=1)
    reference_rmse = np.sqrt(((raw[0][:, :, None] - raw[0][:, None]) ** 2)[..., :-1].mean(axis=(-1, -2)))[:, pairs]
    reference_gripper = np.abs(raw[0][:, :, None, :, -1] - raw[0][:, None, :, :, -1]).mean(axis=-1)[:, pairs]
    motion = tips - tips[..., :1, :]  # tip path from the chunk's own first step
    reference_motion_cos = np.stack([cosine(motion[0][:, i], motion[0][:, j]) for i in range(seeds) for j in range(seeds) if i != j], axis=1)
    rows.append(("two flow seeds of the reference", "", float(reference_cos.mean()), float(np.median(reference_cos.mean(axis=1))), float(reference_motion_cos.mean()), float(reference_norm_cos.mean()), float(reference_rmse.mean()), float(reference_gripper.mean())))
    cosines = {}
    for i, c in enumerate(conditions, start=1):
        same = np.linalg.norm((tips[i] - tips[0]).mean(axis=1), axis=-1).mean(axis=-1)  # [frames]
        cosines[c] = cosine(paths[i], paths[0])  # [frames, seeds]
        rows.append((NAMES[c], float(same.mean()), float(cosines[c].mean()), float(np.median(cosines[c].mean(axis=1))), float(cosine(motion[i], motion[0]).mean()), float(cosine(norm[i], norm[0]).mean()),
                     float(np.sqrt(((raw[i] - raw[0]) ** 2)[..., :-1].mean(axis=(-1, -2))).mean()), float(np.abs(raw[i] - raw[0])[..., -1].mean())))
    out.append(table("against the reference chunk of the same flow seed", ["", "tip shift of the mean over seeds (cm)", "cosine of the tip paths from the current tip, mean", "same, median frame", "cosine of the tip paths from the chunk's own first step",
                                                                         "cosine of the chunks, normalized action space", "arm joints RMSE (deg)", "gripper, mean absolute difference (deg)"], rows))
    rows = [(NAMES.get(cell, cell), float(to_recorded[i].mean()), float(to_recorded[i][..., -1].mean()),
             "" if i == 0 else f"{int((to_recorded[i].mean(axis=(1, 2)) < to_recorded[0].mean(axis=(1, 2))).sum())} of {count}") for i, cell in enumerate(cells)]
    out.append(table("tip distance to the recorded chunk (cm)", ["", "mean over the chunk", "last step", "frames closer to the recording than the reference"], rows))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    steps = np.arange(1, tips.shape[3] + 1)
    axes[0].plot(steps, travel.mean(axis=(0, 1)), color="#c7c7c7", lw=1.4, ls=":", label="reference: distance from the current pose")
    axes[0].plot(steps, curves["two flow seeds of the reference"].mean(axis=0), color="black", lw=1.6, ls="--", label="two flow seeds of the reference")
    for c in conditions:
        axes[0].plot(steps, curves[c].mean(axis=0), color=COLORS[c], lw=1.6, label=NAMES[c])
    axes[0].axvline(5, color="#c7c7c7", lw=0.8)
    axes[0].set(xlabel="step of the chunk (30 per second; 5 = executed before the next chunk)", ylabel="tip distance to the reference chunk (cm)", title="along the chunk, mean over frames and seeds", ylim=(0, None))
    axes[0].legend(fontsize=7)
    for key, values in per_frame.items():
        axes[1].plot(np.sort(values), (np.arange(count) + 1) / count, color=COLORS.get(key, "black"), ls="-" if key in COLORS else "--", lw=1.6)
    axes[1].set(xlabel="tip distance to the reference chunk, mean over the chunk (cm)", ylabel="share of frames", title="over the frames", xscale="log", ylim=(0, 1))
    values = reference_cos.mean(axis=1)
    axes[2].plot(np.sort(values), (np.arange(count) + 1) / count, color="black", ls="--", lw=1.6)
    for c in conditions:
        axes[2].plot(np.sort(cosines[c].mean(axis=1)), (np.arange(count) + 1) / count, color=COLORS[c], lw=1.6)
    axes[2].set(xlabel="cosine of the tip paths from the current tip position", ylabel="share of frames", title="direction of the motion, over the frames", ylim=(0, 1), xlim=(-0.25, 1.01))
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}_output.png"))
    plt.close(fig)
    return out


def layers(data: dict, folder: str, name: str) -> list[str]:
    conditions, frames, stats = data["conditions"], data["frames"], data["stats"]
    curves = {}  # (quantity, kind, condition) -> [stats, layers], mean over frames
    for condition in conditions:
        for quantity, per_kind in frames[0]["layers"][condition].items():
            for kind in per_kind:
                curves[(quantity, kind, condition)] = torch.stack([f["layers"][condition][quantity][kind] for f in frames if kind in f["layers"][condition][quantity]]).mean(dim=0).numpy()
    at = {s: i for i, s in enumerate(stats)}
    panels = [("hidden", "image", "language model block output,\nimage positions"), ("hidden", "text", "language model block output,\ntext positions"), ("values", "text", "values the expert reads,\ntext positions"),
              ("cross", "all", "expert, cross-attention output"), ("expert", "all", "expert, block output")]
    measures = [("difference / length", "|x - a| / |a|"), ("difference / spread", "|x - a| / |a - mean|"), ("cosine, mean removed", "cosine(x - mean, a - mean)")]

    out = []
    for quantity, kind, title in [("hidden", "image", ""), ("hidden", "depth", ""), ("hidden", "text", ""), ("values", "image", ""), ("values", "text", ""), ("keys", "text", ""), ("cross", "all", ""), ("expert", "all", "")]:
        title = {"hidden": "language model block output", "values": "values the expert reads", "keys": "keys the expert reads", "cross": "expert cross-attention output", "expert": "expert block output"}[quantity] + ("" if kind == "all" else f", {kind} positions")
        picked = ROWS if quantity == "hidden" else [r - 1 for r in ROWS[1:]]
        label = (lambda r: "input" if r == 0 else f"{r}") if quantity == "hidden" else (lambda r: f"{r + 1}")
        columns = ["layer", "length", "spread"] + [f"{c}: {m}" for c in conditions for m in ("d / spread", "cos", "cos, mean removed")]
        rows = [(label(r), float(curves[(quantity, kind, conditions[0])][at["length"], r]), float(curves[(quantity, kind, conditions[0])][at["spread"], r]),
                 *[float(curves[(quantity, kind, c)][at[m], r]) for c in conditions for m in ("difference / spread", "cosine", "cosine, mean removed")]) for r in picked]
        out.append(table(title, columns, rows))

    fig, axes = plt.subplots(len(measures), len(panels), figsize=(3.3 * len(panels), 2.6 * len(measures)), sharex="col")
    for column, (quantity, kind, title) in enumerate(panels):
        for row, (measure, formula) in enumerate(measures):
            ax = axes[row, column]
            for condition in conditions:
                curve = curves[(quantity, kind, condition)][at[measure]]
                ax.plot(np.arange(len(curve)) + (0 if quantity == "hidden" else 1), curve, color=COLORS[condition], label=NAMES[condition], lw=1.5)
            if measure.startswith("cosine"):
                ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
            else:
                ax.set_ylim(0, min(ax.get_ylim()[1], 1.5))  # the shuffled change goes far over this in the first expert blocks
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(formula)
            if row == len(measures) - 1:
                ax.set_xlabel("expert block" if quantity in ("cross", "expert") else "language model layer" + (" (0 = input)" if quantity == "hidden" else ""))
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}_layers.png"))
    plt.close(fig)

    # the difference against the two lengths it can be divided by, one point per expert block
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row, (quantity, title) in enumerate([("cross", "expert, cross-attention output"), ("expert", "expert, block output")]):
        for column, (denominator, formula) in enumerate([("length", "length of the reference |a|"), ("spread", "spread of the reference over frames |a - mean|")]):
            ax = axes[row, column]
            for condition in ["base", "other"]:
                x, y = curves[(quantity, "all", condition)][at[denominator]], curves[(quantity, "all", condition)][at["difference"]]
                ax.plot(x, y, color=COLORS[condition], lw=0.6, alpha=0.5)
                ax.scatter(x, y, color=COLORS[condition], s=14, label=NAMES[condition], zorder=3)
                if condition == "base":
                    for block in [0, 4, 9, 14, 19, 24, 29, 33, 34, 35]:
                        ax.annotate(str(block + 1), (x[block], y[block]), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set(xscale="log", yscale="log", xlabel=formula, ylabel="difference to the reference |x - a|", title=title)
            (low, high), limits = ax.get_xlim(), ax.get_ylim()
            for ratio in [0.01, 0.1, 1.0]:
                ax.plot([low, high], [low * ratio, high * ratio], color="#c7c7c7", lw=0.8, zorder=0)
                if limits[0] < high * ratio < limits[1]:
                    ax.annotate(f"ratio {ratio:g}", (high, high * ratio), fontsize=6, color="#7f7f7f", ha="right", va="bottom")
            ax.set(xlim=(low, high), ylim=limits)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}_lengths.png"))
    plt.close(fig)
    return out


def main(path: str) -> None:
    data = torch.load(path, weights_only=False)
    folder, name = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
    out = [f"# Vision path across depth\n\n{data['model']}, image tokens of the vision path of {data['vision']}; {len(data['frames'])} frames of {data['root']}, {data['seeds']} flow seeds.\n",
           "## Output\n", *output(data, folder, name), "## Layers\n\nlength = |a|, spread = |a - mean| (mean of the reference over frames), d = |x - a|; median over positions, mean over frames.\n", *layers(data, folder, name)]
    with open(os.path.join(folder, f"{name}.md"), "w") as file:
        file.write("\n".join(out))
    print("\n".join(out))
    print("wrote", *[os.path.join(folder, f"{name}{end}") for end in (".md", "_output.png", "_layers.png", "_lengths.png")])


if __name__ == "__main__":
    main(sys.argv[1])
