"""Figures for run_grad_drift.py: drift of the probe gradient per data group, per block and per pair of steps.

    uv run python lerobot/adhocs/grad_drift/plot_grad_drift.py outputs/grad_drift/base_2026-11-12
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"internal": "#2a78d6", "external": "#eb6834", "corrections": "#1baf7a", "diverse": "#eda100"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

out_dir = sys.argv[1]
run = json.load(open(os.path.join(out_dir, "run.json")))
groups, blocks = run["groups"], run["blocks"]
paths = sorted(glob.glob(os.path.join(out_dir, "step_*.npz")))
steps = [int(os.path.basename(p)[5:11]) for p in paths]
files = [np.load(p) for p in paths]
grad = np.stack([f["grad"] for f in files]).astype(np.float64)  # [step, group, block, dim]
grad_sq = np.stack([f["grad_sq"] for f in files])  # [step, group, block]
disp = np.stack([f["disp"] for f in files]).astype(np.float64)  # [step, block, dim]
disp_sq = np.stack([f["disp_sq"] for f in files])  # [step, block]
floor = np.load(os.path.join(out_dir, "floor.npz"))
losses = pd.read_csv(os.path.join(out_dir, "losses.csv")).groupby(["step", "coarse"]).loss.mean().unstack()[groups]

norm = np.sqrt(grad_sq.sum(2))  # [step, group]
# Cosines use sketched dot products over exact norms.
cos_start = np.einsum("sgbd,gbd->sg", grad, grad[0]) / (norm * norm[0])
cos_floor = np.einsum("gbd,gbd->g", floor["grad"].astype(np.float64), grad[0]) / (np.sqrt(floor["grad_sq"].sum(1)) * norm[0])
cos_pairs = np.einsum("sgbd,tgbd->gst", grad, grad) / np.einsum("sg,tg->gst", norm, norm)
with np.errstate(invalid="ignore"):  # a block without gradient for a group stays blank
    cos_block = np.einsum("sgbd,gbd->gbs", grad, grad[0]) / np.sqrt(grad_sq * grad_sq[0]).transpose(1, 2, 0)
predicted = np.einsum("gbd,sbd->sg", grad[0], disp)  # first-order loss change from the start gradient
moved = np.diff(disp, axis=0)  # [step - 1, block, dim], the movement between two measurements
moved_norm = np.sqrt(np.einsum("sbd,sbd->s", moved, moved))
cos_descent = -np.einsum("sgbd,sbd->sg", grad[:-1], moved) / (norm[:-1] * moved_norm[:, None])

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.minorticks_off()
    ax.set_xlabel("training steps from the start")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
for g, name in enumerate(groups):
    line = {"color": COLORS[name], "lw": 2, "marker": "o", "ms": 5, "mec": SURFACE, "mew": 1.2}
    axes[0, 0].plot(steps, cos_start[:, g], label=name, **line)
    axes[0, 0].axhline(cos_floor[g], color=COLORS[name], lw=1, ls=(0, (2, 3)), alpha=0.7)
    axes[0, 1].plot(steps, norm[:, g] / norm[0, g], **line)
    axes[1, 0].plot(steps, losses[name].to_numpy() - losses[name].iloc[0], **line)
    axes[1, 0].plot(steps, predicted[:, g], color=COLORS[name], lw=1.2, ls=(0, (4, 2)))
    axes[1, 1].plot(steps[1:], cos_descent[:, g], **line)
style(axes[0, 0], "Cosine of the probe gradient to its value at the start")
style(axes[0, 1], "Probe gradient norm relative to the start")
style(axes[1, 0], "Probe loss change: measured (solid), first order from the start gradient (dashed)")
style(axes[1, 1], "Cosine of the movement since the last measurement to the descent direction")
axes[1, 1].axhline(0, color=MUTED, lw=0.8)
axes[0, 0].legend(frameon=False, ncols=len(groups), loc="upper left", bbox_to_anchor=(0, 1.2))
fig.text(0.995, 0.005, "dotted = same weights, other flow noise (floor)", ha="right", color=MUTED, fontsize=9)
fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08, wspace=0.16, hspace=0.36)
fig.savefig(os.path.join(out_dir, "drift.png"), dpi=150)

fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 1.4, 0.13 * len(blocks) + 1.6), sharey=True)
for ax, name, values in zip(axes, groups, cos_block, strict=True):
    image = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title(name, loc="left", fontsize=10.5)
    ax.set_xticks(range(len(steps)), [str(s) for s in steps], rotation=90)
    ax.set_xlabel("training steps")
axes[0].set_yticks(range(len(blocks)), blocks, fontsize=6)
fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="cosine to the block's gradient at the start")
fig.savefig(os.path.join(out_dir, "blocks.png"), dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 1.4, 4.2), sharey=True)
for ax, name, values in zip(axes, groups, cos_pairs, strict=True):
    image = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(name, loc="left", fontsize=10.5)
    ax.set_xticks(range(len(steps)), [str(s) for s in steps], rotation=90)
    ax.set_yticks(range(len(steps)), [str(s) for s in steps])
    ax.set_xlabel("training steps")
fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="cosine between the gradients at two steps")
fig.savefig(os.path.join(out_dir, "pairs.png"), dpi=150, bbox_inches="tight")

with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    for title, values in [("cosine to the start gradient", cos_start), ("gradient norm", norm),
                          ("probe loss", losses.to_numpy()), ("first-order loss change from the start gradient", predicted)]:
        f.write(f"{title}\n{pd.DataFrame(values, index=steps, columns=groups).round(4).to_string()}\n\n")
    f.write(f"cosine to the descent direction of the movement since the last measurement\n"
            f"{pd.DataFrame(cos_descent, index=steps[1:], columns=groups).round(4).to_string()}\n\n")
    f.write(f"floor (same weights, other flow noise)\n{pd.Series(cos_floor, index=groups).round(4).to_string()}\n\n")
    f.write(f"displacement norm from the start\n{pd.Series(np.sqrt(disp_sq.sum(1)), index=steps).round(4).to_string()}\n\n")
    ratio = np.einsum("sgbd,sgbd->sgb", grad, grad)[grad_sq > 0] / grad_sq[grad_sq > 0]
    f.write(f"sketch check, sketched / exact squared norm per (step, group, block): "
            f"median {np.median(ratio):.3f}, 5-95% {np.quantile(ratio, 0.05):.3f}-{np.quantile(ratio, 0.95):.3f}\n")
print(open(os.path.join(out_dir, "summary.txt")).read())
