"""Figures for run_loss_share.py: shares per checkpoint and per-frame distributions.

    uv run python lerobot/adhocs/loss_share/plot_loss_share.py outputs/loss_share/corr10_2026-11-12
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GROUPS = ["internal", "external", "corrections", "diverse"]
COLORS = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

out_dir = sys.argv[1]
groups = pd.concat(pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out_dir, "step_*_groups.csv"))))
samples = pd.concat(pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out_dir, "step_*_samples.csv"))))
steps = sorted(groups.step.unique())
groups.to_csv(os.path.join(out_dir, "groups_all.csv"), index=False)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})


def label_ends(ax, ends):
    """Direct labels at the line ends, pushed apart so they never overlap."""
    low, high = ax.get_ylim()
    gap = (high - low) * 0.055
    placed = []
    for value, text in sorted(ends):
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        ax.annotate(text, (steps[-1], y), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)


panels = [("loss_share", "Share of the summed flow loss"),
          ("grad_norm_sum_share", "Share of the summed per-frame gradient norms"),
          ("projection_share", "Share of the update direction (projection on the total gradient)")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, (column, title) in zip(axes, panels, strict=True):
    ends = []
    for name in GROUPS:
        part = groups[groups.group == name].sort_values("step")
        ax.axhline(part.frame_share.iloc[0], color=COLORS[name], lw=1, ls=(0, (2, 3)), alpha=0.7)
        ax.plot(part.step, part[column], color=COLORS[name], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=name)
        ends.append((part[column].iloc[-1], f"{name} {part[column].iloc[-1]:.2f}"))
    label_ends(ax, ends)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xticks(steps)
    ax.set_xlabel("checkpoint step")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
axes[0].set_ylabel("share")
axes[0].legend(frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0, 1.22))
fig.text(0.995, 0.01, "dotted = the group's share of the frames (training mix)", ha="right", color=MUTED, fontsize=9)
fig.subplots_adjust(left=0.05, right=0.93, top=0.82, bottom=0.14, wspace=0.28)
fig.savefig(os.path.join(out_dir, "shares.png"), dpi=150)

fig, axes = plt.subplots(2, len(steps), figsize=(3.3 * len(steps), 7), sharey="row")
for row, (column, label) in enumerate([("loss", "flow loss per frame"), ("grad_norm", "gradient norm per frame")]):
    for ax, step in zip(axes[row], steps, strict=True):
        part = samples[samples.step == step]
        boxes = ax.boxplot([part[part.coarse == name][column] for name in GROUPS], widths=0.6, patch_artist=True,
                           showfliers=True, flierprops={"marker": ".", "ms": 3, "mfc": MUTED, "mec": "none", "alpha": 0.5},
                           medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch, name in zip(boxes["boxes"], GROUPS, strict=True):
            patch.set(facecolor=COLORS[name], edgecolor=SURFACE, alpha=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 5), GROUPS, rotation=30, ha="right")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        if row == 0:
            ax.set_title(f"step {step}", loc="left", fontsize=10.5)
    axes[row][0].set_ylabel(label + " (log)")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "distributions.png"), dpi=150)

table = groups.pivot(index="group", columns="step").loc[GROUPS]
with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    for column in ["frame_share", "loss_mean", "loss_share", "target_sq_mean", "grad_norm_mean", "grad_norm_sum_share",
                   "group_grad_norm", "group_grad_norm_share", "projection_share", "cos_to_total"]:
        f.write(f"{column}\n{table[column].round(4).to_string()}\n\n")
print(open(os.path.join(out_dir, "summary.txt")).read())
