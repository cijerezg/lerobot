"""Compare draw-spread runs (checkpoints): the spread-vs-error lines, the error/spread calibration ratio, and the
headline medians, train vs val, one color per run. Absolute units (normalized action space).

    uv run python lerobot/adhocs/draw_spread/compare_runs.py <out_dir_for_comparison> <label>=<run_dir> [<label>=<run_dir> ...]
e.g.
    uv run python lerobot/adhocs/draw_spread/compare_runs.py outputs/draw_spread/compare_2026-11-07 \\
        internal1200=outputs/draw_spread/internal1200_2026-11-07 mix40_1200=outputs/draw_spread/mix40_1026_1200_2026-11-07 \\
        mix40_1800=outputs/draw_spread/mix40_resume600_2026-11-07
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
runs = [arg.split("=", 1) for arg in sys.argv[2:]]
os.makedirs(OUT, exist_ok=True)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
LINESTYLE = {"train": "-", "val": "--"}

frames = []
for label, d in runs:
    e = pd.read_csv(f"{d}/spread_vs_error.csv") if os.path.exists(f"{d}/spread_vs_error.csv") else None
    a = pd.read_csv(f"{d}/anchors.csv"); a = a[a.setting == "clean"].copy()
    a["spread"] = np.sqrt(a.spread_total); a["bias"] = np.sqrt(a.bias2_total); a["demo"] = np.sqrt(a.hold_total)
    a["ratio"] = a.bias / a.spread
    a["verb"] = a.subtask.fillna("").str.split().str[0].where(lambda s: s.isin(["grasp", "move", "release"]), "other")
    a["run"] = label
    frames.append(a)
df = pd.concat(frames, ignore_index=True)

# table: medians per run x set
rows = []
for (label, s), g in df.groupby(["run", "set"], sort=False):
    rows.append({"run": label, "set": s, "anchors": len(g), "rms spread": g.spread.median(), "rms error of mean": g.bias.median(),
                 "error/spread (median of ratio)": g.ratio.median(), "error/spread q25": g.ratio.quantile(0.25), "error/spread q75": g.ratio.quantile(0.75),
                 "R (mse/hold)": g.R_total.median(), "spread std / data std": np.sqrt(g.spread_ratio_total).median(),
                 "style proj offset": g.style_proj_offset.median(), "corr(log spread, log error)": np.corrcoef(np.log(g.spread), np.log(g.bias))[0, 1]})
tab = pd.DataFrame(rows)
tab.to_csv(f"{OUT}/summary.csv", index=False)
verb = df.groupby(["run", "set", "verb"], sort=False).ratio.median().unstack("verb").round(2)
verb.to_csv(f"{OUT}/ratio_by_verb.csv")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(tab.round(3).to_string(index=False)); print("\nerror/spread by verb\n", verb)

# figure 1: spread vs error lines per run, train solid / val dashed
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.28})
for i, (label, _) in enumerate(runs):
    for s in ("train", "val"):
        g = df[(df.run == label) & (df.set == s)]
        bins = np.quantile(g.spread, np.linspace(0, 1, 8))
        idx = np.clip(np.searchsorted(bins, g.spread, side="right") - 1, 0, len(bins) - 2)
        med = g.groupby(idx).agg(x=("spread", "median"), y=("bias", "median"), r=("ratio", "median"))
        axes[0].plot(med.x, med.y, color=PALETTE[i], ls=LINESTYLE[s], lw=2, marker="o", ms=4, markeredgecolor="white", label=f"{label} {s}")
        axes[1].plot(med.x, med.r, color=PALETTE[i], ls=LINESTYLE[s], lw=2, marker="o", ms=4, markeredgecolor="white", label=f"{label} {s}")
lo, hi = df.spread.quantile(0.01), df.spread.quantile(0.99)
axes[0].plot([lo, hi], [lo, hi], color="#898781", lw=0.8, ls=":")
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("rms spread of the 64 draws (binned medians)"); axes[0].set_ylabel("rms error of the draw mean")
axes[0].set_title("error vs spread, train solid / val dashed", loc="left")
axes[1].set_xscale("log"); axes[1].axhline(1.0, color="#898781", lw=0.8, ls=":"); axes[1].set_xlabel("rms spread of the 64 draws (binned medians)"); axes[1].set_ylabel("error / spread")
axes[1].set_title("calibration ratio along the spread axis", loc="left")
for ax in axes:
    ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
fig.savefig(f"{OUT}/compare_spread_vs_error.png", dpi=130, bbox_inches="tight")

# figure 2: the ratio per run as box plots, train vs val, plus by verb
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.6]})
pos = 0; ticks = []; labels = []
for i, (label, _) in enumerate(runs):
    for s in ("train", "val"):
        g = df[(df.run == label) & (df.set == s)].ratio
        axes[0].boxplot(g, positions=[pos], widths=0.7, showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor=PALETTE[i], alpha=0.75 if s == "train" else 0.35, edgecolor=PALETTE[i]), medianprops=dict(color="#0b0b0b"),
                        whiskerprops=dict(color=PALETTE[i]), capprops=dict(color=PALETTE[i]))
        ticks.append(pos); labels.append(f"{label}\n{s}"); pos += 1
    pos += 0.6
axes[0].set_xticks(ticks); axes[0].set_xticklabels(labels, fontsize=8); axes[0].axhline(1.0, color="#898781", lw=0.8, ls=":")
axes[0].set_ylabel("error of the mean / spread, per anchor"); axes[0].set_title("calibration ratio per run (dark = train, light = val)", loc="left")
verbs = ["grasp", "move", "release", "other"]
w = 0.8 / max(len(runs), 1)
for i, (label, _) in enumerate(runs):
    for s, alpha in (("train", 0.9), ("val", 0.4)):
        v = [df[(df.run == label) & (df.set == s) & (df.verb == vb)].ratio.median() for vb in verbs]
        x = np.arange(len(verbs)) + (i - (len(runs) - 1) / 2) * w + (-0.18 if s == "train" else 0.18) * w
        axes[1].bar(x, v, width=w * 0.34, color=PALETTE[i], alpha=alpha, label=f"{label} {s}")
axes[1].set_xticks(np.arange(len(verbs))); axes[1].set_xticklabels(verbs); axes[1].axhline(1.0, color="#898781", lw=0.8, ls=":")
axes[1].set_ylabel("median error / spread"); axes[1].set_title("by verb (dark = train, light = val)", loc="left"); axes[1].legend(frameon=False, fontsize=7, ncol=len(runs))
for ax in axes:
    ax.grid(color="#e1e0d9", lw=0.6, axis="y"); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"{OUT}/compare_ratio.png", dpi=130, bbox_inches="tight")
print(f"\nwrote {OUT}/summary.csv, ratio_by_verb.csv, compare_spread_vs_error.png, compare_ratio.png")
