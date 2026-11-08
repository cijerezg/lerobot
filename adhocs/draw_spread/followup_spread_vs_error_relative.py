"""spread_vs_error with both axes divided by the motion size: rms of the demo chunk, floored at its 10th
percentile over all anchors so tiny motions do not blow the ratios up.

    uv run python lerobot/adhocs/draw_spread/followup_spread_vs_error_relative.py outputs/draw_spread/internal1200_2026-11-07
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
COLOR = {"train": "#2a78d6", "val": "#eb6834"}
df = pd.read_csv(f"{OUT}/spread_vs_error.csv")
floor = float(np.quantile(df.demo, 0.10))
den = np.maximum(df.demo, floor)
for k in ("spread", "bias", "best", "mean_draw_err"):
    df[f"{k}_rel"] = df[k] / den

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.3})
for ax, key, label in ((axes[0], "bias", "error of the draw mean / motion"), (axes[1], "best", "error of the best draw / motion"), (axes[2], "mean_draw_err", "error of a typical draw / motion")):
    for s in ("train", "val"):
        d = df[df.set == s]
        ax.scatter(d.spread_rel, d[f"{key}_rel"], s=8, color=COLOR[s], alpha=0.4, edgecolors="none", label=s)
        bins = np.quantile(d.spread_rel, np.linspace(0, 1, 8))
        idx = np.clip(np.searchsorted(bins, d.spread_rel, side="right") - 1, 0, len(bins) - 2)
        med = d.groupby(idx).agg(x=("spread_rel", "median"), y=(f"{key}_rel", "median"))
        ax.plot(med.x, med.y, color=COLOR[s], lw=2.2, marker="o", ms=5, markeredgecolor="white")
    lo, hi = df.spread_rel.min(), df.spread_rel.max()
    ax.plot([lo, hi], [lo, hi], color="#898781", lw=0.8, ls="--")
    ax.axhline(1.0, color="#898781", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rms spread of the 64 draws / motion"); ax.set_ylabel(label)
    ax.legend(loc="upper left", markerscale=2, frameon=False)
    ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("mean vs demo, relative", loc="left"); axes[1].set_title("best of 64 vs demo, relative", loc="left"); axes[2].set_title("typical draw vs demo, relative", loc="left")
fig.suptitle(f"motion = rms of the demo chunk, floored at its 10th percentile ({floor:.3f}); dotted = error equals the motion (holding still)", x=0.01, ha="left", fontsize=9, color="#52514e")
fig.savefig(f"{OUT}/spread_vs_error_relative.png", dpi=130, bbox_inches="tight")

df["spread_bin"] = pd.qcut(df.spread_rel, 5, labels=["lowest", "low", "mid", "high", "highest"])
print(f"floor {floor:.3f}; anchors at the floor: {(df.demo < floor).mean():.0%}")
print(df.groupby(["spread_bin", "set"], observed=True)[["spread_rel", "bias_rel", "best_rel"]].median().round(3))
print("corr(log spread_rel, log bias_rel):", {s: round(np.corrcoef(np.log(d.spread_rel), np.log(d.bias_rel))[0, 1], 2) for s, d in df.groupby("set")})
