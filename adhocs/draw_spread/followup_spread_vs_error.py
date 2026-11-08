"""Spread vs error per anchor, from the saved draws. x = rms spread of the 64 draws; y = rms error of the draw mean
(bias) and rms error of the best single draw. Binned medians per set. Low spread + low error = predictable and
learned; low spread + high error = confidently wrong.

    uv run python lerobot/adhocs/draw_spread/followup_spread_vs_error.py outputs/draw_spread/internal1200_2026-11-07
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
COLOR = {"train": "#2a78d6", "val": "#eb6834"}
rows = []
for path in sorted(glob.glob(f"{OUT}/episodes/*.npz")):
    z = np.load(path, allow_pickle=True)
    set_name = os.path.basename(path).split("_")[0]
    draws, gt = z["norm_clean"].astype(np.float64), z["gt_norm"].astype(np.float64)  # [N, K, T, D], [N, T, D]
    mean = draws.mean(1)
    spread = np.sqrt(((draws - mean[:, None]) ** 2).mean(axis=(1, 2, 3)))
    bias = np.sqrt(((mean - gt) ** 2).mean(axis=(1, 2)))
    per_draw = np.sqrt(((draws - gt[:, None]) ** 2).mean(axis=(2, 3)))  # [N, K]
    for i in range(len(gt)):
        rows.append({"set": set_name, "episode": os.path.basename(path)[:-4], "frame": int(z["frame"][i]), "subtask": str(z["subtask"][i]),
                     "spread": spread[i], "bias": bias[i], "best": per_draw[i].min(), "mean_draw_err": per_draw[i].mean(),
                     "demo": np.sqrt((gt[i] ** 2).mean())})
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/spread_vs_error.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.3})
for ax, key, label in ((axes[0], "bias", "rms error of the draw mean"), (axes[1], "best", "rms error of the best draw"), (axes[2], "mean_draw_err", "rms error of a typical draw")):
    for s in ("train", "val"):
        d = df[df.set == s]
        ax.scatter(d.spread, d[key], s=8, color=COLOR[s], alpha=0.4, edgecolors="none", label=s)
        bins = np.quantile(d.spread, np.linspace(0, 1, 8))
        idx = np.clip(np.searchsorted(bins, d.spread, side="right") - 1, 0, len(bins) - 2)
        med = d.groupby(idx).agg(x=("spread", "median"), y=(key, "median"))
        ax.plot(med.x, med.y, color=COLOR[s], lw=2.2, marker="o", ms=5, markeredgecolor="white")
    lo, hi = df.spread.min(), df.spread.max()
    ax.plot([lo, hi], [lo, hi], color="#898781", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rms spread of the 64 draws"); ax.set_ylabel(label)
    ax.legend(loc="upper left", markerscale=2, frameon=False)
    ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("mean vs demo (bias)", loc="left"); axes[1].set_title("best of 64 vs demo", loc="left"); axes[2].set_title("typical draw vs demo", loc="left")
fig.savefig(f"{OUT}/spread_vs_error.png", dpi=130, bbox_inches="tight")

df["spread_bin"] = pd.qcut(df.spread, 5, labels=["lowest", "low", "mid", "high", "highest"])
print(df.groupby(["spread_bin", "set"], observed=True)[["spread", "bias", "best", "demo"]].median().round(3))
print("\ncorr(log spread, log bias):", {s: round(np.corrcoef(np.log(d.spread), np.log(d.bias))[0, 1], 2) for s, d in df.groupby("set")})
