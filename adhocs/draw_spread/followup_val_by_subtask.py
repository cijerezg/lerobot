"""Val anchors only, relative axes as in followup_spread_vs_error_relative, colored by the subtask verb
(grasp / move / release / other). Prints the same medians split by object family to see whether that matters.

    uv run python lerobot/adhocs/draw_spread/followup_val_by_subtask.py outputs/draw_spread/internal1200_2026-11-07
"""
import re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
COLOR = {"grasp": "#2a78d6", "move": "#eb6834", "release": "#1baf7a", "other": "#898781"}
df = pd.read_csv(f"{OUT}/spread_vs_error.csv")
floor = float(np.quantile(df.demo, 0.10))
den = np.maximum(df.demo, floor)
for k in ("spread", "bias", "best", "mean_draw_err"):
    df[f"{k}_rel"] = df[k] / den
v = df[df.set == "val"].copy()
v["verb"] = v.subtask.fillna("").str.split().str[0].where(lambda s: s.isin(["grasp", "move", "release"]), "other")


def family(s: str) -> str:
    s = s or ""
    if re.search(r"sock|shirt", s): return "clothes"
    if re.search(r"bottle|cup|pills", s): return "bottles/cup"
    if re.search(r"bit|clipper|flap|kit|headphone", s): return "bits/clipper/kit"
    return "other"


FAMILY_COLOR = {"clothes": "#2a78d6", "bottles/cup": "#eb6834", "bits/clipper/kit": "#1baf7a", "other": "#898781"}


v["family"] = v.subtask.fillna("").map(family)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.3})
for ax, key, label in ((axes[0], "bias", "error of the draw mean / motion"), (axes[1], "best", "error of the best draw / motion"), (axes[2], "mean_draw_err", "error of a typical draw / motion")):
    for verb in ("grasp", "move", "release", "other"):
        d = v[v.verb == verb]
        ax.scatter(d.spread_rel, d[f"{key}_rel"], s=10, color=COLOR[verb], alpha=0.5, edgecolors="none", label=f"{verb} ({len(d)})")
        if len(d) >= 30:
            bins = np.quantile(d.spread_rel, np.linspace(0, 1, 6))
            idx = np.clip(np.searchsorted(bins, d.spread_rel, side="right") - 1, 0, len(bins) - 2)
            med = d.groupby(idx).agg(x=("spread_rel", "median"), y=(f"{key}_rel", "median"))
            ax.plot(med.x, med.y, color=COLOR[verb], lw=2.2, marker="o", ms=5, markeredgecolor="white")
    lo, hi = v.spread_rel.min(), v.spread_rel.max()
    ax.plot([lo, hi], [lo, hi], color="#898781", lw=0.8, ls="--")
    ax.axhline(1.0, color="#898781", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rms spread of the 64 draws / motion"); ax.set_ylabel(label)
    ax.legend(loc="upper left", markerscale=1.5, frameon=False, fontsize=8)
    ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("val: mean vs demo", loc="left"); axes[1].set_title("val: best of 64 vs demo", loc="left"); axes[2].set_title("val: typical draw vs demo", loc="left")
fig.suptitle(f"val anchors by subtask verb; motion = rms demo floored at {floor:.3f}; dotted = holding still", x=0.01, ha="left", fontsize=9, color="#52514e")
fig.savefig(f"{OUT}/val_spread_vs_error_by_verb.png", dpi=130, bbox_inches="tight")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.3})
for ax, key, label in ((axes[0], "bias", "error of the draw mean / motion"), (axes[1], "best", "error of the best draw / motion"), (axes[2], "mean_draw_err", "error of a typical draw / motion")):
    for fam in FAMILY_COLOR:
        d = v[v.family == fam]
        ax.scatter(d.spread_rel, d[f"{key}_rel"], s=10, color=FAMILY_COLOR[fam], alpha=0.5, edgecolors="none", label=f"{fam} ({len(d)})")
        if len(d) >= 30:
            bins = np.quantile(d.spread_rel, np.linspace(0, 1, 6))
            idx = np.clip(np.searchsorted(bins, d.spread_rel, side="right") - 1, 0, len(bins) - 2)
            med = d.groupby(idx).agg(x=("spread_rel", "median"), y=(f"{key}_rel", "median"))
            ax.plot(med.x, med.y, color=FAMILY_COLOR[fam], lw=2.2, marker="o", ms=5, markeredgecolor="white")
    lo, hi = v.spread_rel.min(), v.spread_rel.max()
    ax.plot([lo, hi], [lo, hi], color="#898781", lw=0.8, ls="--")
    ax.axhline(1.0, color="#898781", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rms spread of the 64 draws / motion"); ax.set_ylabel(label)
    ax.legend(loc="upper left", markerscale=1.5, frameon=False, fontsize=8)
    ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("val: mean vs demo", loc="left"); axes[1].set_title("val: best of 64 vs demo", loc="left"); axes[2].set_title("val: typical draw vs demo", loc="left")
fig.suptitle(f"val anchors by object family; motion = rms demo floored at {floor:.3f}; dotted = holding still", x=0.01, ha="left", fontsize=9, color="#52514e")
fig.savefig(f"{OUT}/val_spread_vs_error_by_family.png", dpi=130, bbox_inches="tight")

v["ratio"] = v.bias / v.spread
pd.set_option("display.width", 160)
print("by verb:\n", v.groupby("verb").agg(n=("ratio", "size"), spread_rel=("spread_rel", "median"), bias_rel=("bias_rel", "median"), best_rel=("best_rel", "median"), bias_over_spread=("ratio", "median")).round(2))
print("\nby object family x verb:\n", v.groupby(["family", "verb"]).agg(n=("ratio", "size"), spread_rel=("spread_rel", "median"), bias_rel=("bias_rel", "median"), bias_over_spread=("ratio", "median")).round(2))
