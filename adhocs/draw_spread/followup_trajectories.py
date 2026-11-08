"""The 64 drawn chunks, their mean and the demo chunk for the highest-spread anchors.

Selects the 6 val and 2 train anchors with the highest relative spread (spread / max(demo rms, 10th percentile),
subtask not empty, at most 2 per episode). One PNG per anchor: frame thumbnail, 7 joint strips in degrees
(draws thin, mean bold, demo dashed, state at step 0), the end-effector paths in 3D through RebotKinematics.
A k-means (k=2) check on the flattened normalized chunks says whether the draws are one blob or two clusters.

    uv run python lerobot/adhocs/draw_spread/followup_trajectories.py outputs/draw_spread/internal1200_2026-11-07

Writes <out>/trajectories/<rank>_<set>_<tag>_f<frame>.png, summary.csv, index.html.
"""
import base64
import html
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics

OUT = sys.argv[1]
DST = f"{OUT}/trajectories"
os.makedirs(DST, exist_ok=True)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
C_DRAW, C_MEAN, C_DEMO, C_STATE = "#9ec5f4", "#2a78d6", "#eb6834", "#0b0b0b"
N_VAL, N_TRAIN, PER_EPISODE = 6, 2, 2

# ---- selection -------------------------------------------------------------------------------------------------
e = pd.read_csv(f"{OUT}/spread_vs_error.csv")
floor = float(np.quantile(e.demo, 0.10))
e["motion"] = np.maximum(e.demo, floor)
for k in ("spread", "bias", "mean_draw_err"):
    e[f"{k}_rel"] = e[k] / e.motion
e["root_base"] = e.episode.str.replace(r"^(train|val)_", "", regex=True).str.replace(r"_ep\d+$", "", regex=True)
e["ep"] = e.episode.str.extract(r"_ep(\d+)$").astype(int)
a = pd.read_csv(f"{OUT}/anchors.csv")
a = a[a.setting == "clean"].copy()
a["root_base"] = a.root.map(os.path.basename)
df = e.merge(a[["set", "root_base", "episode", "frame", "global_idx", "time_s"]],
             left_on=["set", "root_base", "ep", "frame"], right_on=["set", "root_base", "episode", "frame"], suffixes=("", "_a"))
assert len(df) == len(e), (len(df), len(e))
df = df[df.subtask.fillna("").str.strip() != ""]


def pick(d, n):
    d = d.sort_values("spread_rel", ascending=False)
    keep, count = [], {}
    for r in d.itertuples():
        if count.get(r.episode, 0) < PER_EPISODE:
            keep.append(r.Index)
            count[r.episode] = count.get(r.episode, 0) + 1
        if len(keep) == n:
            break
    return d.loc[keep]


sel = pd.concat([pick(df[df.set == "val"], N_VAL), pick(df[df.set == "train"], N_TRAIN)]).reset_index(drop=True)
sel["rank"] = np.arange(1, len(sel) + 1)
thumbs = pd.read_json(f"{OUT}/thumbs.json", typ="series")
kin = RebotKinematics()


# ---- multimodality check ---------------------------------------------------------------------------------------
def kmeans2(x):
    """Labels of a k=2 clustering of the rows of x."""
    try:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=2, n_init=10, random_state=0).fit(x).labels_
    except ImportError:
        rng = np.random.default_rng(0)
        centres = x[rng.choice(len(x), 2, replace=False)]
        for _ in range(10):
            labels = np.argmin(((x[:, None] - centres[None]) ** 2).sum(-1), axis=1)
            centres = np.stack([x[labels == k].mean(0) if (labels == k).any() else centres[k] for k in range(2)])
        return labels


def cluster_check(norm):
    x = norm.reshape(len(norm), -1).astype(np.float64)
    labels = kmeans2(x)
    total = ((x - x.mean(0)) ** 2).sum()
    within = sum(((x[labels == k] - x[labels == k].mean(0)) ** 2).sum() for k in range(2))
    return np.sqrt(within / total), min((labels == 0).mean(), (labels == 1).mean())


# ---- figure ----------------------------------------------------------------------------------------------------
def render(r, draws_deg, norm, gt_deg, state_deg, png):
    mean_deg = draws_deg.mean(0)
    per_joint_sd = draws_deg.std(0).mean(0)
    steps = np.arange(draws_deg.shape[1])
    ee_draws = np.stack([kin.ee_path(c) for c in draws_deg.astype(np.float64)])
    ee_mean, ee_gt = kin.ee_path(mean_deg.astype(np.float64)), kin.ee_path(gt_deg.astype(np.float64))
    ee_start = kin.ee_path(state_deg.astype(np.float64)[None])[0]

    fig = plt.figure(figsize=(20, 9.5), facecolor="white")
    gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.38, left=0.04, right=0.985, top=0.92, bottom=0.13)
    ax_img = fig.add_subplot(gs[0, :4])
    img = Image.open(io.BytesIO(base64.b64decode(thumbs[f"{r.set}|{r.root_base}|{int(r.global_idx)}"])))
    ax_img.imshow(img)
    ax_img.set_axis_off()
    ax_img.set_title(f"frame {int(r.frame)} ({r.time_s:.0f} s) · top camera | wrist camera", loc="left", fontsize=11)

    ax3 = fig.add_subplot(gs[0, 4:], projection="3d")
    for p in ee_draws:
        ax3.plot(p[:, 0], p[:, 1], p[:, 2], color=C_DRAW, lw=1.0, alpha=0.5)
    ax3.plot(ee_mean[:, 0], ee_mean[:, 1], ee_mean[:, 2], color=C_MEAN, lw=3.0, label="mean of 64 draws", zorder=10)
    ax3.plot(ee_gt[:, 0], ee_gt[:, 1], ee_gt[:, 2], color=C_DEMO, lw=3.0, ls="--", label="demo chunk", zorder=10)
    ax3.scatter(*ee_start, color=C_STATE, s=70, marker="D", depthshade=False, label="end effector now", zorder=11)
    pts = np.concatenate([ee_draws.reshape(-1, 3), ee_gt, ee_start[None]])
    centre, half = (pts.min(0) + pts.max(0)) / 2, max((pts.max(0) - pts.min(0)).max() / 2, 0.02) * 1.1
    ax3.set_xlim(centre[0] - half, centre[0] + half); ax3.set_ylim(centre[1] - half, centre[1] + half); ax3.set_zlim(centre[2] - half, centre[2] + half)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x (m)", labelpad=6); ax3.set_ylabel("y (m)", labelpad=6); ax3.set_zlabel("z (m)", labelpad=6)
    ax3.tick_params(labelsize=8)
    ax3.view_init(elev=28, azim=-55)
    ax3.set_title(f"end-effector paths over the 1 s chunk (box ±{half * 100:.0f} cm)", loc="left", fontsize=11)
    ax3.legend(loc="upper left", bbox_to_anchor=(-0.12, 1.0), frameon=False, fontsize=9)
    ax3.xaxis.pane.fill = ax3.yaxis.pane.fill = ax3.zaxis.pane.fill = False

    for j, name in enumerate(JOINTS):
        ax = fig.add_subplot(gs[1, j])
        for d in draws_deg:
            ax.plot(steps, d[:, j], color=C_DRAW, lw=0.6, alpha=0.35)
        ax.plot(steps, mean_deg[:, j], color=C_MEAN, lw=2.2)
        ax.plot(steps, gt_deg[:, j], color=C_DEMO, lw=2.2, ls="--")
        ax.scatter([0], [state_deg[j]], color=C_STATE, s=40, marker="D", zorder=5)
        ax.set_title(f"{name}\nsd over draws {per_joint_sd[j]:.1f}°", fontsize=10, loc="left")
        ax.set_xlabel("chunk step", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(color="#e1e0d9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("degrees", fontsize=9)

    fig.suptitle(f"#{int(r.rank)}  {r.set}  {r.episode}  frame {int(r.frame)}  —  {r.subtask}", x=0.04, ha="left", fontsize=14, y=0.975)
    fig.text(0.04, 0.03, caption(r), fontsize=11, color="#333333")
    fig.savefig(png, dpi=110)
    plt.close(fig)


def caption(r):
    return (f"{r.set} · {r.episode} · frame {int(r.frame)} · {r.time_s:.0f} s · \"{r.subtask}\" · rms spread / motion {r.spread_rel:.2f} · "
            f"error of the mean / motion {r.bias_rel:.2f} · error of a typical draw / motion {r.mean_draw_err_rel:.2f}")


rows = []
for r in sel.itertuples():
    d = np.load(f"{OUT}/episodes/{r.episode}.npz", allow_pickle=True)
    i = int(np.flatnonzero(d["frame"] == r.frame)[0])
    assert str(d["subtask"][i]) == r.subtask, (d["subtask"][i], r.subtask)
    draws_deg, norm, gt_deg, state_deg = d["abs"][i, 0], d["norm_clean"][i], d["gt_abs"][i], d["state"][i]
    within_ratio, small_frac = cluster_check(norm)
    png = f"{DST}/{int(r.rank)}_{r.set}_{r.episode}_f{int(r.frame)}.png"
    render(r, draws_deg, norm, gt_deg, state_deg, png)
    rows.append({"rank": int(r.rank), "set": r.set, "episode": r.episode, "frame": int(r.frame), "time_s": round(float(r.time_s), 1),
                 "subtask": r.subtask, "spread_rel": round(float(r.spread_rel), 3), "mean_err_rel": round(float(r.bias_rel), 3),
                 "draw_err_rel": round(float(r.mean_draw_err_rel), 3), "within_over_total": round(float(within_ratio), 3),
                 "smaller_cluster_frac": round(float(small_frac), 3), "png": os.path.basename(png), "caption": caption(r)})
    print("wrote", png)

summary = pd.DataFrame(rows)
summary.to_csv(f"{DST}/summary.csv", index=False)
print(summary.drop(columns=["caption", "png"]).to_string(index=False))

cards = "\n".join(
    f'<figure><img src="{row.png}" alt="{html.escape(row.caption)}"><figcaption>{html.escape(row.caption)} · '
    f'k-means k=2: within / total spread {row.within_over_total:.2f}, smaller cluster {row.smaller_cluster_frac:.0%}</figcaption></figure>'
    for row in summary.itertuples())
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>draw_spread: high-spread anchors</title>
<style>body{{font-family:sans-serif;max-width:1800px;margin:20px auto;padding:0 16px;color:#222;background:#fff}}
figure{{margin:0 0 40px 0}} img{{width:100%;border:1px solid #ddd}} figcaption{{font-size:14px;margin-top:6px;color:#444}}
p{{color:#444;font-size:14px}}</style></head><body>
<h1>The 64 draws vs the demo on the highest-spread anchors</h1>
<p>Relative spread = rms spread of the 64 draws / max(rms of the demo chunk, {floor:.3f}). {N_VAL} val and {N_TRAIN} train anchors with the highest
relative spread, subtask not empty, at most {PER_EPISODE} per episode. Draws thin light blue, mean bold blue, demo dashed orange, current state black diamond.
within / total spread near 1 = one blob; well below 1 with a balanced smaller cluster = two groups.</p>
{cards}</body></html>"""
open(f"{DST}/index.html", "w").write(page)
print("wrote", f"{DST}/index.html")
