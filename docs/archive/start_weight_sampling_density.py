"""Expected draws per eligible start, per frame, under the trainer's start weights (2026-10-16 v4: deployed path).

Weights come from lerobot.rl.data_sources.start_weights.frame_weights with the `diverse.rebot_start_weights`
block of config_rl.yaml (smooth) and the same block with smooth=false (hard). Group shares are the config's
rebot_group_weights; inside a group a start is drawn in proportion to its weight. Unit of the curves: draws
per start per 2000-step run (256k samples x group share x weight / group mass). Diverse untouched.
Writes one JSON per picked episode (per-frame densities + segments + close/open frames) and a PNG.

Archive. The two migration/ paths below are removed inputs (scratch folders deleted after 2026-11-13):
OUT was the output folder, segments.parquet the subtask spans of the 2026-10-15 subtask-frames pass.
"""
import dataclasses, json, yaml, draccus, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lerobot.configs.diverse import DiverseCollectionConfig
from lerobot.rl.data_sources.episode_groups import read_episode_ranges
from lerobot.rl.data_sources.start_weights import frame_weights

OUT = Path("migration/sampling_density_2026-10-16"); FPS = 30; SAMPLES = 256_000
raw = yaml.safe_load(open("config_rl.yaml")); ROOT = {s["name"]: s["root"] for s in raw["dataset"]["sources"]}
CFG = draccus.decode(DiverseCollectionConfig, raw["diverse"]).rebot_start_weights
SCHEME_CFG = {"hard": dataclasses.replace(CFG, smooth=False), "smooth": CFG}
COLORS = {"hard": "#c98500", "smooth": "#d55181"}
SEG = pd.read_parquet("migration/subtask_frames_2026-10-15/segments.parquet")   # episode-local spans, weight-0 groups dropped
RANGES = {n: read_episode_ranges(r) for n, r in ROOT.items()}
SIZE = {n: max(b for _, b in rg.values()) for n, rg in RANGES.items()}
W = {k: {n: frame_weights(ROOT[n], SIZE[n], c) for n in ROOT} for k, c in SCHEME_CFG.items()}

SHARE = raw["diverse"]["rebot_group_weights"]   # absolute shares of all samples per group (diverse = the rest)

def report(name):
    """Per-group scale (draws per unit weight) and the implied verb shares, with the config's group shares."""
    rows = []
    for r in SEG.itertuples():
        lo = RANGES[r.source][r.episode_index][0]
        rows.append((r.verb if r.verb in ("grasp", "move", "release", "return") else "skill", r.group, int(r.precision), W[name][r.source][lo + r.from_index:lo + r.to_index][::3].sum()))
    m = pd.DataFrame(rows, columns=["verb", "group", "precision", "mass"])
    scale = {g: SAMPLES * SHARE[g] / x for g, x in m.groupby("group").mass.sum().items()}   # draws per start per unit weight
    m["samples"] = m.mass * m.group.map(scale); tot = m.samples.sum(); ev = m.groupby("verb").size(); per = m.groupby("verb").samples.sum() / ev
    print(f"{name:7s} ReBot samples {tot:.0f} | batch % " + " ".join(f"{v} {100 * x / tot:.0f}" for v, x in m.groupby('verb').samples.sum().items())
          + " | per event vs move: " + " ".join(f"{v} {per[v] / per['move']:.1f}" for v in ("grasp", "release", "skill", "return"))
          + " | draws/start at base by group: " + " ".join(f"{g[:5]} {x:.2f}" for g, x in scale.items()))
    g = m[m.verb == "grasp"]; print("        grasp per event by precision vs move:", {int(k): round(float(v / per["move"]), 1) for k, v in (g.groupby("precision").samples.sum() / g.groupby("precision").size()).items()})
    return scale

if __name__ == "__main__":
    SCALE = {k: report(k) for k in SCHEME_CFG}
    GROUP = SEG.groupby(["source", "episode_index"]).group.first()
    ev = pd.concat(pd.read_parquet(f"{r}/meta/depth_gripper_events.parquet").assign(source=n) for n, r in ROOT.items())
    ext = SEG[SEG.group == "external_rebot"].groupby("episode_index").agg(q=("quality", "mean"), grasps=("verb", lambda v: (v == "grasp").sum()), frames=("frames", "first"))
    ext_ep = int(ext[(ext.grasps >= 2) & (ext.frames >= 600)].sort_values("q", ascending=False).index[0])
    PICKS = [("internal_oct07", 0, "nail clippers"), ("internal_bits_kit", 10, "bit"), ("internal_oct04", 0, "headphone case"),
             ("internal_oct08", 0, "stand-up bottle"), ("internal_main", 6, "clothing"), ("external_rebot_train", ext_ep, "external")]
    dens = {(s, e): {k: W[k][s][RANGES[s][e][0]:RANGES[s][e][1]] * SCALE[k][GROUP[(s, e)]] for k in SCHEME_CFG} for s, e, _ in PICKS}
    ymax = max(v.max() for d in dens.values() for v in d.values())
    for source, ep, label in PICKS:
        segs = SEG[(SEG.source == source) & (SEG.episode_index == ep)].sort_values("segment_index"); lo, hi = RANGES[source][ep]; n = hi - lo
        e = ev[(ev.source == source) & (ev.episode_index == ep)]; closes = e[e.event_type == "close"].frame_index.tolist(); opens = e[e.event_type == "open"].frame_index.tolist()
        rec = {"source": source, "root": ROOT[source], "episode": ep, "label": label, "task": segs.task.iloc[0], "n": n, "fps": FPS, "ymax": float(ymax), "closes": closes, "opens": opens,
               "segments": [{"from": int(s.from_index), "to": int(s.to_index), "subtask": s.subtask, "contact": s.contact_slug, "precision": int(s.precision), "quality": int(s.quality)} for s in segs.itertuples()],
               "density": {k: np.round(v, 4).tolist() for k, v in dens[(source, ep)].items()}, "colors": COLORS}
        json.dump(rec, open(OUT / f"{label.replace(' ', '_')}.json", "w"))
        fig, (ax, bar) = plt.subplots(2, 1, figsize=(14, 5.2), height_ratios=[4, 1], sharex=True, gridspec_kw={"hspace": 0.05}); t = np.arange(n) / FPS
        for name, arr in dens[(source, ep)].items(): ax.step(t, arr, where="post", lw=1.6, color=COLORS[name], label=name)
        for f in closes: ax.axvline(f / FPS, color="#888", lw=0.8, ls=":")
        for f in opens: ax.axvline(f / FPS, color="#bbb", lw=0.8, ls="--")
        ax.set_yscale("log"); ax.set_ylim(0.02, ymax * 1.3); ax.set_ylabel("draws per start / run"); ax.grid(axis="y", color="#ddd", lw=0.6); ax.legend(ncol=2, frameon=False, fontsize=9, loc="upper left")
        ax.set_title(f"{label}: {source} ep{ep}  --  {rec['task']}   (dotted = close, dashed = open)", fontsize=10, loc="left")
        for i, s in enumerate(rec["segments"]):
            bar.axvspan(s["from"] / FPS, s["to"] / FPS, color="#cfd8dc" if i % 2 else "#90a4ae", lw=0); bar.text((s["from"] + s["to"]) / 2 / FPS, 0.5, f"{s['subtask'].split()[0]} p{s['precision']}", ha="center", va="center", fontsize=7)
        bar.set_yticks([]); bar.set_xlabel("s"); bar.set_xlim(0, n / FPS)
        fig.savefig(OUT / f"{label.replace(' ', '_')}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"{label:16s} {source} ep{ep} n={n}  mean draws/start: " + "  ".join(f"{k} {v[::3].mean():.2f}" for k, v in dens[(source, ep)].items()))
