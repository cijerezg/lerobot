"""Read the two probe caches of one robot episode into the review's figures and numbers.

Inputs: the representations cache (cache/<group>.npy [frames, layers, D], meta.json rows), the action-trace records
(action_trace_records.npz) and the run's online labels (is_intervention). Everything is in real episode seconds.

Per window (clipper, lid) and layer:
- matrix_*.png: episode-centered cosine matrix per token group (centered on the whole episode's sampled mean, as
  the probe's "episode_centered" geometry), interventions shaded, segment boundaries drawn.
- blocks_*.csv: mean cosine between segments (within-segment on the diagonal) per group.
- series_*.png: local cosine to the previous frame, mean cosine to the "did the right thing" reference frames,
  and the executed joint speed, on one time axis.
- intent_*.png + intent.csv: from the trace records, per anchor: end-effector displacement the model commands over
  its 1 s chunk (median over flow samples, and the FAST decode) against what was executed (the operator during an
  intervention): magnitudes, direction agreement (cosine of the two displacement vectors), fan spread, gripper.
- keyframes_*.png: top-camera strip.
- summary.md: the numbers.

    uv run python lerobot/adhocs/intervention_episode_reps/readout.py [reps_dir] [trace_dir] [labels] [out]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics  # noqa: E402

REPS = "migration/inference_review_2026-10-20/reps_probe/validation/step_00001200/representations"
TRACE = "migration/inference_review_2026-10-20/trace_probe/action_trace"
LABELS = "outputs/train/2026-10-20/21-35-32_rebot-mix40-ckpt1200-20261020/inference_dataset/meta/online_labels.parquet"
OUT = "migration/inference_review_2026-10-20/representations"
FPS = 30
GROUPS = ("img_external_0", "img_wrist_0", "state", "subtask", "action_output", "action")
LAYERS = (14, 28, 32)
# Windows and their segments (real s). "reference" = where the model did the right thing.
# "every": keep every k-th sampled frame (k x 0.5 s) so a matrix stays readable.
WINDOWS = {
    "interv1_zoom": {"span": (78.0, 118.0), "every": 1,
                     "segments": [("off", 78.0, 91.4), ("interv1", 91.4, 102.0), ("after", 102.0, 113.3), ("recovery", 113.3, 118.0)],
                     "reference": [("clean grasp 5-17.5", 5.0, 17.5), ("recovery 113.3-132", 113.3, 132.0)]},
    "clipper": {"span": (52.5, 132.0), "every": 4,
                "segments": [("off", 52.5, 91.4), ("interv1", 91.4, 102.0), ("after", 102.0, 113.3), ("recovery", 113.3, 132.0)],
                "reference": [("clean grasp 5-17.5", 5.0, 17.5), ("recovery 113.3-132", 113.3, 132.0)]},
    "lid": {"span": (208.0, 248.5), "every": 2,
            "segments": [("pre", 208.0, 217.4), ("interv3", 217.4, 223.7), ("between", 223.7, 229.5), ("interv4", 229.5, 237.0), ("after", 237.0, 248.5)],
            "reference": [("after 237-248.5", 237.0, 248.5)]},
}


def md(df):
    """Markdown table without the tabulate dependency."""
    df = df.reset_index()
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def interventions(labels):
    iv = labels["is_intervention"].to_numpy().astype(int)
    f = labels["frame_index"].to_numpy()
    e = np.flatnonzero(np.diff(np.r_[0, iv, 0]))
    return [(f[a] / FPS, (f[b - 1] + 1) / FPS) for a, b in zip(e[::2], e[1::2])]


def shade(ax, spans, axis="x", within=None):
    for a, b in spans:
        if within is not None and (b < within[0] or a > within[1]):
            continue
        (ax.axvspan if axis == "x" else ax.axhspan)(a, b, color="red", alpha=0.12, lw=0)


def main(reps_dir=REPS, trace_dir=TRACE, labels_path=LABELS, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(reps_dir) / "cache" / "meta.json").read_text())
    rows = [r for r in meta["rows"] if r["text"] == "real"]
    t = np.array([r["time_s"] for r in rows])
    layer_pos = {l: meta["layers"].index(l) for l in LAYERS if l in meta["layers"]}
    reps = {g: np.load(Path(reps_dir) / "cache" / f"{g}.npy", mmap_mode="r")[[r["row"] for r in rows]] for g in GROUPS
            if (Path(reps_dir) / "cache" / f"{g}.npy").exists()}
    labels = pd.read_parquet(labels_path).sort_values("frame_index")
    iv = interventions(labels)
    state_all = None
    summary = ["# Representations and intent along the 2026-10-20 21-35-32 episode\n",
               f"Representations: {len(rows)} frames at {meta['stride_s']} s, layers {meta['layers']}, view {meta['view']}, "
               f"checkpoint {meta['checkpoint']}.", f"Interventions (s): {[(round(a, 1), round(b, 1)) for a, b in iv]}\n"]

    # ---- representations ----
    for wname, w in WINDOWS.items():
        a, b = w["span"]
        sel = np.flatnonzero((t >= a - 1e-6) & (t <= b + 1e-6))[::int(w.get("every", 1))]
        if len(sel) < 2:
            summary.append(f"{wname}: fewer than 2 sampled frames in the window, skipped.")
            continue
        tw = t[sel]
        ref_sets = {name: np.flatnonzero((t >= ra - 1e-6) & (t <= rb + 1e-6)) for name, ra, rb in w["reference"]}
        seg_of = np.array([next((n for n, sa, sb in w["segments"] if sa <= x < sb), "") for x in tw])
        for layer, lp in layer_pos.items():
            fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), layout="constrained")
            blocks = []
            for ax, g in zip(axes.ravel(), GROUPS):
                if g not in reps:
                    ax.set_axis_off()
                    continue
                x = reps[g][:, lp].astype(np.float32)
                u = unit(x - x.mean(axis=0, keepdims=True))  # episode-centered on all sampled frames
                c = u[sel] @ u[sel].T
                im = ax.imshow(c, extent=[tw[0], tw[-1], tw[-1], tw[0]], cmap="RdBu_r", vmin=-1, vmax=1)
                shade(ax, iv, "x", within=(a, b))
                ax.set_xlim(tw[0], tw[-1])
                ax.set_ylim(tw[-1], tw[0])
                for _, sa, sb in w["segments"][1:]:
                    ax.axvline(sa, color="k", lw=0.6)
                    ax.axhline(sa, color="k", lw=0.6)
                ax.set_title(f"{g}  L{layer}")
                ax.set_xlabel("s")
                names = [n for n, _, _ in w["segments"]]
                tab = pd.DataFrame(index=names, columns=names, dtype=float)
                for n1 in names:
                    for n2 in names:
                        m1, m2 = seg_of == n1, seg_of == n2
                        if m1.any() and m2.any():
                            sub = c[np.ix_(m1, m2)]
                            tab.loc[n1, n2] = sub[~np.eye(len(sub), dtype=bool)].mean() if n1 == n2 and m1.sum() > 1 else sub.mean()
                tab.insert(0, "group", g)
                blocks.append(tab.round(3))
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="cosine (episode-centered)")
            fig.suptitle(f"{wname} window {a}-{b} s, {len(sel)} frames every {np.median(np.diff(tw)):.1f} s, layer {layer}; "
                         "red = operator intervention, lines = segment boundaries")
            fig.savefig(out / f"matrix_{wname}_L{layer}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            bl = pd.concat(blocks).rename_axis("segment")
            bl.to_csv(out / f"blocks_{wname}_L{layer}.csv")
            if layer == 28:
                summary.append(f"\n## {wname}: segment-to-segment mean cosine, L{layer} (diagonal = within segment)\n")
                summary.append(bl.pipe(md))

        # time series: local cosine, reference similarity, executed speed
        fig, axes = plt.subplots(len(reps) + 1, 1, figsize=(16, 2.1 * (len(reps) + 1)), sharex=True)
        lp = layer_pos.get(28, list(layer_pos.values())[0])
        ref_lines = []
        for ax, g in zip(axes, [g for g in GROUPS if g in reps]):
            x = reps[g][:, lp].astype(np.float32)
            u = unit(x - x.mean(axis=0, keepdims=True))
            local = np.r_[np.nan, (u[sel][1:] * u[sel][:-1]).sum(-1)]
            ax.plot(tw, local, color="k", lw=1.2, label="local (vs previous frame)")
            for (name, idx), col in zip(ref_sets.items(), ("tab:green", "tab:blue")):
                if len(idx):
                    other = np.array([u[j] @ u[idx[idx != j]].T if (idx != j).any() else np.array([np.nan]) for j in sel], dtype=object)
                    sim = np.array([o.mean() for o in other])
                    ax.plot(tw, sim, color=col, lw=1.2, label=f"mean cos to {name}")
                    ref_lines.append((g, name, tw, sim))
            shade(ax, iv, within=(a, b))
            ax.set_ylabel(g, fontsize=9)
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="gray", lw=0.5)
        axes[0].legend(loc="lower left", fontsize=8, ncol=3)
        st = labels_state(labels_path)
        tt = np.arange(len(st)) / FPS
        m = (tt >= a) & (tt <= b)
        speed = np.r_[0, np.linalg.norm(np.diff(st[:, :6], axis=0), axis=1)] * FPS
        axes[-1].plot(tt[m], speed[m], color="k", lw=0.8, label="executed arm speed (deg/s, 6 joints)")
        ax2 = axes[-1].twinx()
        ax2.plot(tt[m], st[m, 6], color="tab:orange", lw=0.8, label="gripper")
        shade(axes[-1], iv, within=(a, b))
        axes[-1].set_xlim(a, b)
        axes[-1].set_ylabel("speed")
        ax2.set_ylabel("gripper")
        axes[-1].legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("real episode s")
        fig.suptitle(f"{wname}: L28 episode-centered cosine, red = intervention")
        fig.savefig(out / f"series_{wname}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        summary.append(f"\n## {wname}: mean cosine to the reference frames, per segment (L28)\n")
        tab = {}
        for g, name, tw_, sim in ref_lines:
            for n, sa, sb in w["segments"]:
                mm = (tw_ >= sa) & (tw_ < sb)
                tab.setdefault((g, name), {})[n] = np.nanmean(sim[mm]) if mm.any() else np.nan
        summary.append(pd.DataFrame(tab).T.rename_axis(["group", "reference"]).astype(float).round(3).pipe(md))
        # discontinuities: the lowest local cosines
        g = "img_external_0" if "img_external_0" in reps else list(reps)[0]
        x = reps[g][:, lp].astype(np.float32)
        u = unit(x - x.mean(axis=0, keepdims=True))
        local = (u[sel][1:] * u[sel][:-1]).sum(-1)
        order = np.argsort(local)[:6]
        summary.append(f"\n{wname}: lowest local cosines ({g}, L28): " + ", ".join(f"{tw[i + 1]:.1f}s={local[i]:.2f}" for i in sorted(order)))

        # key frames strip
        thumbs = Path(reps_dir) / "cache" / "thumbs"
        picks = sel[::max(1, len(sel) // 24)][:24]
        fig, axes = plt.subplots(2, (len(picks) + 1) // 2, figsize=(2.2 * ((len(picks) + 1) // 2), 4.2))
        for ax, j in zip(axes.ravel(), picks):
            p = thumbs / f"{rows[j]['sample']:04d}.external_0.jpg"
            if p.exists():
                ax.imshow(Image.open(p))
            inter = any(ia <= t[j] < ib for ia, ib in iv)
            ax.set_title(f"{t[j]:.1f}s" + (" INTERV" if inter else ""), fontsize=8, color="red" if inter else "k")
            ax.set_axis_off()
        for ax in axes.ravel()[len(picks):]:
            ax.set_axis_off()
        fig.savefig(out / f"keyframes_{wname}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ---- intent from the trace records ----
    rec = Path(trace_dir) / "action_trace_records.npz"
    if rec.exists():
        z = np.load(rec)
        kin = RebotKinematics()
        tf = z["frame"] / FPS
        st, gt, smp = z["state"], z["gt_chunk"], z["sample_chunks"]
        fast = z["fast_chunk"] if "fast_chunk" in z.files else None
        ee0 = kin.ee_path(st)
        ee_gt = kin.ee_path(gt[:, -1]) - ee0
        ee_s = np.stack([kin.ee_path(smp[:, k, -1]) - ee0 for k in range(smp.shape[1])], axis=1)  # [N, K, 3]
        ee_med = np.median(ee_s, axis=1)
        cosd = lambda p, q: (p * q).sum(-1) / np.maximum(np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1), 1e-6)
        rows_i = pd.DataFrame({
            "t": tf, "subtask": z["subtask"],
            "intervention": [any(a <= x < b for a, b in iv) for x in tf],
            "exec_cm": np.linalg.norm(ee_gt, axis=1) * 100,
            "model_cm": np.linalg.norm(ee_med, axis=1) * 100,
            "dir_agree": cosd(ee_med, ee_gt),
            "fan_spread_cm": np.array([np.linalg.norm(ee_s[i][:, None] - ee_s[i][None], axis=-1).mean() for i in range(len(tf))]) * 100,
            "grip_exec": gt[:, -1, 6] - st[:, 6], "grip_model": np.median(smp[:, :, -1, 6], axis=1) - st[:, 6],
            "joint_err_deg": np.abs(np.median(smp[:, :, -1, :6], axis=1) - gt[:, -1, :6]).mean(-1),
            "hold_err_deg": np.abs(st[:, None, :6] - gt[:, -1:, :6]).mean(-1)[:, 0],
        })
        if fast is not None:
            ee_f = kin.ee_path(fast[:, -1]) - ee0
            rows_i["fast_cm"] = np.linalg.norm(ee_f, axis=1) * 100
            rows_i["fast_agree_flow"] = cosd(ee_f, ee_med)
            rows_i["fast_agree_exec"] = cosd(ee_f, ee_gt)
        rows_i.to_csv(out / "intent.csv", index=False)
        for wname, w in WINDOWS.items():
            a, b = w["span"]
            m = (rows_i.t >= a) & (rows_i.t <= b)
            if not m.any():
                continue
            d = rows_i[m]
            fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
            axes[0].plot(d.t, d.exec_cm, "k", label="executed EE displacement over 1 s (cm)")
            axes[0].plot(d.t, d.model_cm, "tab:blue", label="model flow median")
            if fast is not None:
                axes[0].plot(d.t, d.fast_cm, "tab:orange", lw=0.8, label="model FAST")
            axes[0].fill_between(d.t, 0, d.fan_spread_cm, color="tab:blue", alpha=0.15, label="fan spread (mean pairwise, cm)")
            axes[1].plot(d.t, d.dir_agree, "tab:blue", label="direction agreement flow vs executed (cos)")
            if fast is not None:
                axes[1].plot(d.t, d.fast_agree_flow, "tab:orange", lw=0.8, label="FAST vs flow")
            axes[1].axhline(0, color="gray", lw=0.5)
            axes[1].set_ylim(-1, 1)
            axes[2].plot(d.t, d.grip_exec, "k", label="executed gripper change")
            axes[2].plot(d.t, d.grip_model, "tab:blue", label="model gripper change")
            axes[3].plot(d.t, d.joint_err_deg, "tab:blue", label="|model - executed| at chunk end, mean over 6 joints (deg)")
            axes[3].plot(d.t, d.hold_err_deg, "gray", lw=0.8, label="|hold still - executed|")
            for ax in axes:
                shade(ax, iv, within=(a, b))
                ax.legend(loc="upper left", fontsize=8)
            axes[-1].set_xlim(a, b)
            axes[-1].set_xlabel("real episode s")
            fig.suptitle(f"{wname}: model intent per anchor vs what was executed (red = operator holds the arm)")
            fig.savefig(out / f"intent_{wname}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            summary.append(f"\n## {wname}: intent per segment (medians over anchors)\n")
            d = d.assign(segment=[next((n for n, sa, sb in w["segments"] if sa <= x < sb), "") for x in d.t])
            cols = ["exec_cm", "model_cm", "dir_agree", "fan_spread_cm", "joint_err_deg", "hold_err_deg", "grip_exec", "grip_model"]
            if fast is not None:
                cols += ["fast_cm", "fast_agree_flow"]
            summary.append(d.groupby("segment")[cols].median().round(2).pipe(md))
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print((out / "summary.md").read_text())


def labels_state(labels_path):
    root = Path(labels_path).parent.parent
    import glob
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "data/*/*.parquet")))])
    return np.stack(df.sort_values("frame_index")["observation.state"].to_numpy())


if __name__ == "__main__":
    main(*sys.argv[1:])
