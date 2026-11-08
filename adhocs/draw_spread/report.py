"""HTML report of the draw-spread study from the per-anchor table (one row per anchor x setting).

Columns the table must carry: set (train/val), setting (clean / state_noise / image_noise), episode,
time_s, subtask, progress_bin, plus every metric of metrics.anchor_metrics (clean rows) and
metrics.noise_metrics (noise rows). Figures are matplotlib PNGs embedded in one self-contained page.
"""
from __future__ import annotations

import base64
import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {"train": "#2a78d6", "val": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COMPONENTS = ("total", "offset", "shape")
# (column key, label, sqrt?) ; squared quantities are shown as rms in normalized action units
HEADLINE = [
    ("bias2", "rms bias: draw mean vs demo", True),
    ("spread", "rms spread of the 64 draws", True),
    ("mse", "rms error of the draws (bias² + spread)", True),
    ("hold", "rms demo = motion size (hold baseline)", True),
    ("R", "R = mse / hold (1 = holding still)", False),
    ("var_data", "cross-episode std of the demo (bucket)", True),
    ("spread_ratio", "spread std / data std (1 = calibrated)", True),
    ("style_proj", "style projection (0 bucket mean, 1 this episode)", False),
    ("style_cos", "style cosine", False),
]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "-"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3g}"


def _q(s: pd.Series, root: bool = False) -> str:
    s = s.dropna()
    if root:
        s = np.sqrt(s.clip(lower=0))
    if s.empty:
        return "-"
    return f"{_fmt(s.median())} <span class=iqr>[{_fmt(s.quantile(0.25))}, {_fmt(s.quantile(0.75))}]</span>"


def table_headline(clean: pd.DataFrame) -> str:
    rows = []
    for key, desc, root in HEADLINE:
        cells = []
        for s in ("train", "val"):
            sub = clean[clean.set == s]
            cells += [_q(sub[f"{key}_{c}"], root) for c in COMPONENTS]
        rows.append(f"<tr><td>{desc}<br><span class=desc>column {key}_*</span></td>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    e = "".join(f"<td colspan=3>{_q(clean[clean.set == s]['energy'])}</td>" for s in ("train", "val"))
    rows.append(f"<tr><td>energy score of the draw set (lower is better)<br><span class=desc>column energy</span></td>{e}</tr>")
    n = {s: int((clean.set == s).sum()) for s in ("train", "val")}
    head = (f"<tr><th rowspan=2>metric</th><th colspan=3>train ({n['train']} anchors)</th><th colspan=3>val ({n['val']} anchors)</th></tr>"
            "<tr>" + "".join(f"<th>{c}</th>" for c in COMPONENTS * 2) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table><p class=note>median [q25, q75] over anchors; rms rows are square roots of the squared metrics, in normalized action units.</p>"


def table_by_subtask(clean: pd.DataFrame) -> str:
    cols = [("spread_ratio_total", "spread std / data std", True), ("style_proj_offset", "style proj, offset", False),
            ("style_proj_shape", "style proj, shape", False), ("R_total", "R", False), ("hold_total", "rms demo", True)]
    head = "<tr><th rowspan=2>subtask</th><th colspan=2>anchors</th>" + "".join(f"<th colspan=2>{d}</th>" for _, d, _ in cols) + "</tr>"
    head += "<tr>" + "<th>train</th><th>val</th>" * (len(cols) + 1) + "</tr>"
    rows = []
    for st in sorted(clean.subtask.unique()):
        tr, va = clean[(clean.subtask == st) & (clean.set == "train")], clean[(clean.subtask == st) & (clean.set == "val")]
        cells = [str(len(tr)), str(len(va))]
        for key, _, root in cols:
            f = (lambda v: np.sqrt(v)) if root else (lambda v: v)
            cells += [_fmt(f(tr[key].median())) if len(tr) else "-", _fmt(f(va[key].median())) if len(va) else "-"]
        rows.append(f"<tr><td>{st}</td>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table><p class=note>medians over anchors.</p>"


def table_noise(df: pd.DataFrame) -> str:
    settings = [s for s in df.setting.unique() if s != "clean"]
    if not settings:
        return "<p>no noise settings.</p>"
    head = "<tr><th rowspan=2>setting</th><th colspan=3>rms shift of the mean, train</th><th colspan=3>rms shift, val</th><th colspan=3>spread rel. to clean, train</th><th colspan=3>spread rel., val</th></tr>"
    head += "<tr>" + "".join(f"<th>{c}</th>" for c in COMPONENTS * 4) + "</tr>"
    rows = []
    for setting in settings:
        cells = []
        for key in ("shift", "spread_rel"):
            for s in ("train", "val"):
                sub = df[(df.setting == setting) & (df.set == s)]
                cells += [_q(sub[f"{key}_{c}"], key == "shift") for c in COMPONENTS]
        rows.append(f"<tr><td>{setting}</td>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def table_suspects(clean: pd.DataFrame, n: int = 12) -> str:
    tr = clean[clean.set == "train"].nsmallest(n, "spread_ratio_total")
    head = "<tr><th>episode</th><th>t (s)</th><th>frame</th><th>subtask</th><th>spread std / data std</th><th>style proj offset</th><th>style proj shape</th><th>rms bias</th><th>rms demo</th></tr>"
    rows = [f"<tr><td>{r.episode}</td><td>{r.time_s:.0f}</td><td>{getattr(r, 'frame', '')}</td><td>{r.subtask}</td><td>{_fmt(np.sqrt(r.spread_ratio_total))}</td>"
            f"<td>{_fmt(r.style_proj_offset)}</td><td>{_fmt(r.style_proj_shape)}</td><td>{_fmt(np.sqrt(r.bias2_total))}</td><td>{_fmt(np.sqrt(r.hold_total))}</td></tr>"
            for r in tr.itertuples()]
    return f"<table>{head}{''.join(rows)}</table>"


def _bands(ax, ep: pd.DataFrame, labels: bool):
    ymin, ymax = ax.get_ylim()
    change = [0] + [i for i in range(1, len(ep)) if ep.subtask.iloc[i] != ep.subtask.iloc[i - 1]] + [len(ep)]
    for j, (a, b) in enumerate(zip(change[:-1], change[1:])):
        t0, t1 = ep.time_s.iloc[a], ep.time_s.iloc[b - 1] + 1
        if j % 2:
            ax.axvspan(t0, t1, color="#f1f0eb", zorder=0)
        if labels:
            ax.text((t0 + t1) / 2, ymax * 1.02, ep.subtask.iloc[a], ha="center", va="bottom", fontsize=7, color=MUTED, rotation=90)


def fig_timeline(clean: pd.DataFrame, episodes: dict[str, str]) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col", gridspec_kw={"hspace": 0.35})
    for col, s in enumerate(("train", "val")):
        if episodes.get(s) is None:
            for ax in axes[:, col]:
                ax.axis("off")
            continue
        ep = clean[(clean.set == s) & (clean.episode == episodes[s])].sort_values("time_s")
        c = COLOR[s]
        ax = axes[0, col]
        ax.fill_between(ep.time_s, 0, np.sqrt(ep.hold_total), color=c, alpha=0.25, step="mid")
        ax.plot(ep.time_s, np.sqrt(ep.hold_total), color=c, lw=1.2, drawstyle="steps-mid")
        ax.set_title(f"{s}: episode {episodes[s]}", loc="left", color=INK, pad=28)
        ax.set_ylabel("rms demo (motion size)")
        ax = axes[1, col]
        ax.plot(ep.time_s, np.sqrt(ep.bias2_offset), color=c, lw=1.6, label="offset")
        ax.plot(ep.time_s, np.sqrt(ep.bias2_shape), color=c, lw=1.6, ls="--", label="shape")
        ax.set_ylabel("rms bias of the mean")
        ax.set_ylim(0, 1.3 * max(np.sqrt(ep.bias2_offset).max(), np.sqrt(ep.bias2_shape).max()))
        ax.legend(loc="upper center", ncol=2, fontsize=8)
        ax = axes[2, col]
        ax.plot(ep.time_s, np.sqrt(ep.spread_offset), color=c, lw=1.6, label="offset")
        ax.plot(ep.time_s, np.sqrt(ep.spread_shape), color=c, lw=1.6, ls="--", label="shape")
        ax.plot(ep.time_s, np.sqrt(ep.var_data_total), color=MUTED, lw=1.0, ls=":", label="data std (reference)")
        ax.set_ylabel("rms spread of the draws")
        ax.set_xlabel("time in episode (s)")
        ax.set_ylim(0, 1.3 * max(np.sqrt(ep.spread_shape).max(), np.sqrt(ep.var_data_total).max()))
        ax.legend(loc="upper center", ncol=3, fontsize=8)
        for row, ax in enumerate(axes[:, col]):
            if row == 0:
                ax.set_ylim(bottom=0)
            _bands(ax, ep, labels=row == 0)
    return _png(fig)


def _scatter(ax, df, x, y, diag=False, root=True):
    f = np.sqrt if root else (lambda v: v)
    for s in ("train", "val"):
        sub = df[df.set == s]
        ax.scatter(f(sub[x]), f(sub[y]), s=9, color=COLOR[s], alpha=0.55, label=s, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if diag:
        lo = f(min(df[x].min(), df[y].min()))
        hi = f(max(df[x].max(), df[y].max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    ax.legend(loc="upper left", markerscale=2)


def fig_scatters(clean: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"wspace": 0.3})
    _scatter(axes[0], clean, "var_data_total", "spread_total", diag=True)
    axes[0].set_xlabel("data std (cross-episode, bucket)")
    axes[0].set_ylabel("rms spread of the draws")
    axes[0].set_title("calibration (diagonal = matches the data)", loc="left")
    _scatter(axes[1], clean, "hold_total", "mse_total")
    axes[1].set_xlabel("rms demo (motion size)")
    axes[1].set_ylabel("rms error of the draws")
    axes[1].set_title("raw error vs motion size", loc="left")
    _scatter(axes[2], clean, "hold_total", "R_total", root=False)
    axes[2].axhline(1.0, color=MUTED, lw=0.8, ls="--")
    axes[2].set_xlabel("msq demo (motion size)")
    axes[2].set_ylabel("R = mse / hold")
    axes[2].set_title("R (1 = holding still)", loc="left")
    return _png(fig)


def fig_style(clean: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, comp in zip(axes, ("offset", "shape")):
        bins = np.linspace(-1, 2, 46)
        for s in ("train", "val"):
            v = clean[clean.set == s][f"style_proj_{comp}"].clip(-1, 2)
            ax.hist(v, bins=bins, color=COLOR[s], alpha=0.55, label=f"{s} (median {v.median():.2f})")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.axvline(1, color=MUTED, lw=0.8, ls="--")
        ax.set_title(f"style projection, {comp}", loc="left")
        ax.set_xlabel("projection coefficient (clipped to [-1, 2])")
        ax.legend()
    return _png(fig)


def fig_noise(df: pd.DataFrame) -> str:
    settings = [s for s in df.setting.unique() if s != "clean"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, label in ((axes[0], "shift_total", "rms shift of the mean"), (axes[1], "spread_rel_total", "spread relative to clean")):
        for i, setting in enumerate(settings):
            for j, s in enumerate(("train", "val")):
                v = df[(df.setting == setting) & (df.set == s)][key].dropna()
                if key == "shift_total":
                    v = np.sqrt(v)
                pos = i * 3 + j
                ax.boxplot(v, positions=[pos], widths=0.7, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=COLOR[s], alpha=0.55, edgecolor=COLOR[s]),
                           medianprops=dict(color=INK), whiskerprops=dict(color=COLOR[s]), capprops=dict(color=COLOR[s]))
        ax.set_xticks([i * 3 + 0.5 for i in range(len(settings))])
        ax.set_xticklabels(settings)
        ax.set_ylabel(label)
        if key == "spread_rel_total":
            ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    axes[0].set_title("shift of the mean under input noise", loc="left")
    axes[1].set_title("spread under input noise, relative to clean", loc="left")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR[s], alpha=0.55, label=s) for s in ("train", "val")], loc="upper right")
    return _png(fig)


CSS = """
body{font-family:system-ui,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#f9f9f7;font-size:14px}
h1{font-size:20px}h2{font-size:16px;margin-top:32px}table{border-collapse:collapse;margin:8px 0;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #e1e0d9;padding:4px 8px;text-align:right;vertical-align:top}th{color:#52514e;font-weight:600;text-align:center}
td:first-child,th:first-child{text-align:left}.iqr{color:#898781;font-size:12px}.desc{color:#898781;font-size:12px}.note{color:#898781;font-size:12px}
img{max-width:100%;display:block;margin:8px 0}.banner{background:#fff3cd;border:1px solid #eda100;padding:8px 12px;border-radius:4px}
"""


def fig_timelines(clean: pd.DataFrame, example_episodes: dict[str, str] | None) -> str:
    """One two-column figure per pair of (train, val) episodes; all episodes when none are named."""
    if example_episodes is not None:
        return f"<img src='{fig_timeline(clean, example_episodes)}'>"
    train_eps = list(dict.fromkeys(clean[clean.set == "train"].sort_values(["episode", "time_s"]).episode))
    val_eps = list(dict.fromkeys(clean[clean.set == "val"].sort_values(["episode", "time_s"]).episode))
    parts = []
    for i in range(max(len(train_eps), len(val_eps))):
        pair = {"train": train_eps[i % len(train_eps)] if train_eps else None, "val": val_eps[i % len(val_eps)] if val_eps else None}
        parts.append(f"<img src='{fig_timeline(clean, pair)}'>")
    return "\n".join(parts)


def write_report(df: pd.DataFrame, path: str, title: str, preamble: str, example_episodes: dict[str, str] | None) -> None:
    clean = df[df.setting == "clean"]
    parts = [f"<!doctype html><html><head><meta charset=utf-8><title>{title}</title><style>{CSS}</style></head><body>",
             f"<h1>{title}</h1>", preamble,
             "<h2>1. Train vs val, clean inputs</h2>", table_headline(clean),
             "<h2>2. Along the episode</h2>",
             "<p class=note>train left, val right; bands = subtask segments; dotted gray = the cross-episode std of the demo at the reference neighbours.</p>",
             fig_timelines(clean, example_episodes),
             "<h2>3. Calibration and scale</h2>", f"<img src='{fig_scatters(clean)}'>",
             "<h2>4. Style reproduction</h2>",
             "<p class=note>per anchor: the demo's deviation from the cross-episode bucket mean (delta) and the draw mean's deviation (rho); shown is &lt;rho, delta&gt; / &lt;delta, delta&gt;. "
             "A model that outputs the cross-episode mean scores 0; one that reproduces this episode's own deviation scores 1. Part of delta is explained by the state inside the coarse bucket, so val is not expected at exactly 0: the train minus val gap is the number.</p>",
             f"<img src='{fig_style(clean)}'>",
             "<h2>5. By subtask</h2>", table_by_subtask(clean),
             "<h2>6. Input noise</h2>", table_noise(df), f"<img src='{fig_noise(df)}'>",
             "<h2>7. Train anchors with the smallest spread ratio</h2>",
             "<p class=note>where to look first for collapsed draws; the real run links each row to the frame.</p>", table_suspects(clean),
             "</body></html>"]
    with open(path, "w") as f:
        f.write("\n".join(parts))
