"""Aggregate the readouts into coverage plots and an empty-cell list. Run from the workspace root:

    uv run python lerobot/adhocs/object_coverage/aggregate.py

Writes coverage_<object>.png, qa_<object>.png and coverage.md next to this file.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

D = Path("lerobot/adhocs/object_coverage")
COLS, ROWS = "ABCDEF", 4
FACING = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ANGLE_BINS = np.arange(0, 181, 30)  # 30-degree bins over [0, 180)
BLUE = "#2f6db5"


def load():
    m = pd.read_csv(D / "manifest.csv")
    first = pd.concat(pd.read_csv(p, dtype=str) for p in sorted((D / "readouts").glob("*.csv")) if not p.name.startswith("redo_"))
    redo_files = sorted((D / "readouts").glob("redo_*.csv"))
    r = first.drop_duplicates("image", keep="last")
    if redo_files:
        # second read on a frame 3 s earlier for rows where the gripper hid the object; it wins when it found the object
        redo = pd.concat(pd.read_csv(p, dtype=str) for p in redo_files)
        back = pd.read_csv(D / "batches/redo_all.csv")[["image", "orig_image"]]
        redo = redo.merge(back, on="image").drop(columns="image").rename(columns={"orig_image": "image"})
        good = redo.cell.notna() & ~redo.note.fillna("").str.contains("occlu|not found", case=False)
        redo = redo[good]
        r = pd.concat([r[~r.image.isin(redo.image)], redo])
    df = m.merge(r, on="image", how="left")
    df["cell"] = df.cell.str.strip().str.upper()
    df["angle"] = pd.to_numeric(df.angle, errors="coerce")
    df["facing"] = df.facing.str.strip().str.upper()
    df["state"] = df.state.str.strip().str.lower()
    df["extra_cell"] = df.extra_cell.str.strip().str.upper()
    ok = df.cell.str.match(r"^[A-F][1-4]$", na=False)
    df.loc[~ok, "cell"] = np.nan
    return df


def cell_grid(cells):
    g = np.zeros((ROWS, len(COLS)), int)
    for c in cells.dropna():
        g[int(c[1]) - 1, COLS.index(c[0])] += 1
    return g


def draw_cells(ax, g, title):
    ax.imshow(g, cmap="Blues", vmin=0, vmax=max(1, g.max()))
    for j in range(ROWS):
        for i in range(len(COLS)):
            ax.text(i, j, str(g[j, i]) if g[j, i] else "", ha="center", va="center",
                    color="white" if g[j, i] > 0.6 * max(1, g.max()) else "#222", fontsize=10)
    ax.set_xticks(range(len(COLS)), list(COLS)); ax.set_yticks(range(ROWS), [str(j + 1) for j in range(ROWS)])
    ax.set_title(title, fontsize=10); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(2.5, -0.9, "robot side", ha="center", fontsize=8, color="#666")


def plot_object(df, obj):
    d = df[df.object == obj]
    n = len(d); found = d.cell.notna().sum()
    has_facing = d.facing.isin(FACING).any()
    fig = plt.figure(figsize=(18, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.4, 1.6, 1.2, 1, 0.9])
    ax0 = fig.add_subplot(gs[0]); draw_cells(ax0, cell_grid(d.cell), f"{obj}: position, {found}/{n} read")
    # one short oriented segment per observation, jittered inside its cell
    ax1 = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    for _, r in d.dropna(subset=["cell"]).iterrows():
        cx, cy = COLS.index(r.cell[0]) + 0.5 + rng.uniform(-0.3, 0.3), int(r.cell[1]) - 0.5 + rng.uniform(-0.3, 0.3)
        if pd.isna(r.angle):
            ax1.plot(cx, cy, "o", color=BLUE, ms=4)
        else:
            t = np.deg2rad(r.angle); dx, dy = 0.22 * np.cos(t), -0.22 * np.sin(t)  # image y points down
            ax1.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=BLUE, lw=2, solid_capstyle="round")
    ax1.set_xlim(0, len(COLS)); ax1.set_ylim(ROWS, 0); ax1.set_aspect("equal")
    ax1.set_xticks(np.arange(len(COLS)) + 0.5, list(COLS)); ax1.set_yticks(np.arange(ROWS) + 0.5, [str(j + 1) for j in range(ROWS)])
    ax1.set_xticks(range(len(COLS) + 1), minor=True); ax1.set_yticks(range(ROWS + 1), minor=True)
    ax1.grid(which="minor", color="#ccc", lw=0.8); ax1.tick_params(length=0, which="both")
    ax1.set_title("position + long-axis angle", fontsize=10)
    for sp in ax1.spines.values(): sp.set_visible(False)
    # angle histogram, bins centred on 0, 30, ..., 150 so near-horizontal is one bin
    ax2 = fig.add_subplot(gs[2])
    a = (d.angle.dropna() + 15) % 180 - 15
    h, _ = np.histogram(a, bins=ANGLE_BINS - 15)
    ax2.bar(ANGLE_BINS[:-1], h, width=26, color=BLUE)
    ax2.set_xticks(ANGLE_BINS[:-1]); ax2.set_xlabel("long-axis angle (0 = along table, 90 = toward robot)", fontsize=8)
    ax2.set_title("orientation (bin centre, +-15)", fontsize=10)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    # facing rose
    ax3 = fig.add_subplot(gs[3], projection="polar")
    if has_facing:
        f = d.facing[d.facing.isin(FACING)]
        cnt = [int((f == k).sum()) for k in FACING]
        th = np.deg2rad(45 * np.arange(8))
        ax3.bar(th, cnt, width=np.deg2rad(40), color=BLUE)
        ax3.set_theta_zero_location("N"); ax3.set_theta_direction(-1)
        ax3.set_xticks(th, FACING); ax3.set_yticks([]); ax3.set_title("facing (N = robot)", fontsize=10)
    else:
        ax3.axis("off"); ax3.set_title("facing: n/a", fontsize=10)
    ax4 = fig.add_subplot(gs[4]); ax4.axis("off")
    txt = ["state:"] + [f"  {k}: {v}" for k, v in d.state.value_counts(dropna=False).items()]
    if d.extra_cell.notna().any():
        txt += ["", "destination cell:"] + [f"  {k}: {v}" for k, v in d.extra_cell.value_counts().head(8).items()]
    txt += ["", "notes:"] + [f"  {k}: {v}" for k, v in d.note.fillna("").str.extract(r"(occluded|not found)")[0].value_counts().items()]
    txt += ["", f"unsure: {(d.confidence == 'unsure').sum()}/{n}"]
    ax4.text(0, 1, "\n".join(txt), va="top", fontsize=9, family="monospace")
    fig.tight_layout(); fig.savefig(D / f"coverage_{obj}.png", dpi=110); plt.close(fig)
    return cell_grid(d.cell), h


def qa_sheet(df, obj, per_row=5, thumb=320):
    d = df[df.object == obj].reset_index(drop=True)
    rows = int(np.ceil(len(d) / per_row)); th = int(thumb * 0.75)
    sheet = Image.new("RGB", (per_row * thumb, rows * (th + 28)), "white")
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for i, r in d.iterrows():
        im = Image.open(D / r.image).resize((thumb, th))
        x, y = (i % per_row) * thumb, (i // per_row) * (th + 28)
        sheet.paste(im, (x, y))
        lab = f"{r.root[:18]} ep{r.episode} | {r.cell} {'' if pd.isna(r.angle) else int(r.angle)} {r.facing if isinstance(r.facing, str) else ''} {r.state if isinstance(r.state, str) else ''}"
        dr.text((x + 3, y + th + 2), lab[:60], fill="black")
        dr.text((x + 3, y + th + 14), str(r.clause)[:60], fill="#555")
    sheet.save(D / f"qa_{obj}.png")


def main():
    df = load()
    lines = ["# Object coverage (internal ReBot roots, first top frame of each grasp segment)", ""]
    for obj in ["nail_clipper", "large_clipper", "headphone_case", "bit_kit", "bit"]:
        g, h = plot_object(df, obj); qa_sheet(df, obj)
        d = df[df.object == obj]
        empty = [f"{COLS[i]}{j + 1}" for j in range(ROWS) for i in range(len(COLS)) if g[j, i] == 0]
        thin = [f"{COLS[i]}{j + 1}({g[j, i]})" for j in range(ROWS) for i in range(len(COLS)) if 0 < g[j, i] <= 2]
        lines += [f"## {obj}", "", f"images {len(d)}, read {d.cell.notna().sum()}", "",
                  "| angle bin | count |", "|---|---|"] + [f"| {ANGLE_BINS[k]} +-15 | {h[k]} |" for k in range(len(h))]
        lines += ["", f"empty cells: {', '.join(empty) or 'none'}", f"thin cells (<=2): {', '.join(thin) or 'none'}", ""]
    (D / "coverage.md").write_text("\n".join(lines))
    df.to_csv(D / "readouts_merged.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
