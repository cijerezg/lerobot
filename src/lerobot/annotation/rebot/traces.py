"""Per-episode trace PNG + text event list of a pass: <work>/review/trace_<IDX>.png, events_<IDX>.txt.

    uv run python -m lerobot.annotation.rebot.traces <work> [IDX ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lerobot.annotation.rebot.episode import interventions, records, states  # noqa: E402
from lerobot.annotation.rebot.motion import approach_angle, closed_intervals, runs_of, still_runs  # noqa: E402

NAMES = ["pan", "lift", "elbow", "wflex", "wyaw", "wroll", "grip"]


def trace(work, k):
    r = records(work)[k]
    s, a = states(r)
    n = len(s)
    iv = interventions(r)
    ivs = runs_of(iv) if iv is not None else []
    closes, stills = closed_intervals(s[:, 6]), still_runs(s)
    rows = int(np.ceil(n / 3000))
    fig, axes = plt.subplots(rows, 1, figsize=(22, 3.2 * rows), squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        lo, hi = j * 3000, min(n, (j + 1) * 3000)
        x = np.arange(lo, hi)
        for c in range(6):
            ax.plot(x, s[lo:hi, c], lw=0.7, label=NAMES[c])
        ax.plot(x, s[lo:hi, 6], "k", lw=1.6, label="grip state")
        ax.plot(x, a[lo:hi, 6], "r--", lw=0.8, label="grip action")
        for p, q in ivs:
            ax.axvspan(p, q, color="orange", alpha=0.15)
        for p, q in stills:
            ax.axvspan(p, q, color="grey", alpha=0.3)
        ax.set_xticks(np.arange(lo, hi + 1, 150))
        ax.tick_params(axis="x", labelsize=6, rotation=90)
        ax.grid(alpha=0.3)
        ax.set_xlim(lo, lo + 3000)
    axes[0, 0].legend(ncol=9, fontsize=7)
    axes[0, 0].set_title(
        f"{k:02d} {r['key']}  (grip 0 = shut, negative = open; orange = TELEOP, grey = still >= 1 s)"
    )
    review = Path(work) / "review"
    review.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(review / f"trace_{k:02d}.png", dpi=80)
    plt.close(fig)

    def spans(xs):
        return ", ".join(f"[{p},{q}) {(q - p) / 30:.1f}s" for p, q in xs)

    teleop = ", ".join(f"[{p},{q})" for p, q in ivs) if iv is not None else "n/a (teleop recording)"
    lines = [
        f"{k:02d} {r['key']} frames={n} fps=30",
        "closed intervals (state gripper, relative-travel 25 deg): " + spans(closes),
        "FK approach angle at each close start (< 45 top-pinch, >= 45 side-pinch): "
        + ", ".join(f"f{p} {approach_angle(s[p]):.0f} deg" for p, _ in closes),
        "still runs >= 1 s: " + (spans(stills) or "none"),
        "teleop (is_intervention) runs: " + teleop,
    ]
    (review / f"events_{k:02d}.txt").write_text("\n".join(lines) + "\n")
    print(lines[0], len(closes), "closes", len(stills), "still runs")


if __name__ == "__main__":
    work = sys.argv[1]
    for k in map(int, sys.argv[2:]) if len(sys.argv) > 2 else range(len(records(work))):
        trace(work, k)
