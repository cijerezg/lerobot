"""Built corrections in the Action Inspector page: black = recorded chunk, blue = correction.

  uv run python -m lerobot.annotation.corrections.inspector <work> --source NAME [--episode N]
      every built correction of that source (or of one episode) -> <work>/inspector/<NAME>[_ep<NNN>].html
  uv run python -m lerobot.annotation.corrections.inspector <work> --flagged [--random N]
      the rows that fail the check or are flagged inconsistent, plus N random others -> <work>/inspector/flagged.html

Reads ``<work>/corrections.parquet``, ``<work>/<source>/action_trace_records.npz`` and the frame folders'
``waypoints.json`` (``build``). One slider step per frame; the page is ``lerobot.probes.action_trace_probe``'s.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from lerobot.annotation.corrections.arm import FPS
from lerobot.probes.action_trace_probe import (
    _analyse,
    _figure,
    _image_data_uri,
    _resolve_actuator_config,
    _write_dashboard_html,
)
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--flagged", action="store_true")
    parser.add_argument("--random", type=int, default=0)
    args = parser.parse_args()
    table = pd.read_parquet(args.work / "corrections.parquet")
    table = table[table.error == ""]
    if args.flagged:
        flagged = table[~table.ok | table.inconsistent]
        rest = table.drop(flagged.index)
        table, name = pd.concat([flagged, rest.sample(min(args.random, len(rest)), random_state=0)]), "flagged"
    else:
        table, name = table[table.source == args.source], args.source
        if args.episode is not None:
            table, name = table[table.episode == args.episode], f"{name}_ep{args.episode:03d}"

    kin = RebotKinematics()
    speeds, lower, upper = _resolve_actuator_config(None, 7)
    records = []
    for source, rows in table.groupby("source", sort=False):
        d = dict(np.load(args.work / source / "action_trace_records.npz"))
        at = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(d["episode"], d["frame"], strict=True))}
        for r in rows.sort_values(["episode", "frame"]).itertuples():
            i = at[(r.episode, r.frame)]
            spec = json.loads((args.work / source / f"ep{r.episode:03d}" / f"f{r.frame:06d}" / "waypoints.json").read_text())
            record = _analyse(
                kin, d["state"][i], d["gt_chunk"][i], list(d["sample_chunks"][i]), 0.0,
                fps=float(FPS), motor_speeds_deg_s=speeds, joint_lower=lower, joint_upper=upper,
            )
            kind = f"demo grade {r.grade}" if r.kind == "demo" else r.kind
            flags = ("" if r.ok else " · FAILS THE CHECK") + (" · inconsistent tip" if r.inconsistent else "")
            record.update(
                episode=r.episode, frame=r.frame, global_idx=int(d["global_idx"][i]), subtask=r.subtask,
                trace_label=f"{source} ep{r.episode} · frame {r.frame} · {kind}{flags}",
                sample_names=["correction"],
                cameras=[{"label": cam, "src": _image_data_uri(torch.from_numpy(d[f"image_{cam}"][i]))} for cam in ("top", "wrist")],
                notes=f"contact: {spec.get('contact')}\nintent: {spec.get('intent')}\nwrong: {spec.get('wrong')}\nplan: {spec.get('note')}\n"
                f"travel: recorded {r.recorded_travel_deg:.0f} deg, corrected {r.corrected_travel_deg:.0f} deg; "
                f"contact-pose tip spread in this episode and subtask {100 * r.tip_spread_m:.1f} cm",
            )
            records.append(record)

    fig = _figure(records, SimpleNamespace(trace_table_z=0.0, trace_clearance_warn_m=0.01), fps=float(FPS))
    for trace in [*fig.data, *(t for f in fig.frames for t in f.data)]:
        if trace.name == "ground truth targets":
            trace.name = "recorded chunk"
    out = args.work / "inspector"
    out.mkdir(exist_ok=True)
    _write_dashboard_html(
        fig, records, str(out / f"{name}.html"),
        page_title="Correction Inspector",
        subtitle="One labelled frame per step: the 1 s chunk that was recorded against the correction built from the "
        "labeller's waypoints. The correction is Butterworth-filtered like a deployed chunk.",
        legend_note="Black = recorded chunk · blue = correction. Solid RGB axes = recorded terminal tool "
        "orientation · dotted RGB axes = correction.",
        contexts=[
            {"label": r["trace_label"], "subtask": r["subtask"], "cameras": r["cameras"], "notes": r["notes"]} for r in records
        ],
    )
    print("wrote", out / f"{name}.html", len(records), "frames")


if __name__ == "__main__":
    main()
