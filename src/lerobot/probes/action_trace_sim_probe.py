"""Action Inspector in sim: the action_trace probe's flow samples played on the MuJoCo reBot, one video per anchor.

Reads ``action_trace/action_trace_records.npz`` from the same validation step (so
``enable_action_trace`` must be on; this probe runs right after it) and renders it with
``lerobot.probes.action_trace_sim`` in its own process, so mujoco, EGL and av never enter
the training process. ``probe_parameters.trace_sim_args`` passes extra flags to that CLI,
e.g. ``--time-scale 2`` (half speed) or ``--execute`` (MuJoCo physics: achieved trails,
end-effector lag, table contact).

Every video frame is one tick of the chunk: a panel per arm, the demonstrated chunk in grey
first and then each flow sample in its Action Inspector colour, each with its own
end-effector path and the demo's path for reference; the dataset's camera frames over the
same window; and the joint curves, demo against every sample, with a cursor at the current
tick. The camera is framed per anchor on that anchor's end-effector paths and holds still
through the chunk.

anchor<NNN>_ep<EEEE>_fr<FFFFFF>.mp4 — one video per anchor
summary.json — per anchor the final end-effector spread across samples and, per sample, probe clearance, final distance to the demo, path length and (``--execute``) table-contact ticks and peak lag
"""

import json
import os
import shlex
import subprocess
import sys

from lerobot.probes.manifest import Metric, Panel, write_index


def run(adapter, dataset, cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    records = os.path.join(
        os.path.dirname(os.path.normpath(output_dir)), "action_trace", "action_trace_records.npz"
    )
    if not os.path.exists(records):
        raise FileNotFoundError(f"{records} missing: action_trace_sim plays the action_trace probe's records")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.probes.action_trace_sim",
            "--records",
            records,
            "--out",
            output_dir,
            *shlex.split(cfg.probe_parameters.trace_sim_args),
        ],
        check=True,
    )
    with open(os.path.join(output_dir, "summary.json")) as f:
        summary = json.load(f)
    how = (
        "Grey arm and path: the demonstrated chunk. Coloured arms: the flow samples, one panel each, "
        "with the demo path repeated for reference. Bottom: joint commands over the chunk, cursor at the current tick."
    )
    if summary["execute"]:
        how += (
            " Physics is on: arms show what the plant did, the small sphere is the command target, "
            "the trail is the achieved path, dashed curves are the simulated joints and black is the real follower."
        )
    panels = [
        Panel(
            a["file"],
            f"anchor {a['anchor']}: episode {a['episode']} frame {a['frame']}, {a['subtask']}",
            how=how,
            primary=i < 2,
        )
        for i, a in enumerate(summary["anchors"])
    ]
    metrics = [
        Metric(
            "final_spread_mm_mean",
            "Final end-effector spread across samples, mean over anchors (mm)",
            fmt=0,
            note="Distance between the furthest two samples' chunk ends: how multimodal the policy is at these anchors, not an error.",
            primary=True,
        ),
        Metric("n_anchors", "Anchors rendered", fmt=0),
    ]
    if summary["execute"]:
        metrics += [
            Metric(
                "contact_anchor_fraction",
                "Anchors where a sample touched the table in sim",
                good="low",
                fmt=2,
                primary=True,
            ),
            Metric(
                "max_lag_mm_mean", "Peak end-effector lag behind the command, mean over samples (mm)", fmt=0
            ),
        ]
    write_index(
        output_dir,
        sys.modules[__name__],
        claim="What does the arm do with each flow sample of the Action Inspector's chunks?",
        group="Actions",
        metrics=metrics,
        panels=panels,
        summary=summary,
        see_also=["action_trace"],
        extra={"trace_sim_args": cfg.probe_parameters.trace_sim_args},
    )
