#!/usr/bin/env python
"""Compare two sim-eval runs in one rerun viewer: run A on the top row, run B on the bottom, one shared time axis.

Each run is the rerun.rrd that run_sim_eval.sh writes per layout. Both recordings are re-logged into one
recording under ``/A/...`` and ``/B/...`` so the viewer scrubs them together, then the viewer opens with a
blueprint of two rows: wrist camera, top camera, wrist depth, joint traces, subtask events.

Examples::

    uv run python -m lerobot.scripts.compare_sim \\
        outputs/rebot-stratified_sampling-20261016/checkpoints/000400 \\
        outputs/rebot-stratified_sampling-20261016/checkpoints/001200
    uv run python -m lerobot.scripts.compare_sim RUN_A/validation/step_00001200 RUN_B/validation/step_00001200 --layout nail_ep2
    uv run python -m lerobot.scripts.compare_sim a/rerun.rrd b/rerun.rrd --label old --label new --save compare.rrd

A path may be a checkpoint step dir (``.../checkpoints/000400``), a ``validation/step_XXXXXXXX`` dir, a ``sim_nail``
dir, a layout dir or a ``.rrd`` file; ``--layout`` picks the layout for everything above a layout dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
import rerun.blueprint as rrb


class _Batch:
    """Flat arrow values + descriptor, the ComponentBatchLike shape ``rr.ComponentColumn`` expects."""

    def __init__(self, descriptor: rr.ComponentDescriptor, array: pa.Array) -> None:
        self.descriptor = descriptor
        self.array = array

    def component_descriptor(self) -> rr.ComponentDescriptor:
        return self.descriptor

    def as_arrow_array(self) -> pa.Array:
        return self.array


def resolve_rrd(path: Path, layout: str) -> Path:
    """Turn any of the accepted inputs into the layout's rerun.rrd."""
    if path.is_file():
        return path
    if (path / "rerun.rrd").is_file():
        return path / "rerun.rrd"
    if path.parent.name == "checkpoints" and path.name.isdigit():
        path = path.parent.parent / "validation" / f"step_{int(path.name):08d}"
    if (path / "sim_nail").is_dir():
        path = path / "sim_nail"
    rrd = path / layout / "rerun.rrd"
    if not rrd.is_file():
        raise SystemExit(f"no sim recording at {rrd}")
    return rrd


def default_label(rrd: Path) -> str:
    """``<run>-<step>`` for ``<run>/validation/step_X/sim_nail/<layout>/rerun.rrd``, else the folder name."""
    parts = rrd.resolve().parts
    if "validation" in parts:
        i = parts.index("validation")
        return f"{parts[i - 1]}-{parts[i + 1].split('_')[-1].lstrip('0') or '0'}"
    return rrd.parent.name


def relog(rrd: Path, prefix: str) -> None:
    """Copy every entity of ``rrd`` into the current recording under ``/<prefix>``, columns passed through as-is."""
    rec = rr.dataframe.load_recording(str(rrd))
    columns = [c for c in rec.schema().component_columns() if c.entity_path != "/__properties"]
    for entity in sorted({c.entity_path for c in columns}):
        table = rec.view(index="time", contents=entity).select().read_all()
        times = table.column("time").to_numpy(zero_copy_only=False)
        batches = []
        for c in columns:
            if c.entity_path != entity:
                continue
            array = table.column(f"{entity}:{c.component}").combine_chunks()
            lengths = pc.fill_null(pc.list_value_length(array), 0).to_numpy()
            descriptor = rr.ComponentDescriptor(
                c.component, archetype=c.archetype, component_type=c.component_type
            )
            batches.append(
                rr.ComponentColumn(descriptor, _Batch(descriptor, array.flatten()), lengths=lengths)
            )
        rr.send_columns(
            f"/{prefix}{entity}", indexes=[rr.TimeColumn("time", duration=times)], columns=batches
        )


def log_events(rrd: Path, prefix: str, label: str) -> None:
    """Subtask events as a text log at their time, plus the per-episode outcome as a static document."""
    events_path = rrd.parent / "subtask_events.json"
    lines = [f"# {label}", f"`{rrd}`", ""]
    if events_path.is_file():
        for ep in json.loads(events_path.read_text()):
            furthest = max([e["cursor"] for e in ep["events"]], default=0)
            lines.append(f"- episode {ep['episode']}: success {ep['success']}, steps done {furthest}")
            for e in ep["events"]:
                rr.set_time("time", duration=float(e["t"]))
                rr.log(f"/{prefix}/events", rr.TextLog(f"ep{ep['episode']} step {e['cursor']}: {e['why']}"))
    else:
        lines.append("no subtask_events.json")
    rr.log(
        f"/{prefix}/summary", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True
    )


def row(prefix: str, label: str) -> rrb.Container:
    p = f"/{prefix}"
    return rrb.Horizontal(
        rrb.Spatial2DView(origin=f"{p}/observation.wrist_0", name=f"{label} wrist"),
        rrb.Spatial2DView(origin=f"{p}/observation.external_0", name=f"{label} top"),
        rrb.Spatial2DView(origin=f"{p}/observation.wrist_0.depth", name=f"{label} depth"),
        rrb.TimeSeriesView(origin=p, name=f"{label} joints", contents="$origin/**"),
        rrb.Vertical(
            rrb.TextDocumentView(origin=f"{p}/summary", name=f"{label} outcome"),
            rrb.TextLogView(origin=f"{p}/events", name=f"{label} events"),
        ),
        column_shares=[3, 3, 2, 4, 2],
        name=label,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--layout", default="nail_ep5", help="layout folder under sim_nail (default nail_ep5)")
    ap.add_argument(
        "--label", action="append", default=[], help="row label, once per run (default <run>-<step>)"
    )
    ap.add_argument(
        "--save", type=Path, default=None, help="write the merged .rrd here instead of opening the viewer"
    )
    args = ap.parse_args()

    rrds = [resolve_rrd(args.run_a, args.layout), resolve_rrd(args.run_b, args.layout)]
    labels = args.label + [default_label(r) for r in rrds[len(args.label) :]]

    rr.init("compare_sim")
    if args.save is not None:
        rr.save(str(args.save))
    else:
        rr.spawn()
    rr.send_blueprint(
        rrb.Blueprint(rrb.Vertical(row("A", labels[0]), row("B", labels[1])), collapse_panels=True)
    )
    for prefix, rrd, label in zip(("A", "B"), rrds, labels, strict=True):
        print(f"{prefix}: {label}  {rrd}")
        relog(rrd, prefix)
        log_events(rrd, prefix, label)


if __name__ == "__main__":
    main()
