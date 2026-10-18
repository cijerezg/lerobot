#!/usr/bin/env python
"""Compare two sim-eval runs in one rerun viewer: run A on the top row, run B on the bottom, one shared time axis.

Each run is the rerun.rrd that run_sim_eval.sh writes per layout. Both recordings are re-logged into one
merged .rrd under ``/A/...`` and ``/B/...`` (default outputs/compare_sim/, ~2 GB per run), then the viewer opens
that file with a blueprint of two rows: wrist camera, top camera, wrist depth, joint traces, subtask events.

Examples::

    uv run python -m lerobot.scripts.compare_sim \\
        outputs/rebot-stratified_sampling-20261016/checkpoints/000400 \\
        outputs/rebot-stratified_sampling-20261016/checkpoints/001200
    uv run python -m lerobot.scripts.compare_sim RUN_A/validation/step_00001200 RUN_B/validation/step_00001200 --layout ep2
    uv run python -m lerobot.scripts.compare_sim a/rerun.rrd b/rerun.rrd --label old --label new --no-open

A path may be a checkpoint step dir (``.../checkpoints/000400``), a ``validation/step_XXXXXXXX`` dir, a ``sim_nail``
dir, a layout dir or a ``.rrd`` file; ``--layout`` picks the layout for everything above a layout dir.
Layouts are ep0, ep1, ep2 and ep3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


CHUNK_BYTES = 128 * 2**20


def send_chunk(
    entity: str, batches: list[pa.RecordBatch], descriptors: dict[str, rr.ComponentDescriptor]
) -> None:
    """One ``send_columns`` call (= one chunk in the file) for a run of time-sorted record batches of one entity."""
    table = pa.Table.from_batches(batches).combine_chunks()
    times = table.column("time").chunk(0).to_numpy(zero_copy_only=False)
    sent = []
    for name in table.column_names:
        if name not in descriptors:
            continue  # the index column; static or off-timeline components never make it into the schema
        array = table.column(name).chunk(0)
        lengths = pc.fill_null(pc.list_value_length(array), 0).to_numpy()
        descriptor = descriptors[name]
        sent.append(rr.ComponentColumn(descriptor, _Batch(descriptor, array.flatten()), lengths=lengths))
    if sent:
        rr.send_columns(entity, indexes=[rr.TimeColumn("time", duration=times)], columns=sent)


def relog(runs: list[tuple[Path, str]]) -> None:
    """Copy every entity of each run into the current recording under ``/<prefix>``, columns passed through as-is.

    The viewer shows chunks as it reads the file, so the file order is the fill order: small entities (joint
    scalars) first, images last, and the runs interleaved per entity so both rows fill together.
    """
    recs = [(rr.dataframe.load_recording(str(rrd)), prefix) for rrd, prefix in runs]
    columns = {
        prefix: [c for c in rec.schema().component_columns() if c.entity_path != "/__properties"]
        for rec, prefix in recs
    }
    entities = {c.entity_path for cols in columns.values() for c in cols}
    heavy = {c.entity_path for cols in columns.values() for c in cols if c.component.endswith((":blob", ":buffer"))}
    for entity in sorted(entities, key=lambda e: (e in heavy, e)):
        for rec, prefix in recs:
            relog_entity(rec, prefix, entity, [c for c in columns[prefix] if c.entity_path == entity])


def relog_entity(rec, prefix: str, entity: str, columns) -> None:
    """One entity of one run, ~CHUNK_BYTES per send_columns call."""
    if not columns:
        return
    descriptors = {
        f"{entity}:{c.component}": rr.ComponentDescriptor(
            c.component, archetype=c.archetype, component_type=c.component_type
        )
        for c in columns
    }
    # select() yields one record batch per time value. Sent one by one, a layout became 245 000 one-row chunks
    # (330 MB of repeated schema) that the viewer ingests and re-compacts for minutes, filling the rows
    # one entity at a time. Gather ~CHUNK_BYTES per chunk instead; one chunk per entity would be simpler but a
    # layout's JPEG column (~2 GB) overflows the 32-bit offsets of one arrow list array.
    pending: list[pa.RecordBatch] = []
    nbytes = 0
    for batch in rec.view(index="time", contents=entity).select():
        pending.append(batch)
        nbytes += batch.nbytes
        if nbytes >= CHUNK_BYTES:
            send_chunk(f"/{prefix}{entity}", pending, descriptors)
            pending, nbytes = [], 0
    if pending:
        send_chunk(f"/{prefix}{entity}", pending, descriptors)


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
        rrb.Spatial2DView(origin=f"{p}/observation.wrist_0", name=f"wrist {label}"),
        rrb.Spatial2DView(origin=f"{p}/observation.external_0", name=f"top {label}"),
        rrb.Spatial2DView(origin=f"{p}/observation.wrist_0.depth", name=f"depth {label}"),
        rrb.TimeSeriesView(origin=p, name=f"joints {label}", contents="$origin/**"),
        rrb.Vertical(
            rrb.TextDocumentView(origin=f"{p}/summary", name=f"outcome {label}"),
            rrb.TextLogView(origin=f"{p}/events", name=f"events {label}"),
        ),
        column_shares=[3, 3, 2, 4, 2],
        name=label,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--layout", default="ep0", help="layout folder under sim_nail (default ep0)")
    ap.add_argument(
        "--label", action="append", default=[], help="row label, once per run (default <run>-<step>)"
    )
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="merged .rrd path (default outputs/compare_sim/<a>_vs_<b>_<layout>.rrd)",
    )
    ap.add_argument(
        "--no-open", action="store_true", help="only write the merged .rrd, do not open the viewer"
    )
    args = ap.parse_args()

    rrds = [resolve_rrd(args.run_a, args.layout), resolve_rrd(args.run_b, args.layout)]
    labels = args.label + [default_label(r) for r in rrds[len(args.label) :]]
    out = args.save or Path("outputs/compare_sim") / f"{labels[0]}_vs_{labels[1]}_{args.layout}.rrd"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build the whole merged recording first, then open it: streaming into a live viewer makes it follow the
    # newest timestamp while loading (looks like fast-forward, rows appear one entity at a time).
    rr.init("compare_sim")
    rr.save(str(out))
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(row("A", labels[0]), row("B", labels[1])),
            rrb.TimePanel(timeline="time", duration_cursor=0.0, playback_speed=1.0),
            collapse_panels=True,
        )
    )
    # Outcomes and events first, so their panels are filled while the viewer is still loading the images.
    for prefix, rrd, label in zip(("A", "B"), rrds, labels, strict=True):
        print(f"{prefix}: {label}  {rrd}")
        log_events(rrd, prefix, label)
    relog(list(zip(rrds, ("A", "B"), strict=True)))
    rr.disconnect()
    print(f"merged: {out} ({out.stat().st_size / 1e9:.1f} GB)")
    if not args.no_open:
        # --port 0: a viewer already listening on the default port (e.g. one left open by rr.spawn) would
        # otherwise be sent the file as a live stream instead of a new viewer opening it.
        os.execvp(sys.executable, [sys.executable, "-m", "rerun", "--port", "0", str(out)])


if __name__ == "__main__":
    main()
