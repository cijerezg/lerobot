"""Find and render parent-terminal quality-1 idle tails in a diverse store.

The scan is metadata-only.  ``render`` decodes source videos and is therefore run
on Spark for the sampled-audit workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from lerobot.annotation.diverse_audit_render import decode_common, fit_tile, font
from lerobot.annotation import diverse_fix_normalizer_v2 as fix_normalizer
from lerobot.annotation import diverse_semantic_sweep_compiler as semantic_compiler


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scan(root: Path) -> list[dict[str, Any]]:
    store = root / "corpus"
    parents = {
        (str(row["episode_id"]), int(row["interval_index"])): row
        for row in _read_jsonl(store / "critic_intervals.jsonl")
    }
    quality = _read_jsonl(store / "quality_spans.jsonl")
    by_atom: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in quality:
        by_atom[(str(row["episode_id"]), int(row["parent_interval_index"]), int(row["atom_index"]))].append(row)

    output = []
    for row in quality:
        if row.get("cause") != "idle" or int(row.get("quality", 0)) != 1:
            continue
        episode = str(row["episode_id"])
        parent_index = int(row["parent_interval_index"])
        atom_index = int(row["atom_index"])
        parent = parents.get((episode, parent_index))
        if parent is None or int(row["raw_to_index"]) != int(parent["end_timestep_exclusive"]):
            continue
        raw_start = int(row["raw_from_index"])
        lead = [
            other
            for other in by_atom[(episode, parent_index, atom_index)]
            if int(other.get("quality", 0)) in {2, 3}
            and other.get("cause") in {"stall", "hover"}
            and int(other.get("raw_to_index", -1)) == raw_start
        ]
        # The v2 compiler materializes one second of critique headroom into
        # ``from_index``.  For terminal idle, that effective start is also the
        # beginning of the visibly still run (the raw quality-1 span begins
        # after its first second).  Retention must not preserve that first
        # second merely because the quality taxonomy calls it headroom.
        proposed = int(row["from_index"])
        output.append(
            {
                "episode_id": episode,
                "parent_interval_index": parent_index,
                "atom_index": atom_index,
                "parent_start": int(parent["start_timestep"]),
                "parent_end": int(parent["end_timestep_exclusive"]),
                "idle_raw_start": raw_start,
                "idle_effective_start": int(row["from_index"]),
                "proposed_cut": proposed,
                "native_rate_hz": float(parent["native_rate_hz"]),
                "source": str(parent["source"]),
                "component": str(parent.get("component", "")),
                "split": str(parent["split"]),
                "lead_quality_rows": [
                    {
                        "cause": str(other["cause"]),
                        "quality": int(other["quality"]),
                        "raw_from_index": int(other["raw_from_index"]),
                        "raw_to_index": int(other["raw_to_index"]),
                    }
                    for other in lead
                ],
                "idle_note": str(row.get("note", "")),
            }
        )
    return sorted(output, key=lambda item: (item["episode_id"], item["parent_interval_index"]))


def _frames(candidate: dict[str, Any]) -> list[int]:
    start, stop = int(candidate["proposed_cut"]), int(candidate["parent_end"])
    rate = float(candidate["native_rate_hz"])
    lookback = max(0, start - round(2.0 * rate))
    values = np.linspace(lookback, max(lookback, stop - 1), num=16)
    frames = {int(round(value)) for value in values}
    frames.update({start - 1, start, start + 1, stop - 1})
    return sorted(frame for frame in frames if 0 <= frame < stop)


def render_one(root: Path, out: Path, candidate: dict[str, Any]) -> Path:
    episode = str(candidate["episode_id"])
    episode_dir = root / "corpus" / "episodes" / episode
    record = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    wanted = _frames(candidate)
    names, decoded = decode_common(record, episode_dir, wanted)
    tile_w, tile_h, band_h = 320, 180, 52
    columns = 4
    cell_w, cell_h = tile_w * len(names), tile_h + band_h
    rows = (len(wanted) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_w * columns, cell_h * rows + 72), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    cut, stop = int(candidate["proposed_cut"]), int(candidate["parent_end"])
    draw.text((10, 8), f"{episode} | proposed rejected tail [{cut},{stop})", font=font(22, True), fill="white")
    draw.text(
        (10, 38),
        f"parent p{candidate['parent_interval_index']} atom a{candidate['atom_index']} | "
        f"{candidate['native_rate_hz']:g} Hz | {candidate['idle_note']}",
        font=font(15),
        fill=(210, 210, 210),
    )
    for index, frame in enumerate(wanted):
        x, y = (index % columns) * cell_w, 72 + (index // columns) * cell_h
        for camera_index, name in enumerate(names):
            image = decoded[name].get(frame)
            if image is None:
                image = Image.new("RGB", (tile_w, tile_h), (80, 0, 0))
            canvas.paste(fit_tile(image, tile_w, tile_h), (x + camera_index * tile_w, y))
        color = (255, 110, 110) if frame >= cut else (120, 240, 140)
        draw.rectangle((x, y + tile_h, x + cell_w, y + cell_h), fill=(0, 0, 0))
        draw.text(
            (x + 8, y + tile_h + 5),
            f"f{frame}  t={frame / float(candidate['native_rate_hz']):.2f}s  "
            f"{'PROPOSED REJECT' if frame >= cut else 'RETAIN'}",
            font=font(16, True),
            fill=color,
        )
        draw.text((x + 8, y + tile_h + 28), " | ".join(names), font=font(13), fill=(180, 180, 180))
    target = out / f"{episode}__p{candidate['parent_interval_index']}__terminal_idle.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92)
    return target


def compile_corrections(
    root: Path,
    containment_path: Path,
    out_dir: Path,
    rejected_episodes: set[str],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    containment = json.loads(containment_path.read_text(encoding="utf-8"))
    by_episode = {str(row["episode_id"]): row for row in containment["episodes"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in scan(root):
        grouped[str(candidate["episode_id"])].append(candidate)

    edits, decisions = [], []
    for episode, candidates in sorted(grouped.items()):
        if episode in rejected_episodes:
            decisions.extend({**row, "decision": "rejected_contextual_hold"} for row in candidates)
            continue
        info = by_episode.get(episode)
        if info is None:
            raise ValueError(f"{episode}: absent from containment manifest")
        intervals = [list(map(int, item)) for item in info["retained_intervals"]]
        for row in sorted(candidates, key=lambda item: int(item["proposed_cut"])):
            gap = [int(row["proposed_cut"]), int(row["parent_end"])]
            before = intervals
            intervals = semantic_compiler._subtract_gap(intervals, gap)
            if intervals == before:
                raise ValueError(f"{episode}: proposed cut {gap} changed no retained interval")
            decisions.append({**row, "decision": "confirmed_terminal_idle", "removed_interval": gap})
        edits.append({
            "kind": "retained_intervals",
            "family": str(candidates[0]["source"]),
            "episode_id": episode,
            "intervals": intervals,
            "provenance": "targeted_terminal_idle_sweep_v2_visual_spark",
        })

    conflicts: list[dict[str, Any]] = []
    specs = semantic_compiler.build_specs(
        root, containment, edits, conflicts, sampled_quality_provenance=False
    )
    if len(specs) != len(edits):
        raise ValueError(f"compiled {len(specs)} specs for {len(edits)} edited episodes")
    artifacts: list[tuple[Path, bytes]] = []
    encoded: list[tuple[Path, bytes, dict[str, Any]]] = []
    for spec in specs:
        episode = str(spec["episode_id"])
        payload = fix_normalizer._trace_artifact(root, spec)
        relative = Path("artifacts") / spec["store"] / "speed_hybrid_v1" / f"{episode}.npz"
        spec["artifacts"] = {
            "speed_hybrid_v1": {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "keys": sorted(fix_normalizer.TRACE_KEYS),
            }
        }
        for row in spec["replacements"]["speed_atoms_hybrid_v1.jsonl"]:
            row["state_speed_file"] = f"{spec['store']}/speed_hybrid_v1/{episode}.npz"
        artifacts.append((relative, payload))
        body = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
        encoded.append((Path(f"{episode}.json"), body, spec))

    report = {
        "schema_version": 1,
        "status": "VALID_DRY_RUN" if dry_run else "COMPILED",
        "source_root": str(root.resolve()),
        "containment_manifest": str(containment_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "candidate_count": sum(len(rows) for rows in grouped.values()),
        "confirmed_count": sum(row["decision"] == "confirmed_terminal_idle" for row in decisions),
        "rejected_count": sum(row["decision"] != "confirmed_terminal_idle" for row in decisions),
        "spec_count": len(specs),
        "trace_artifact_count": len(artifacts),
        "conflict_ledger_count": len(conflicts),
        "decisions": decisions,
        "spec_inventory": [
            {
                "episode_id": spec["episode_id"], "store": spec["store"],
                "path": path.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload, spec in encoded
        ],
    }
    if dry_run:
        return report
    temporary = out_dir.with_name(f".{out_dir.name}.compiling-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse compiler temporary directory {temporary}")
    try:
        temporary.mkdir(parents=True)
        for relative, payload, _ in encoded:
            (temporary / relative).write_bytes(payload)
        for relative, payload in artifacts:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        (temporary / "terminal_idle_conflict_ledger.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in conflicts), encoding="utf-8"
        )
        (temporary / "terminal_idle_compiler_manifest.txt").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from lerobot.annotation import assemble_diverse_corrected_store as store_assembler
        store_assembler.load_corrections(temporary)
        store_assembler.validate_trace_artifacts(root, temporary)
        temporary.rename(out_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--containment-manifest", type=Path)
    parser.add_argument("--compile-dir", type=Path)
    parser.add_argument("--reject-episode", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    candidates = scan(args.root)
    report = {
        "schema_version": 1,
        "root": str(args.root.resolve()),
        "candidate_rule": "quality=1 cause=idle raw end equals parent end",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    if args.render_dir is not None:
        rendered = [str(render_one(args.root, args.render_dir, row)) for row in candidates]
        report["render_host"] = socket.gethostname()
        report["rendered"] = rendered
    if (args.containment_manifest is None) != (args.compile_dir is None):
        parser.error("--containment-manifest and --compile-dir must be provided together")
    if args.compile_dir is not None:
        report["compiler"] = compile_corrections(
            args.root.resolve(), args.containment_manifest.resolve(), args.compile_dir.resolve(),
            set(args.reject_episode), dry_run=args.dry_run,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "rendered": len(report.get("rendered", []))}, indent=2))


if __name__ == "__main__":
    main()
