#!/usr/bin/env python

r"""Speed label per subtask segment: work-normalized duration, bucketed 1-5.

Definition. A segment of class $c$ in population group $g$ takes wall time $T$ and moves
the arm joints by the net displacement $D = \lVert q_{\text{end}} - q_{\text{start}} \rVert$
(gripper excluded; degrees for ReBot, radians for the diverse corpus). Then

    expected_s = a_{g,c} + b_{g,c} * D
    ratio      = expected_s / T
    speed      = 1 + #{edge in EDGES : ratio >= edge}

$(a, b)$ is a Theil-Sen fit of $T$ on $D$ over the *training* population of the cell
when the cell has >= MIN_FIT_SEGMENTS segments and Spearman(T, D) >= MIN_FIT_RHO;
otherwise $b = 0$ and $a$ is the cell's median $T$. So the label reads "how quickly this
kind of work got done on this robot", relative to what the same class usually takes for
the same amount of travel. Retries, hovering and wandering all take time without
adding net displacement, so they read as slow; the gripper and the path length are
deliberately not in the formula (path per second ranks a clean 1.2 s deposit below a
10 s wandering release, because it measures motion, not progress).

Release is not rated on its own duration: a deposit is always quick, so its ratio says
nothing about pace, and a sub-second "terminal opening" stub would rate 5. A segment
whose class is in INHERIT_CLASSES takes the speed of the segment before it in the same
episode (the carry it ends); with no predecessor it gets 3. The ratio is still stored.

Buckets (EDGES on the ratio, fixed rather than quantiles so 1/2 mean genuinely slow):
    1  ratio < 0.40         more than 2.5x the expected time: stalled / almost stationary
    2  0.40 <= ratio < 0.65 1.5x - 2.5x the expected time: slow
    3  0.65 <= ratio < 1.10 about the expected time
    4  1.10 <= ratio < 1.40 brisk
    5  ratio >= 1.40        under 71% of the expected time: fast (the inference request)

Populations. ReBot roots are LeRobot v3 datasets whose meta/episode_metadata.parquet
has one row per subtask segment (class = the subtask's verb). The diverse corpus
(outputs/diverse_robot_dataset) contributes one segment per reviewed subtask atom:
FMB keyed by source primitive, other sources by the atom verb within robot group.
Historical parent intervals require explicit --diverse-layer critic_intervals. A class with fewer than
MIN_CLASS_SEGMENTS segments falls back to its group's pooled cell.

Storage mirrors mistakes.parquet: a new meta/speed.parquet per ReBot root (never
rewrites an existing table; --force to replace) and speed_atoms.jsonl beside each diverse
subtask_atoms.jsonl (historical parent labels remain in speed.jsonl), both carrying duration, displacement, expected time and ratio
next to the bucket so the label is auditable. The reference is fit once on the train
roots and applied to val / inference roots unchanged.

    uv run python -m lerobot.data_processing.annotate.speed_annotate fit \
        --root outputs/rebot_socks_basket-annotated-v2 ... --diverse outputs/diverse_robot_dataset \
        --out outputs/stats/speed_reference_<name>.json
    uv run python -m lerobot.data_processing.annotate.speed_annotate annotate --reference <json> --root <root> ...
    uv run python -m lerobot.data_processing.annotate.speed_annotate annotate-diverse --reference <json> --diverse <dir>
    uv run python -m lerobot.data_processing.annotate.speed_annotate review --root <root> ... --diverse <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EDGES = (0.40, 0.65, 1.10, 1.40)
MIN_FIT_SEGMENTS = 30
MIN_FIT_RHO = 0.3
MIN_CLASS_SEGMENTS = 4
INHERIT_CLASSES = ("release",)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CLIP_MAX_S = 45.0

# DROID / UR7e free-text subtasks collapse onto four classes by their verb.
VERB_CLASSES = {
    "approach": ("approach", "reach", "align", "lower", "reposition", "reorient", "shift", "withdraw", "raise",
                 "tilt", "rotate"),
    "grasp": ("grasp", "pick", "regrasp", "re-grasp", "attempt", "close", "contact", "grab", "hold", "extract",
              "remove", "pull", "open", "insert", "press", "push", "gather", "separate", "switch", "use", "scrub",
              "stir", "wipe", "fold", "sort", "pour", "recover", "continue", "iteratively", "complete", "finish",
              "lift"),
    "transport": ("carry", "transfer", "bring", "move"),
    "place": ("place", "release", "set", "stack", "hang", "put", "drop", "arrange", "make"),
}
_VERB_TO_CLASS = {verb: cls for cls, verbs in VERB_CLASSES.items() for verb in verbs}


def bucket(ratio: float) -> int:
    return 1 + int(sum(ratio >= edge for edge in EDGES))


# ── Segments ─────────────────────────────────────────────────────────────────


def _rebot_states(root: Path) -> np.ndarray:
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    frames = [pd.read_parquet(f, columns=["index", "observation.state"]) for f in files]
    df = pd.concat(frames).sort_values("index")
    assert (df["index"].to_numpy() == np.arange(len(df))).all(), f"{root}: non-contiguous frame index"
    return np.stack(df["observation.state"].to_numpy()).astype(np.float64)


def rebot_segments(root: Path) -> list[dict]:
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    states = _rebot_states(root)[:, :6]  # the six arm joints, degrees
    rows = []
    for seg in pd.read_parquet(root / "meta" / "episode_metadata.parquet").itertuples(index=False):
        a, b = int(seg.from_index), int(seg.to_index)
        rows.append({
            "root": str(root), "group": "rebot", "class": seg.subtask.split(" ")[0], "split": "train",
            "episode_index": int(seg.episode_index), "segment_index": int(seg.segment_index),
            "from_index": a, "to_index": b, "subtask": seg.subtask, "quality": int(seg.quality),
            "note": str(getattr(seg, "note", "")), "duration_s": (b - a) / fps,
            "net_displacement": float(np.linalg.norm(states[b - 1] - states[a])),
        })
    return rows


def _diverse_group(source: str, embodiment: str) -> str:
    if source in ("droid", "droid_success"):
        return "droid"
    if source == "robochallenge":
        return f"robochallenge/{embodiment}"
    return source


def _diverse_class(source: str, description: str, primitive: str | None) -> str:
    if source == "fmb":
        return primitive
    if source == "robochallenge":
        return description
    verb = description.lower().split(" ")[0].strip(",")
    return _VERB_TO_CLASS.get(verb, "other")


def _parent_diverse_segments(diverse: Path) -> list[dict]:
    rows = []
    for line in (diverse / "corpus" / "critic_intervals.jsonl").read_text().splitlines():
        r = json.loads(line)
        episode_dir = diverse / "corpus" / r["state_and_observation_reference"]["episode_directory"]
        state = np.load(episode_dir / "state.npy", mmap_mode="r")
        a, b = int(r["start_timestep"]), int(r["end_timestep_exclusive"])
        rows.append({
            "root": str(diverse / "corpus"), "group": _diverse_group(r["source"], r["embodiment"]),
            "class": _diverse_class(r["source"], r["normalized_description"], r.get("primitive")),
            "split": r["split"], "source": r["source"], "embodiment": r["embodiment"],
            "episode_id": r["episode_id"], "interval_index": int(r["interval_index"]),
            "start_timestep": a, "end_timestep_exclusive": b, "subtask": r["normalized_description"],
            "quality": r.get("quality"), "note": "", "duration_s": float(r["duration_s"]),
            "net_displacement": float(np.linalg.norm(np.asarray(state[b - 1, :-1]) - np.asarray(state[a, :-1]))),
            "native_rate_hz": float(r["native_rate_hz"]), "episode_dir": str(episode_dir),
        })
    for line in (diverse / "fmb" / "critic_intervals.jsonl").read_text().splitlines():
        r = json.loads(line)
        episode_dir = diverse / "fmb" / "episodes" / r["episode_id"]
        q = np.load(episode_dir / "q.npy", mmap_mode="r")
        a, b = int(r["start_timestep"]), int(r["end_timestep_exclusive"])
        rows.append({
            "root": str(diverse / "fmb"), "group": "fmb", "class": r["primitive"], "split": r["split"],
            "source": "fmb", "embodiment": "Franka", "episode_id": r["episode_id"],
            "interval_index": int(r["interval_index"]), "start_timestep": a, "end_timestep_exclusive": b,
            "subtask": r["normalized_description"], "quality": r.get("quality"), "note": "",
            "duration_s": float(r["duration_s_nominal"]),
            "net_displacement": float(np.linalg.norm(np.asarray(q[b - 1]) - np.asarray(q[a]))),
            "native_rate_hz": 10.0, "episode_dir": str(episode_dir),
        })
    return rows


def diverse_segments(diverse: Path, layer: str = "atoms") -> list[dict]:
    """Load atomic supervision explicitly; never silently fall back to parents."""
    if layer == "critic_intervals":
        return _parent_diverse_segments(diverse)
    if layer != "atoms":
        raise ValueError(f"Unknown diverse layer: {layer}")
    paths = [diverse / sub / "subtask_atoms.jsonl" for sub in ("corpus", "fmb")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: finish and validate the atomic layer first")
    rows = []
    seen = set()
    for path in paths:
        sub = path.parent.name
        states = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            eid = r["episode_id"]
            key = (sub, eid, int(r["parent_interval_index"]), int(r["atom_index"]))
            if key in seen:
                raise ValueError(f"Duplicate atom: {key}")
            seen.add(key)
            episode_dir = path.parent / "episodes" / eid
            if eid not in states:
                states[eid] = np.load(episode_dir / ("q.npy" if sub == "fmb" else "state.npy"), mmap_mode="r")
            q = states[eid] if sub == "fmb" else states[eid][:, :-1]
            a, b = int(r["start_timestep"]), int(r["end_timestep_exclusive"])
            rate = float(r["native_rate_hz"])
            if not (0 <= a < b <= len(q)) or rate <= 0:
                raise ValueError(f"Invalid atom span/rate: {key}")
            rows.append({
                "root": str(path.parent), "group": _diverse_group(r["source"], r["embodiment"]),
                "class": r["primitive"] if sub == "fmb" else r["verb"],
                "split": r["split"], "source": r["source"], "embodiment": r["embodiment"],
                "episode_id": eid, "parent_interval_index": key[2], "atom_index": key[3],
                "start_timestep": a, "end_timestep_exclusive": b, "subtask": r["subtask"],
                "quality": r["quality"], "note": r.get("note", ""), "confidence": r["confidence"],
                "duration_s": (b - a) / rate,
                "net_displacement": float(np.linalg.norm(np.asarray(q[b - 1]) - np.asarray(q[a]))),
                "native_rate_hz": rate, "episode_dir": str(episode_dir), "annotation_layer": "atoms",
            })
    return sorted(rows, key=lambda r: (r["root"], r["episode_id"], r["start_timestep"]))


def _diverse_key(row: dict) -> tuple:
    return (row["episode_id"], row["parent_interval_index"], row["atom_index"]) if "atom_index" in row else (row["episode_id"], row["interval_index"])


def _speed_filename(layer: str) -> str:
    return "speed_atoms.jsonl" if layer == "atoms" else "speed.jsonl"


# ── Reference ────────────────────────────────────────────────────────────────


def _fit_cell(durations: np.ndarray, displacements: np.ndarray) -> dict:
    cell = {"n": int(len(durations)), "median_s": float(np.median(durations)), "rho": None, "fit": "median"}
    if len(durations) >= MIN_FIT_SEGMENTS:
        rho = float(stats.spearmanr(durations, displacements)[0])
        cell["rho"] = rho
        if rho >= MIN_FIT_RHO:
            slope, intercept, *_ = stats.theilslopes(durations, displacements)
            cell.update(a=float(intercept), b=float(slope), fit="theil_sen")
    if cell["fit"] == "median":
        cell.update(a=cell["median_s"], b=0.0)
    return cell


def fit_reference(segments: list[dict], roots: list[str], diverse: str | None) -> dict:
    train = pd.DataFrame([s for s in segments if s["split"] == "train"])
    cells = {}
    for group, g in train.groupby("group"):
        cells[f"{group}/*"] = _fit_cell(g["duration_s"].to_numpy(), g["net_displacement"].to_numpy())
        for cls, gc in g.groupby("class"):
            cells[f"{group}/{cls}"] = _fit_cell(gc["duration_s"].to_numpy(), gc["net_displacement"].to_numpy())
    return {
        "definition": "expected_s = a + b * net_displacement; ratio = expected_s / duration_s; "
                      "speed = 1 + #{edges <= ratio}. Displacement over arm joints (gripper excluded): "
                      "degrees for rebot, radians for the diverse groups.",
        "edges": list(EDGES), "min_fit_segments": MIN_FIT_SEGMENTS, "min_fit_rho": MIN_FIT_RHO,
        "min_class_segments": MIN_CLASS_SEGMENTS, "fit_roots": roots, "fit_diverse": diverse,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cells": cells,
    }


def label(segments: list[dict], reference: dict) -> list[dict]:
    cells = reference["cells"]
    edges = tuple(reference["edges"])
    out = []
    for s in segments:
        key = f"{s['group']}/{s['class']}"
        if key not in cells or cells[key]["n"] < reference["min_class_segments"]:
            key = f"{s['group']}/*"
        cell = cells[key]
        expected = cell["a"] + cell["b"] * s["net_displacement"]
        ratio = expected / s["duration_s"]
        out.append(s | {"reference_cell": key, "expected_s": expected, "ratio": ratio,
                        "speed": 1 + int(sum(ratio >= e for e in edges)), "speed_source": "ratio"})
    previous = {}
    for r in out:
        episode = (r["root"], r.get("episode_index", r.get("episode_id")))
        if r["class"] in INHERIT_CLASSES:
            r["speed"] = previous.get(episode, 3)
            r["speed_source"] = "inherited" if episode in previous else "default"
        previous[episode] = r["speed"]
    if reference.get("unclear_policy") == "atoms_default_3_v1":
        _default_unclear_atoms(out, reference)
    return out


def _default_unclear_atoms(rows: list[dict], reference: dict) -> None:
    """Opt-in conservative atom labels; keep the original label for inspection.

    Releases inherit only across contiguous carry/release supervision. Rare verbs
    and unsure reviews default to 3. Same-action parent fragments are flagged;
    their accepted boundaries and metric remain unchanged.
    """
    previous = {}
    for r in rows:
        if r.get("annotation_layer") != "atoms":
            continue
        episode = (r["root"], r["episode_id"])
        prev = previous.get(episode)
        contiguous = prev is not None and prev["end_timestep_exclusive"] == r["start_timestep"]
        flags, defaults = [], []
        r["raw_speed"] = r["speed"]
        if r.get("confidence") == "unsure":
            defaults.append("uncertain atomic review")
        if not np.isfinite(r["expected_s"]) or r["expected_s"] <= 0 or not np.isfinite(r["ratio"]):
            defaults.append("invalid expected duration")
        if r["class"] == "release":
            if not contiguous or prev["class"] not in ("move", "lift", "release"):
                defaults.append("release has no contiguous carry or release predecessor")
            else:
                r["speed"] = prev["speed"]
                r["speed_source"] = "inherited"
                if prev.get("speed_default_reason"):
                    defaults.append("release inherits an uncertain predecessor")
        else:
            cell = reference["cells"].get(f"{r['group']}/{r['class']}")
            if cell is None or cell["n"] < reference["min_class_segments"]:
                defaults.append("too few training examples of this robot/action")
        if (contiguous and prev["parent_interval_index"] != r["parent_interval_index"]
                and prev["subtask"] == r["subtask"]):
            flag = "same action continues across a parent boundary"
            flags.append(flag)
            if flag not in prev["speed_flags"]:
                prev["speed_flags"].append(flag)
        r["speed_default_reason"] = "; ".join(defaults)
        r["speed_flags"] = defaults + flags
        if defaults:
            r["speed"] = 3
            r["speed_source"] = "default_unclear"
        previous[episode] = r


def _shares(rows: list[dict]) -> str:
    counts = np.bincount([r["speed"] for r in rows], minlength=6)[1:]
    return " ".join(f"{k + 1}:{c / max(len(rows), 1):.0%}" for k, c in enumerate(counts)) + f"  (n={len(rows)})"


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_fit(args):
    segments = []
    for root in args.root:
        segments += rebot_segments(Path(root))
    if args.diverse:
        segments += diverse_segments(Path(args.diverse), args.diverse_layer)
    reference = fit_reference(segments, args.root, args.diverse)
    if args.diverse:
        reference["diverse_layer"] = args.diverse_layer
        source_name = "subtask_atoms.jsonl" if args.diverse_layer == "atoms" else "critic_intervals.jsonl"
        reference["diverse_source_sha256"] = {
            sub: hashlib.sha256((Path(args.diverse) / sub / source_name).read_bytes()).hexdigest()
            for sub in ("corpus", "fmb")
        }
    reference["inherit_classes"] = list(INHERIT_CLASSES)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, indent=1))
    print(f"wrote {out}")
    for key, cell in reference["cells"].items():
        fit = f"T = {cell['a']:.2f} + {cell['b']:.3f} * D  (rho {cell['rho']:.2f})" if cell["fit"] == "theil_sen" \
            else f"T = median {cell['median_s']:.2f} s" + (f"  (rho {cell['rho']:.2f} below {MIN_FIT_RHO})" if cell["rho"] is not None else "")
        print(f"  {key:60s} n={cell['n']:4d}  {fit}")
    labelled = label(segments, reference)
    for group in sorted({s["group"] for s in labelled}):
        print(f"train shares {group:22s}", _shares([s for s in labelled if s["group"] == group and s["split"] == "train"]))


def _write_info(root: Path, reference_path: str, rows: list[dict]) -> None:
    (root / "meta" / "speed_info.json").write_text(json.dumps({
        "reference": reference_path, "edges": list(EDGES),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shares": {str(k): int(c) for k, c in zip(range(1, 6), np.bincount([r["speed"] for r in rows], minlength=6)[1:])},
    }, indent=1))


def cmd_annotate(args):
    reference = json.loads(Path(args.reference).read_text())
    for root in args.root:
        root = Path(root)
        target = root / "meta" / "speed.parquet"
        if target.exists() and not args.force:
            raise FileExistsError(f"{target} exists; pass --force to replace it")
        rows = label(rebot_segments(root), reference)
        columns = ["episode_index", "segment_index", "from_index", "to_index", "subtask", "class", "duration_s",
                   "net_displacement", "expected_s", "ratio", "reference_cell", "speed", "speed_source"]
        pd.DataFrame([{k: r[k] for k in columns} for r in rows]).to_parquet(target, engine="pyarrow")
        _write_info(root, args.reference, rows)
        print(f"{root.name}: wrote {len(rows)} rows  shares", _shares(rows))
        ordered = sorted(rows, key=lambda r: r["ratio"])
        for tag, picks in (("slowest", ordered[:3]), ("fastest", ordered[-3:])):
            for r in picks:
                print(f"    {tag:8s} ep{r['episode_index']:3d} seg{r['segment_index']:3d} speed {r['speed']} "
                      f"ratio {r['ratio']:.2f} {r['duration_s']:5.1f}s (exp {r['expected_s']:4.1f}s) "
                      f"{r['subtask']:40s} q{r['quality']} | {r['note'][:90]}")


def cmd_annotate_diverse(args):
    reference = json.loads(Path(args.reference).read_text())
    diverse = Path(args.diverse)
    if reference.get("diverse_layer", "critic_intervals") != args.diverse_layer:
        raise ValueError("Reference and diverse annotation layer differ; refit on the requested layer")
    source_name = "subtask_atoms.jsonl" if args.diverse_layer == "atoms" else "critic_intervals.jsonl"
    for sub, expected_hash in reference.get("diverse_source_sha256", {}).items():
        if hashlib.sha256((diverse / sub / source_name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"{sub}: annotation layer changed since reference fitting")
    rows = label(diverse_segments(diverse, args.diverse_layer), reference)
    columns = ["episode_id", "interval_index", "source", "embodiment", "group", "class", "start_timestep",
               "end_timestep_exclusive", "subtask", "duration_s", "net_displacement", "expected_s", "ratio",
               "reference_cell", "speed", "speed_source"]
    if args.diverse_layer == "atoms":
        columns.remove("interval_index")
        columns += ["parent_interval_index", "atom_index", "annotation_layer", "confidence"]
        if reference.get("unclear_policy") == "atoms_default_3_v1":
            columns += ["raw_speed", "speed_flags", "speed_default_reason"]
    for sub in ("corpus", "fmb"):
        target = diverse / sub / _speed_filename(args.diverse_layer)
        if target.exists() and not args.force:
            raise FileExistsError(f"{target} exists; pass --force to replace it")
        picked = [r for r in rows if r["root"] == str(diverse / sub)]
        target.write_text("".join(json.dumps({k: r[k] for k in columns}) + "\n" for r in picked))
        print(f"{target}: wrote {len(picked)} rows")
        for group in sorted({r["group"] for r in picked}):
            print(f"    shares {group:22s}", _shares([r for r in picked if r["group"] == group]))


# ── Review clips ─────────────────────────────────────────────────────────────


def _overlay(text_path: Path) -> str:
    return (f"drawtext=fontfile={FONT}:textfile={text_path}:fontsize=22:fontcolor=white:"
            f"box=1:boxcolor=black@0.6:boxborderw=6:x=8:y=8")


def _rebot_clip(root: Path, row: dict, out: Path, text_path: Path) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    episodes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))])
    ep = episodes[episodes.episode_index == row["episode_index"]].iloc[0]
    duration = min((row["to_index"] - row["from_index"]) / fps, CLIP_MAX_S)
    inputs = []
    for key in ("observation.images.top", "observation.images.wrist"):
        path = root / info["video_path"].format(video_key=key, chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                                                file_index=int(ep[f"videos/{key}/file_index"]))
        start = float(ep[f"videos/{key}/from_timestamp"]) + (row["from_index"] - int(ep["dataset_from_index"])) / fps
        inputs += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path)]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex",
                    f"[0:v][1:v]hstack,{_overlay(text_path)}", "-c:v", "libx264", "-crf", "26", "-preset",
                    "veryfast", "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


def _corpus_clip(row: dict, out: Path, text_path: Path) -> None:
    videos = sorted(Path(row["episode_dir"], "videos").glob("*.mp4"))
    external = next((v for v in videos if "wrist" not in v.name), videos[0])
    wrist = next((v for v in videos if "wrist" in v.name), videos[-1])
    rate = row["native_rate_hz"]
    start = row["start_timestep"] / rate
    duration = min((row["end_timestep_exclusive"] - row["start_timestep"]) / rate, CLIP_MAX_S)
    inputs = []
    for path in (external, wrist):
        inputs += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path)]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex",
                    f"[0:v]scale=640:-2[a];[1:v]scale=640:-2[b];[a][b]hstack,{_overlay(text_path)}",
                    "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(out)],
                   check=True)


def _fmb_clip(row: dict, out: Path, text_path: Path) -> None:
    episode_dir = Path(row["episode_dir"])
    a, b = row["start_timestep"], row["end_timestep_exclusive"]
    side = np.load(episode_dir / "side_1_rgb.npy", mmap_mode="r")[a:b]
    wrist = np.load(episode_dir / "wrist_1_rgb.npy", mmap_mode="r")[a:b]
    frames = np.concatenate([np.asarray(side), np.asarray(wrist)], axis=2)  # (T, H, 2W, 3)
    t, h, w, _ = frames.shape
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                             "rgb24", "-s", f"{w}x{h}", "-r", str(row["native_rate_hz"]), "-i", "-", "-vf",
                             f"scale={2 * w}:{2 * h}:flags=neighbor,{_overlay(text_path)}", "-c:v", "libx264",
                             "-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(out)],
                            stdin=subprocess.PIPE)
    proc.communicate(np.ascontiguousarray(frames).tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {out}")


def _strip(clip: Path, out: Path, n: int = 8) -> None:
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
                           capture_output=True, text=True, check=True)
    duration = max(float(probe.stdout.strip()), 0.1)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip), "-vf",
                    f"fps={n / duration:.4f},scale=400:-2,tile={n}x1", "-frames:v", "1", str(out)], check=True)


def _sample(rows: list[dict], per_bucket: int, rng: random.Random) -> list[dict]:
    picked = []
    for speed in range(1, 6):
        pool = [r for r in rows if r["speed"] == speed]
        picked += rng.sample(pool, min(per_bucket, len(pool)))
    return picked


def cmd_review(args):
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    # One population per dataset (each ReBot root, each diverse group), sampled per bucket.
    picks = []
    for root in args.root:
        root = Path(root)
        table = pd.read_parquet(root / "meta" / "speed.parquet")
        meta = pd.read_parquet(root / "meta" / "episode_metadata.parquet")[["episode_index", "segment_index", "quality", "note"]]
        table = table.merge(meta, on=["episode_index", "segment_index"])
        rows = [r | {"kind": "rebot", "root": root, "group": "rebot", "population": root.name} for r in table.to_dict("records")]
        picks += _sample(rows, args.per_bucket, rng)
    if args.diverse:
        diverse = Path(args.diverse)
        current = diverse_segments(diverse, args.diverse_layer)
        by_row = {(Path(s["root"]).name, *_diverse_key(s)): s for s in current}
        for sub in ("corpus", "fmb"):
            rows = [json.loads(l) for l in (diverse / sub / _speed_filename(args.diverse_layer)).read_text().splitlines()]
            expected_keys = {key for key in by_row if key[0] == sub}
            keys = [(sub, *_diverse_key(r)) for r in rows]
            if len(set(keys)) != len(keys) or set(keys) != expected_keys:
                raise ValueError(f"{sub}: speed rows do not match current annotation keys")
            for r, key in zip(rows, keys, strict=True):
                source = by_row[key]
                for field in ("start_timestep", "end_timestep_exclusive", "subtask"):
                    if r[field] != source[field]:
                        raise ValueError(f"Stale speed annotation {key}: {field} differs")
            rows = [r | by_row[key] | {"kind": sub} for r, key in zip(rows, keys, strict=True)]
            for group in sorted({r["group"] for r in rows}):
                group_rows = [r | {"population": f"diverse {group}"} for r in rows if r["group"] == group]
                picks += _sample(group_rows, args.diverse_per_bucket, rng)
    cards = []
    for i, r in enumerate(picks):
        name = f"{i:03d}_speed{r['speed']}_{r['kind']}"
        clip, strip, text_path = out / "clips" / f"{name}.mp4", out / "clips" / f"{name}.png", out / "clips" / f"{name}.txt"
        where = (f"{Path(r['root']).name} ep{r['episode_index']} seg{r['segment_index']}" if r["kind"] == "rebot"
                 else (f"{r['episode_id']} parent {r['parent_interval_index']} atom {r['atom_index']}" if "atom_index" in r
                       else f"{r['episode_id']} interval {r['interval_index']}"))
        caption = (f"speed {r['speed']}  |  {r['subtask']}  |  {r['duration_s']:.1f} s, expected "
                   f"{r['expected_s']:.1f} s, ratio {r['ratio']:.2f}  |  {where}")
        if r.get("speed_source") == "inherited":
            caption += "  |  speed inherited from preceding segment; ratio is diagnostic only"
        elif r.get("speed_source") == "default":
            caption += "  |  default speed 3; no preceding segment"
        if r["kind"] != "fmb" and r["duration_s"] > CLIP_MAX_S:
            caption += f"  |  showing first {CLIP_MAX_S:g} s"
        text_path.write_text(caption.replace("|", "-"))
        if r["kind"] == "rebot":
            _rebot_clip(r["root"], r, clip, text_path)
        elif r["kind"] == "corpus":
            _corpus_clip(r, clip, text_path)
        else:
            _fmb_clip(r, clip, text_path)
        _strip(clip, strip)
        cards.append((r["population"], r["speed"], caption, r.get("note") or "", clip.name, strip.name))
        print(f"{name}: {caption}")
    sections = []
    for population in dict.fromkeys(c[0] for c in cards):
        blocks = []
        for speed in range(1, 6):
            items = "".join(
                f'<div class="card"><video controls preload="metadata" src="clips/{clip}"></video>'
                f'<p class="cap">{cap}</p><p class="note">{note}</p><img src="clips/{strip}"></div>'
                for pop, s, cap, note, clip, strip in cards if pop == population and s == speed)
            blocks.append(f"<h3>speed {speed}</h3>{items or '<p class=note>no segment of this dataset falls in this bucket</p>'}")
        sections.append(f"<h2>{population}</h2>{''.join(blocks)}")
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>speed review</title><style>body{font-family:sans-serif;"
        "background:#111;color:#ddd;margin:1em}.card{margin:0 0 1.5em}video{width:100%;max-width:1280px;display:block}"
        "img{width:100%;max-width:1280px;display:block}.cap{margin:.3em 0 0}.note{color:#999;font-size:.9em;margin:.2em 0 .4em}"
        f"h2{{border-top:1px solid #444;padding-top:.5em}}</style><h1>speed labels, {len(cards)} clips, seed {args.seed}</h1>"
        f"<p>edges on ratio = expected / actual: {EDGES}; clips play in real time, top/external left, wrist right; "
        "one section per dataset, one block per bucket.</p>"
        + "".join(sections))
    print(f"wrote {out / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--diverse")
    p.add_argument("--diverse-layer", choices=("atoms", "critic_intervals"), default="atoms")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fit)
    p = sub.add_parser("annotate")
    p.add_argument("--reference", required=True)
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_annotate)
    p = sub.add_parser("annotate-diverse")
    p.add_argument("--reference", required=True)
    p.add_argument("--diverse", required=True)
    p.add_argument("--diverse-layer", choices=("atoms", "critic_intervals"), default="atoms")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_annotate_diverse)
    p = sub.add_parser("review")
    p.add_argument("--root", nargs="*", default=[])
    p.add_argument("--diverse")
    p.add_argument("--diverse-layer", choices=("atoms", "critic_intervals"), default="atoms")
    p.add_argument("--out", required=True)
    p.add_argument("--per-bucket", type=int, default=1)
    p.add_argument("--diverse-per-bucket", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_review)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
