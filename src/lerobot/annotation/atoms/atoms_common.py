"""Shared pieces for the atomic subtask layer over the diverse corpus.

Proprio is the triage: gripper steps give close/open events, arm displacement separates
carries from failed closes, the last fast frame of a carry gives the arrival (arm settle).
Vision is the verdict: every candidate is rendered on a sheet and a reviewer writes the
atoms, names, quality and mistakes per parent in a verdict file.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from lerobot.annotation.paths import WORKSPACE as ROOT

# v2 corpora point these at their own root and review tree through the environment.
DATA_ROOT = ROOT / os.environ.get("DIVERSE_DATASET_ROOT", "outputs/diverse_robot_dataset")
CORPUS = DATA_ROOT / "corpus"
FMB = DATA_ROOT / "fmb"
WORK = ROOT / os.environ.get("ATOMS_WORK_ROOT", "migration/subtask_atoms_2026-09-08")
REVIEW = ROOT / os.environ.get("ATOMS_REVIEW_ROOT", "outputs/_annotation/subtask_atoms_review")
# Sources whose state is an end-effector pose (xyz metres, Euler, gripper), not joints.
EE_SOURCES = {"molmoact"}
DISP_MIN_M = 0.03  # xyz displacement below which a closed run never lifted anything
GRAMMAR_VERSION = "subtask-atoms-v1"

# Gripper channel semantics, per embodiment: closedness in [0, 1] (1 = shut).
GRIPPER = {
    "Franka": {"kind": "position", "closed_high": True, "scale": 1.0},  # DROID 0 open .. 1 closed
    "UR7e": {"kind": "ratio", "closed_high": True, "scale": 1.0},  # 0 open .. 1 closed
    "ARX5": {"kind": "width", "closed_high": False, "scale": 0.0872},  # metres, rests closed
    "UR5": {"kind": "width", "closed_high": False, "scale": 0.085},  # metres, rests open
    "YAM": {"kind": "ratio", "closed_high": True, "scale": 1.0},  # 0 open .. 1 closed after the v3 ingest flip
}
STEP_DELTA = {"Franka": 0.20, "UR7e": 0.20, "ARX5": 0.15, "UR5": 0.15, "YAM": 0.20}  # closedness step that counts
DISP_MIN_RAD = 0.20  # arm-joint L2 displacement below which a closed run never picked anything up
MIN_ATOM_S = 0.5  # runs shorter than this inside a parent merge into a neighbour
SETTLE_FRACTION = 0.40  # of the carry's peak joint speed: below it the arm is settling over the target
YAW_TOL = 0.05  # rad, ReBot's 3 deg pan tolerance
YAW_TRAVEL_MIN = 0.30  # rad of base swing before the yaw rule applies
EXTERNAL_CAMERA = {
    "droid": "left_external",
    "droid_success": "left_external",
    "robochallenge": "global",
    "ur7e": "realsense_topview",
    "molmoact": "primary",
    "yam": ("top", "outside", "cam_high"),  # one per repo: duster, espresso, pick-place
}
WRIST_CAMERA = {
    "droid": "wrist",
    "droid_success": "wrist",
    "robochallenge": "wrist",
    "ur7e": "realsense_wrist",
    "molmoact": "wrist",
    "yam": ("right_wrist", "wrist", "cam_wrist"),
}


def episode_cameras(ep: dict) -> tuple[str, str]:
    """(external, wrist) recorded camera names for one episode record; a tuple entry is resolved
    against the cameras the episode actually carries (YAM's repos spell them differently)."""
    recorded = set(ep["cameras"]) if isinstance(ep.get("cameras"), (list, dict)) else set()
    names = []
    for table in (EXTERNAL_CAMERA, WRIST_CAMERA):
        entry = table[ep["source"]]
        if isinstance(entry, tuple):
            entry = next(name for name in entry if name in recorded)
        names.append(entry)
    return names[0], names[1]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_corpus():
    episodes = {e["episode_id"]: e for e in read_jsonl(CORPUS / "episodes.jsonl")}
    parents = defaultdict(list)
    for row in read_jsonl(CORPUS / "critic_intervals.jsonl"):
        parents[row["episode_id"]].append(row)
    for rows in parents.values():
        rows.sort(key=lambda r: r["interval_index"])
    return episodes, parents


def episode_arrays(episode_id: str):
    root = CORPUS / "episodes" / episode_id
    state = np.load(root / "state.npy")
    timestamps = np.load(root / "timestamp_s.npy")
    return state, timestamps


def closedness(state: np.ndarray, embodiment: str) -> np.ndarray:
    spec = GRIPPER[embodiment]
    g = state[:, -1].astype(np.float64)
    if spec["closed_high"]:
        return np.clip(g / spec["scale"], 0.0, 1.0)
    return np.clip(1.0 - g / spec["scale"], 0.0, 1.0)


def joint_speed(state: np.ndarray, rate: float, smooth_s: float = 0.3) -> np.ndarray:
    q = state[:, :-1]
    v = np.linalg.norm(np.diff(q, axis=0), axis=1) * rate
    v = np.concatenate([v[:1], v])
    k = max(1, int(round(smooth_s * rate)))
    return np.convolve(v, np.ones(k) / k, mode="same")


def gripper_events(c: np.ndarray, rate: float, delta: float, window_s: float = 0.4) -> list[dict]:
    """Step events in the closedness signal.

    d[t] = median over the next window - median over the previous window. A local |d|
    extremum above `delta` is an event; the event frame is where c crosses the midpoint
    between the two medians. Returns events in time order with before/after levels.
    """
    n = len(c)
    w = max(2, int(round(window_s * rate)))
    if n < 2 * w + 2:
        return []
    d = np.zeros(n)
    for t in range(w, n - w):
        d[t] = np.median(c[t : t + w]) - np.median(c[t - w : t])
    events = []
    t = w
    while t < n - w:
        if abs(d[t]) >= delta:
            # extend over the contiguous region above threshold with the same sign, take the peak
            sign = np.sign(d[t])
            u = t
            peak = t
            while u < n - w and np.sign(d[u]) == sign and abs(d[u]) >= delta * 0.5:
                if abs(d[u]) > abs(d[peak]):
                    peak = u
                u += 1
            before = float(np.median(c[max(0, peak - w) : peak]))
            after = float(np.median(c[peak : peak + w]))
            mid = 0.5 * (before + after)
            # crossing frame: first frame in [peak - w, peak + w) on the 'after' side of mid
            seg = c[max(0, peak - w) : min(n, peak + w)]
            base = max(0, peak - w)
            if sign > 0:
                cross = np.flatnonzero(seg >= mid)
            else:
                cross = np.flatnonzero(seg <= mid)
            frame = base + int(cross[0]) if len(cross) else peak
            # ramp bounds: where the signal leaves the 'before' band and reaches the 'after' band
            span = abs(after - before)
            lo = base
            for k in range(frame, base - 1, -1):
                if abs(c[k] - before) <= 0.15 * span:
                    lo = k
                    break
            hi = min(n - 1, frame + w)
            for k in range(frame, min(n, frame + int(1.5 * rate))):
                if abs(c[k] - after) <= 0.15 * span:
                    hi = k
                    break
            events.append(
                {
                    "kind": "close" if sign > 0 else "open",
                    "frame": int(frame),
                    "ramp_begin": int(lo),
                    "ramp_end": int(hi + 1),
                    "before": round(before, 3),
                    "after": round(after, 3),
                    "step": round(float(after - before), 3),
                }
            )
            t = u + 1
        else:
            t += 1
    return events


def closed_runs(c: np.ndarray, events: list[dict], state: np.ndarray, rate: float, ee: bool = False) -> list[dict]:
    """Pair closings with the next opening into closed runs, with arm displacement.

    Runs that start at the episode start (rest state closed, ARX5) have no close event and
    are marked `rest_start`; a run that reaches the episode end has no open event.
    """
    n = len(c)
    runs = []
    closed = bool(c[0] > 0.5)
    current = {"close": None, "rest_start": closed} if closed else None
    primary = []
    for e in events:
        if e["kind"] == "close":
            if current is None:
                current = {"close": e, "rest_start": False}
                primary.append(e)
            else:
                e["secondary"] = True  # tightened while already closed
        else:
            if current is not None:
                current["open"] = e
                runs.append(current)
                current = None
                primary.append(e)
            else:
                e["secondary"] = True  # widened while already open
    if current is not None:
        current["open"] = None
        runs.append(current)
    q = state[:, :3] if ee else state[:, :-1]  # xyz metres for an end-effector state
    out = []
    for r in runs:
        start = 0 if r["rest_start"] else r["close"]["frame"]
        stop = n if r["open"] is None else r["open"]["frame"]
        seg = q[start:stop]
        disp = float(np.linalg.norm(seg - seg[0], axis=1).max()) if len(seg) > 1 else 0.0
        out.append(
            {
                "start": int(start),
                "stop": int(stop),
                "close": r["close"],
                "open": r["open"],
                "rest_start": bool(r["rest_start"]),
                "rest_end": bool(r["open"] is None),
                "disp_rad": round(disp, 3),
                "duration_s": round((stop - start) / rate, 2),
            }
        )
    return out


def arrival_frame(speed: np.ndarray, yaw: np.ndarray, close: int, open_begin: int) -> int:
    """Arrival over the target: the earlier of two measured events.

    * yaw arrival (ReBot's pan rule): the base joint settles into +-YAW_TOL of its value at
      the opening and stays there; only used when the carry swings the base by more than
      YAW_TRAVEL_MIN, so a carry that does not turn cannot be called 'arrived' at its close.
    * speed settle: the frame after the last frame above SETTLE_FRACTION of the carry's
      peak joint speed, i.e. the start of the final slow approach.
    """
    if open_begin - close < 2:
        return close
    seg = speed[close:open_begin]
    peak = float(seg.max())
    settle = close
    if peak > 1e-6:
        fast = np.flatnonzero(seg > SETTLE_FRACTION * peak)
        settle = int(close + fast[-1] + 1) if len(fast) else close
    ys = yaw[close:open_begin]
    travel = float(np.abs(ys - ys[0]).max())
    if travel > YAW_TRAVEL_MIN:
        outside = np.flatnonzero(np.abs(ys - ys[-1]) > YAW_TOL)
        yaw_arrival = int(close + outside[-1] + 1) if len(outside) else close
        return int(max(close, min(settle, yaw_arrival)))
    return int(max(close, settle))


def episode_phases(episode_id: str, embodiment: str, rate: float, source: str | None = None):
    """Episode-level pick-and-place phase timeline from proprio.

    Returns (phase array of str per frame, cycles, failed closes, events, closedness, speed).
    phase strings: grasp:k, move:k, release:k, return, park.
    """
    state, timestamps = episode_arrays(episode_id)
    n = len(state)
    c = closedness(state, embodiment)
    speed = joint_speed(state, rate)
    ee = source in EE_SOURCES
    events = gripper_events(c, rate, STEP_DELTA[embodiment])
    runs = closed_runs(c, events, state, rate, ee=ee)
    disp_min = DISP_MIN_M if ee else DISP_MIN_RAD
    carries, failed = [], []
    for r in runs:
        if r["rest_start"]:
            continue  # closed since the episode start: the parked state, not a carry
        if r["rest_end"] and r["duration_s"] <= 3.0:
            r["park"] = True  # closed to rest right at the end (ARX5 parks shut)
            continue
        if r["disp_rad"] < disp_min:
            failed.append(r)
        else:
            carries.append(r)
    phase = np.array(["grasp:0"] * n, dtype=object)
    cursor = 0
    for k, r in enumerate(carries):
        close = r["close"]["frame"]
        if r["open"] is None:
            open_begin, open_end = n, n
        else:
            open_begin, open_end = r["open"]["ramp_begin"], r["open"]["ramp_end"]
        yaw = np.zeros(n) if ee else state[:, 0]
        arrival = arrival_frame(speed, yaw, close, open_begin) if r["open"] is not None else n
        r["arrival"] = int(arrival)
        r["open_begin"] = int(open_begin)
        r["open_end"] = int(min(open_end, n))
        phase[cursor:close] = f"grasp:{k}"
        phase[close:arrival] = f"move:{k}"
        phase[arrival : r["open_end"]] = f"release:{k}"
        cursor = r["open_end"]
    if cursor < n:
        phase[cursor:n] = "return" if carries else "grasp:0"
    return {
        "n": n,
        "phase": phase,
        "carries": carries,
        "failed": failed,
        "events": events,
        "runs": runs,
        "closedness": c,
        "speed": speed,
        "timestamps": timestamps,
        "state": state,
    }


def runs_of(phase: np.ndarray, start: int, stop: int) -> list[tuple[int, int, str]]:
    out = []
    a = start
    for t in range(start + 1, stop + 1):
        if t == stop or phase[t] != phase[a]:
            out.append((a, t, str(phase[a])))
            a = t
    return out


def merge_short(runs: list[tuple[int, int, str]], rate: float, min_s: float = MIN_ATOM_S):
    """Merge runs shorter than min_s into a neighbour of the same cycle.

    A short release folds back into its move, a short move forward into its release, a
    short grasp forward into its move, a short return back into the last release; a run
    with no same-cycle neighbour folds into whichever neighbour exists (next first).
    """
    runs = list(runs)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (a, b, label) in enumerate(runs):
            if (b - a) / rate >= min_s:
                continue
            kind = label.split(":")[0]
            cycle = label.split(":")[1] if ":" in label else None
            prev_same = i > 0 and runs[i - 1][2].split(":")[-1] == cycle and ":" in runs[i - 1][2]
            next_same = i + 1 < len(runs) and runs[i + 1][2].split(":")[-1] == cycle and ":" in runs[i + 1][2]
            prefer_prev = kind in ("release", "return")
            if prefer_prev and prev_same:
                target = i - 1
            elif (not prefer_prev) and next_same:
                target = i + 1
            elif next_same:
                target = i + 1
            elif prev_same:
                target = i - 1
            elif prefer_prev and i > 0:
                target = i - 1
            elif i + 1 < len(runs):
                target = i + 1
            else:
                target = i - 1
            ta, tb, tl = runs[target]
            runs[target] = (min(a, ta), max(b, tb), tl)
            del runs[i]
            changed = True
            break
    return runs


def parent_proposal(ep: dict, parent: dict, info: dict) -> dict:
    """Candidate atoms for one parent interval from the episode phase timeline."""
    rate = float(ep["native_rate_hz"])
    start, stop = int(parent["start_timestep"]), int(parent["end_timestep_exclusive"])
    raw = runs_of(info["phase"], start, stop)
    merged = merge_short(raw, rate)
    carries = info["carries"]
    atoms = []
    for a, b, label in merged:
        kind = label.split(":")[0]
        k = int(label.split(":")[1]) if ":" in label else None
        verb = {"grasp": "grasp", "move": "move", "release": "release", "return": "return"}[kind]
        provenance_start = "parent" if a == start else ("gripper_event" if kind in ("move", "grasp", "return") else "arm_settle")
        if kind == "grasp" and a != start:
            provenance_start = "gripper_event"  # after the previous opening settled
        atoms.append(
            {
                "start_timestep": int(a),
                "end_timestep_exclusive": int(b),
                "verb": verb,
                "object": None,
                "container": None,
                "cycle": k,
                "start_provenance": provenance_start,
            }
        )
    # failed closes and events inside the parent, for the reviewer
    failed = [
        {
            "close_frame": r["close"]["frame"],
            "open_frame": r["open"]["frame"] if r["open"] else None,
            "disp_rad": r["disp_rad"],
            "duration_s": r["duration_s"],
        }
        for r in info["failed"]
        if start <= r["close"]["frame"] < stop
    ]
    events = [
        {k: e[k] for k in ("kind", "frame", "before", "after", "step", "ramp_begin", "ramp_end")}
        | {"secondary": bool(e.get("secondary", False))}
        for e in info["events"]
        if start <= e["frame"] < stop
    ]
    shut_level = {"ARX5": 0.97, "UR5": 0.98}.get(ep["embodiment"])
    closed_on_nothing = []
    if shut_level is not None and ep["component"] not in ("shred_paper", "arrange_flowers"):
        for k, r in enumerate(carries):
            if r["close"]["frame"] < stop and r["open_end"] > start:
                seg = info["closedness"][r["close"]["frame"] : r["open_begin"]]
                if len(seg) and float(np.median(seg)) >= shut_level:
                    closed_on_nothing.append(k)
    return {
        "closed_on_nothing": closed_on_nothing,
        "episode_id": ep["episode_id"],
        "source": ep["source"],
        "component": ep["component"],
        "embodiment": ep["embodiment"],
        "native_rate_hz": rate,
        "task": ep["task"],
        "parent_interval_index": int(parent["interval_index"]),
        "parent_subtask": parent["normalized_description"],
        "parent_quality": int(parent["quality"]),
        "parent_start": start,
        "parent_end": stop,
        "parent_events": {
            k: parent[k] for k in ("mistake_events", "pause_events", "interruption_events", "recovery_events")
        },
        "critic_eligible": bool(parent["critic_eligible"]),
        "atoms": atoms,
        "failed_closes": failed,
        "gripper_events": events,
        "carries_in_parent": [
            {
                "cycle": k,
                "close": r["close"]["frame"],
                "arrival": r["arrival"],
                "open_begin": r["open_begin"],
                "open_end": r["open_end"],
                "disp_rad": r["disp_rad"],
            }
            for k, r in enumerate(carries)
            if r["close"]["frame"] < stop and r["open_end"] > start
        ],
    }
