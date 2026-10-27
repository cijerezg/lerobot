"""Hybrid motion+duration speed labels (the adopted method) for a diverse corpus root.

    DIVERSE_DATASET_ROOT=outputs/diverse_robot_dataset_v3 uv run python -m lerobot.annotation.speed.hybrid_diverse

Same arithmetic as the first hybrid pass (v6, migration/subtask_atoms_2026-09-08), without the
v4 -> v5 -> v6 chain: the motion pools are fit here from the training atoms of this root, the
duration labels come from this root's speed_atoms.jsonl. Writes only new files: the reference,
speed_hybrid_v1/<episode>.npz, speed_atoms_hybrid_v1.jsonl (one row per atom) and
speed_runs_hybrid_v1.jsonl (the frame-level label the trainer reads, SPEED_ATOMS_VIEW) per store.

The runs sidecar alone, for a store that already has the atom sidecar (2026-10-26):

    uv run python -m lerobot.annotation.speed.hybrid_diverse runs <store root> [<trace map .json>]

traces default to <store>/<corpus|fmb>/speed_hybrid_v1/<episode>.npz; the map ({"corpus/<episode>": path, ...})
names them for a store assembled without that folder.
"""
from __future__ import annotations
import collections, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from lerobot.annotation.speed import hybrid_motion_speed as hybrid
from lerobot.annotation.speed import joint_motion_speed as motion
from lerobot.annotation.speed import speed_annotate as speed

DATA = Path(os.environ.get("DIVERSE_DATASET_ROOT", "outputs/diverse_robot_dataset_v3"))
REF = Path(os.environ.get("SPEED_HYBRID_REF", "outputs/stats/speed_reference_hybrid_motion_duration_v1_diverse-v3.json"))
SIDE = "speed_atoms_hybrid_v1.jsonl"
RUNS = "speed_runs_hybrid_v1.jsonl"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r): return speed._diverse_key(r)


def main():
    assert not REF.exists(), f"{REF} exists; preserve it before regenerating"
    segments = speed.diverse_segments(DATA)
    by_episode = collections.defaultdict(list)
    for s in segments: by_episode[(Path(s["root"]).name, s["episode_id"])].append(s)
    duration = {}
    for sub in ("corpus", "fmb"):
        for r in map(json.loads, open(DATA / sub / "speed_atoms.jsonl")): duration[(sub, *key(r))] = r
    # motion pools from training atoms, per robot group (an EE-pose source is its own group)
    traces, training, state_sha = {}, collections.defaultdict(list), {}
    for (sub, eid), members in by_episode.items():
        path = Path(members[0]["episode_dir"]) / ("q.npy" if sub == "fmb" else "state.npy")
        q = np.load(path)
        if sub != "fmb": q = q[:, :-1]
        values, valid = motion.motion_trace(q, members[0]["native_rate_hz"])
        traces[(sub, eid)] = (q, values, valid); state_sha[str(path)] = sha(path)
        for s in members:
            if s["split"] == "train":
                a, b = s["start_timestep"], s["end_timestep_exclusive"]
                training[s["group"]].append(values[a:b - 1][valid[a:b - 1]])
    pools = {g: np.sort(np.concatenate(v)) for g, v in training.items()}
    reference = {"method": hybrid.METHOD, "root": str(DATA),
        "motion_units": "L2 of per-step state differences times the native rate; radians/s for joint sources, mixed metres and radians for end-effector sources (molmoact), each pooled only within its own group",
        "groups": {g: {"n_training_transitions": int(len(v)), "quantiles_0.2_0.4_0.6_0.8": np.quantile(v, [.2, .4, .6, .8]).round(4).tolist()} for g, v in pools.items()},
        "local_smoothing_seconds": hybrid.LOCAL_SECONDS, "final_smoothing_seconds": hybrid.FINAL_SECONDS,
        "duration_weight": hybrid.DURATION_WEIGHT, "maximum_duration_adjustment_buckets": hybrid.MAX_DURATION_ADJUSTMENT,
        "formula": "motion_score=clip(0.5+5*training_state_percentile,1,5); adjustment=clip(0.2*(duration_speed-motion_score),-0.35,0.35); smooth(motion_score+adjustment); speed=round(score)",
        "duration_default_policy": "Ignore default/unclear duration labels", "quality_used": False,
        "atom_aggregation": "round(median continuous hybrid score)",
        "duration_sidecar_sha256": {s: sha(DATA / s / "speed_atoms.jsonl") for s in ("corpus", "fmb")},
        "annotation_sha256": {s: sha(DATA / s / "subtask_atoms.jsonl") for s in ("corpus", "fmb")},
        "state_sha256": state_sha}
    REF.write_text(json.dumps(reference, indent=1) + "\n")
    rows = []
    for (sub, eid), members in by_episode.items():
        q, values, valid = traces[(sub, eid)]; rate = members[0]["native_rate_hz"]; group = members[0]["group"]
        atoms = []
        for r in members:
            d = duration[(sub, *key(r))]
            atoms.append(r | {"duration_speed": d["speed"], "use_duration": d["speed_source"] not in ("default", "default_unclear")})
        trace = hybrid.hybrid_trace(q, rate, atoms, pools[group]); mask = trace["supervision_mask"]; ok = trace["valid"]
        label = np.zeros(len(mask), dtype=np.uint8); label[mask] = 3; label[mask & ok] = hybrid.quantize(trace["speed_score"][mask & ok])
        path = DATA / sub / "speed_hybrid_v1" / f"{eid}.npz"; path.parent.mkdir(exist_ok=True); assert not path.exists(), path
        np.savez_compressed(path, **trace, speed=label, native_rate_hz=rate)
        for r, atom in zip(members, atoms, strict=True):
            a, b = r["start_timestep"], r["end_timestep_exclusive"]; v = trace["speed_score"][a:b - 1]
            reason = "fewer than two state transitions in the atom" if len(v) < 2 else ("nonfinite state data" if not np.isfinite(v).all() else "")
            score = float(np.median(v)) if not reason else 3.0
            motion_score = float(np.median(trace["motion_score"][a:b - 1])) if not reason else 3.0
            measured = float(np.median(trace["joint_speed_rad_s"][a:b - 1])) if not reason else None
            flags = [reason] if reason else []
            if abs(int(hybrid.quantize(motion_score)) - atom["duration_speed"]) >= 2: flags.append("motion and duration differ by at least two buckets; adjustment capped")
            row = {k: r[k] for k in ("episode_id", "source", "embodiment", "group", "class", "parent_interval_index", "atom_index", "annotation_layer", "confidence", "start_timestep", "end_timestep_exclusive", "subtask", "duration_s")}
            row.update(method=hybrid.METHOD, speed=3 if reason else int(hybrid.quantize(score)), speed_score=score,
                speed_source="default_unclear" if reason else hybrid.METHOD, speed_default_reason=reason, speed_flags=flags,
                duration_speed=atom["duration_speed"], duration_used=atom["use_duration"],
                motion_speed=int(hybrid.quantize(motion_score)), motion_score=motion_score, motion_median_rad_s=measured, state_speed_file=str(path))
            rows.append((sub, row))
    report = {"reference": str(REF), "stores": {}}
    for sub in ("corpus", "fmb"):
        picked = [row for s, row in rows if s == sub]; target = DATA / sub / SIDE; assert not target.exists(), target
        target.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in picked))
        write_runs(DATA / sub / RUNS, picked, lambda eid: DATA / sub / "speed_hybrid_v1" / f"{eid}.npz")
        by_source = collections.defaultdict(collections.Counter)
        for r in picked: by_source[r["source"]][r["speed"]] += 1
        report["stores"][sub] = {"rows": len(picked), "defaults": sum(r["speed_source"] == "default_unclear" for r in picked),
            "flagged": sum(bool(r["speed_flags"]) for r in picked), "buckets_by_source": {s: dict(sorted(c.items())) for s, c in by_source.items()}}
    print(json.dumps(report, indent=1))


def runs_rows(atom_rows, trace_of):
    """Frame-level speed (hybrid.frame_runs) over every stretch of contiguous atoms of each episode. Frame t carries the
    trace's t -> t+1 score; the stretch's last frame has none (counted as 3, absorbed by its run)."""
    by_episode = collections.defaultdict(list)
    for r in atom_rows: by_episode[r["episode_id"]].append(r)
    out = []
    for eid, members in by_episode.items():
        members = sorted(members, key=lambda r: r["start_timestep"]); path = trace_of(eid); trace = np.load(path)
        stretches = []
        for r in members:
            if stretches and stretches[-1][1] == r["start_timestep"]: stretches[-1][1] = r["end_timestep_exclusive"]
            else: stretches.append([r["start_timestep"], r["end_timestep_exclusive"]])
        k = 0
        for a, b in stretches:
            for s, e, label, mean in hybrid.frame_runs(np.append(trace["speed_score"][a:b - 1], np.nan), float(trace["native_rate_hz"])):
                out.append({**{f: members[0][f] for f in ("episode_id", "source", "embodiment", "group")}, "run_index": k,
                            "start_timestep": a + s, "end_timestep_exclusive": a + e, "speed": label, "speed_score": mean,
                            "method": hybrid.METHOD, "speed_source": "frame_runs", "state_speed_file": str(path)}); k += 1
    return out


def write_runs(target, atom_rows, trace_of):
    assert not target.exists(), target
    rows = runs_rows(atom_rows, trace_of)
    target.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    print(f"{target}: {len(rows)} runs over {len({r['episode_id'] for r in rows})} episodes")


def runs_main(store, trace_map=None):
    paths = json.loads(Path(trace_map).read_text()) if trace_map else {}
    for sub in ("corpus", "fmb"):
        atom_rows = [json.loads(line) for line in open(store / sub / SIDE)]
        write_runs(store / sub / RUNS, atom_rows, lambda eid: Path(paths.get(f"{sub}/{eid}", store / sub / "speed_hybrid_v1" / f"{eid}.npz")))


if __name__ == "__main__":
    if sys.argv[1:2] == ["runs"]: runs_main(Path(sys.argv[2]), *sys.argv[3:4])
    else: main()
