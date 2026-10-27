"""Frame-level speed (hybrid_motion_speed.frame_runs) and the two writers built on it."""
import json

import numpy as np
import pandas as pd

from lerobot.annotation.speed import hybrid_diverse
from lerobot.annotation.speed import hybrid_motion_speed as hybrid
from lerobot.annotation.speed import rebot_speed_pass

RATE = 10.0  # 1 s = 10 frames


def _labels(runs, n):
    out = np.zeros(n, dtype=int)
    for a, b, label, _ in runs:
        out[a:b] = label
    return out


def test_long_stretches_keep_the_rounded_score():
    scores = np.r_[np.full(30, 1.2), np.full(30, 2.8), np.full(30, 4.6)]
    runs = hybrid.frame_runs(scores, RATE)
    assert [(a, b, k) for a, b, k, _ in runs] == [(0, 30, 1), (30, 60, 3), (60, 90, 5)]
    assert np.isclose(runs[1][3], 2.8)


def test_a_run_shorter_than_one_second_takes_the_closer_neighbour():
    # 0.5 s at score 2.6 between a slow (1) and a fast (4) stretch: |2.6 - 4| = 1.4 < |2.6 - 1| = 1.6, so it joins the fast side
    scores = np.r_[np.full(20, 1.0), np.full(5, 2.6), np.full(20, 4.0)]
    labels = _labels(hybrid.frame_runs(scores, RATE), len(scores))
    assert (labels[:20] == 1).all() and (labels[20:] == 4).all()


def test_no_run_is_shorter_than_one_second_and_runs_tile_the_stretch():
    rng = np.random.default_rng(0)
    scores = np.clip(3 + np.cumsum(rng.normal(0, 0.3, 500)), 1, 5)
    runs = hybrid.frame_runs(scores, RATE)
    assert runs[0][0] == 0 and runs[-1][1] == len(scores)
    assert all(b == a2 for (_, b, _, _), (a2, _, _, _) in zip(runs, runs[1:]))
    assert all(b - a >= RATE for a, b, _, _ in runs)
    assert all(k1 != k2 for (_, _, k1, _), (_, _, k2, _) in zip(runs, runs[1:]))


def test_unscored_frames_count_as_three_and_a_short_stretch_is_one_run():
    assert [(a, b, k) for a, b, k, _ in hybrid.frame_runs(np.full(4, np.nan), RATE)] == [(0, 4, 3)]
    # the episode's last frame has no outgoing transition: absorbed by its run
    runs = hybrid.frame_runs(np.r_[np.full(30, 1.0), np.nan], RATE)
    assert [(a, b, k) for a, b, k, _ in runs] == [(0, 31, 1)]


def test_rebot_runs_root_tiles_every_episode(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": RATE}))
    # two episodes of 40 and 30 frames; the trace has one entry per transition (69), the join is invalid
    score = np.r_[np.full(20, 1.0), np.full(19, 4.0), np.nan, np.full(29, 2.0)]
    np.savez_compressed(meta / "speed_hybrid_v1_trace.npz", speed_score=score)
    pd.DataFrame([{"episode_index": 0, "segment_index": 0, "from_index": 0, "to_index": 40},
                  {"episode_index": 1, "segment_index": 0, "from_index": 40, "to_index": 70}]).to_parquet(meta / "episode_metadata.parquet")
    rebot_speed_pass.runs_root(tmp_path)
    table = pd.read_parquet(meta / "speed_hybrid_v1_runs.parquet")
    assert table[["episode_index", "from_index", "to_index", "speed"]].values.tolist() == [[0, 0, 20, 1], [0, 20, 40, 4], [1, 40, 70, 2]]
    assert json.loads((meta / "speed_hybrid_v1_runs_info.json").read_text())["rows"] == 3


def test_diverse_runs_follow_contiguous_atoms_and_stop_at_gaps(tmp_path):
    trace = tmp_path / "ep.npz"
    score = np.full(59, np.nan)
    score[0:29] = 1.0  # atoms [0, 15) + [15, 30): one stretch
    score[40:59] = 5.0  # atom [40, 60) after a gap
    np.savez_compressed(trace, speed_score=score, native_rate_hz=RATE)
    atoms = [dict(episode_id="ep", source="droid", embodiment="Franka", group="droid", start_timestep=a, end_timestep_exclusive=b)
             for a, b in ((15, 30), (0, 15), (40, 60))]
    rows = hybrid_diverse.runs_rows(atoms, lambda eid: trace)
    assert [(r["start_timestep"], r["end_timestep_exclusive"], r["speed"]) for r in rows] == [(0, 30, 1), (40, 60, 5)]
    assert [r["run_index"] for r in rows] == [0, 1]
