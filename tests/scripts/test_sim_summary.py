import json
import shutil
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import lerobot
from lerobot.scripts.compare_probes import build_compare_index, make_compare_handler, resolve_compare_sources
from lerobot.scripts.view_probes import build_index, make_handler

SCRIPT = Path(lerobot.__file__).parents[2] / "scripts" / "probes" / "sim_summary.py"
VIEWS = ("top", "wrist", "wrist_depth")


def _write_sim_tree(run_dir: Path, step: int = 1200) -> Path:
    sim_dir = run_dir / "validation" / f"step_{step:08d}" / "sim_nail"
    for layout, successes in (("ep0", (True, False)), ("ep1", (False, False))):
        (sim_dir / layout / "videos").mkdir(parents=True)
        events = [
            {"episode": ep, "success": ok, "events": [{"t": 3.0, "cursor": 2 if ok else 1, "why": "grasp"}]}
            for ep, ok in enumerate(successes)
        ]
        (sim_dir / layout / "subtask_events.json").write_text(json.dumps(events))
        for ep in range(len(successes)):
            for view in VIEWS:
                (sim_dir / layout / "videos" / f"ep{ep}_{view}.mp4").write_bytes(f"{layout}-{ep}-{view}".encode())
    return sim_dir


def _run_summary(sim_dir: Path, config: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), str(sim_dir), "--config", str(config)],
        check=True, capture_output=True, env={"PYTHONPATH": str(Path(lerobot.__file__).parents[1])},
    )
    return json.loads((sim_dir / "index.json").read_text())


def _serve(handler) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_sim_summary_writes_a_video_manifest_the_viewers_index(tmp_path: Path) -> None:
    config = tmp_path / "config_rl_sim.yaml"
    config.write_text("policy:\n  eval_subtasks: [reach, grasp, place]\n")
    sim_dir = _write_sim_tree(tmp_path / "first")

    index = _run_summary(sim_dir, config)

    assert index["id"] == "sim_nail" and index["group"] == "Actions"
    values = {metric["key"]: metric["value"] for metric in index["metrics"]}
    assert values == {"successes": 1.0, "episodes": 4.0, "mean_steps_done": 1.25}
    files = [panel["file"] for panel in index["panels"]]
    assert files == [f"{layout}/videos/ep{ep}_{view}.mp4" for layout in ("ep0", "ep1") for ep in (0, 1) for view in VIEWS]
    assert [panel["file"] for panel in index["panels"] if panel["primary"]] == ["ep0/videos/ep0_top.mp4", "ep1/videos/ep0_top.mp4"]
    assert all(panel["kind"] == "video" for panel in index["panels"])
    assert index["panels"][0]["caption"] == "ep0 episode 0: top"

    val_dir = sim_dir.parents[1]
    viewer = build_index(val_dir)
    probe = next(p for p in viewer["probes"] if p["id"] == "sim_nail")
    assert probe["has_manifest"] and probe["panels"]["1200"][0]["url"] == "/asset/step_00001200/sim_nail/ep0/videos/ep0_top.mp4"

    server, base = _serve(make_handler(val_dir))
    try:
        assert urlopen(f"{base}/asset/step_00001200/sim_nail/ep1/videos/ep1_wrist_depth.mp4").read() == b"ep1-1-wrist_depth"
    finally:
        server.shutdown()


def test_compare_probes_aligns_sim_videos_across_runs(tmp_path: Path) -> None:
    config = tmp_path / "config_rl_sim.yaml"
    config.write_text("policy:\n  eval_subtasks: [reach, grasp, place]\n")
    first = _write_sim_tree(tmp_path / "first")
    _run_summary(first, config)
    shutil.copytree(tmp_path / "first", tmp_path / "second")
    sources = resolve_compare_sources([tmp_path / "first", tmp_path / "second"])

    index = build_compare_index(sources)

    probes = [next(p for p in run["probes"] if p["id"] == "sim_nail") for run in index["runs"]]
    aligned = [[(panel["align"], panel["url"]) for panel in probe["panels"]["1200"]] for probe in probes]
    assert [key for key, _ in aligned[0]] == [key for key, _ in aligned[1]]
    assert aligned[0][0] == ("ep0/videos/ep0_top.mp4", "/asset/0/step_00001200/sim_nail/ep0/videos/ep0_top.mp4")
    assert aligned[1][0] == ("ep0/videos/ep0_top.mp4", "/asset/1/step_00001200/sim_nail/ep0/videos/ep0_top.mp4")

    server, base = _serve(make_compare_handler(sources))
    try:
        assert urlopen(f"{base}/asset/1/step_00001200/sim_nail/ep0/videos/ep0_top.mp4").read() == b"ep0-0-top"
    finally:
        server.shutdown()
