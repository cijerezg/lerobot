"""Probe recorded inference episodes with the checkpoint that drove them.

    uv run python lerobot/scripts/probes/inference_episode.py outputs/train/<day>/<run> [more runs]

Each run dir (or its inference_dataset) becomes a probe input under outputs/probe_inputs/ (data,
videos, depth symlinked; meta copied plus subtask windows from the operator console labels and
the depth gripper-event labels), gets a config derived from config_rl_validate.yaml with only
--probes enabled, and runs through run_probes.sh into outputs/probe_runs/<run>-<step>.
The checkpoint is config_rl.yaml's policy.pretrained_path (inference_checkpoint_path if set) and
must match the one the actor log says it loaded.
"""
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path

import pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[3]
ANNOTATE = ROOT / "lerobot/src/lerobot/data_processing/annotate/depth_gripper_event_annotate.py"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("runs", nargs="+", type=Path, help="outputs/train/<day>/<run> or its inference_dataset")
parser.add_argument("--probes", nargs="+", default=["attention", "action_trace", "depth_event"])
parser.add_argument("--checkpoint", type=Path, default=None, help="override the config's pretrained_model dir")
parser.add_argument("--trace-stride-s", type=float, default=3.0)
args = parser.parse_args()

cfg_train = yaml.safe_load((ROOT / "config_rl.yaml").read_text())
checkpoint = args.checkpoint or Path(cfg_train.get("inference_checkpoint_path") or cfg_train["policy"]["pretrained_path"])
checkpoint = (ROOT / checkpoint).resolve()
step = checkpoint.parent.name


def build_input(src: Path, out: Path) -> None:
    out.mkdir(parents=True)
    for sub in ("data", "videos", "depth"):
        (out / sub).symlink_to(src / sub)
    shutil.copytree(src / "meta", out / "meta")
    labels = pd.read_parquet(src / "meta/online_labels.parquet")
    windows = {}
    for episode, ep in labels.groupby("episode_index"):
        runs = (ep["subtask"] != ep["subtask"].shift()).cumsum()
        windows[str(int(episode))] = [
            {"from_index": int(r["index"].iloc[0]), "to_index": int(r["index"].iloc[-1]) + 1, "subtask": r["subtask"].iloc[0]}
            for _, r in ep.groupby(runs)
        ]
    (out / "meta/subtask_windows.json").write_text(
        json.dumps({"annotator": "operator console (online_labels.parquet)", "episodes": windows}, indent=1)
    )
    names = sorted(labels["subtask"].unique())
    pd.DataFrame({"subtask_index": range(len(names))}, index=pd.Index(names, name="subtask")).to_parquet(out / "meta/subtasks.parquet")
    subprocess.run([sys.executable, str(ANNOTATE), "--data-dir", str(out), "--plot-all-episodes"], check=True)


def write_config(input_dir: Path, run_dir: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "config_rl_validate.yaml").read_text())
    cfg["val_dataset_path"] = str(input_dir.relative_to(ROOT))
    cfg["offline_output_dir"] = cfg["output_dir"] = str(run_dir.relative_to(ROOT))
    p = cfg["probe_parameters"]
    for key in p:
        if key.startswith("enable_"):
            p[key] = key.removeprefix("enable_") in args.probes
    p["max_episodes"] = 1
    p["trace_anchor_stride_s"] = args.trace_stride_s
    p["trace_max_anchors_per_episode"] = 64
    path = input_dir / "config_probe.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


for run in args.runs:
    run = run.resolve()
    if run.name == "inference_dataset":
        run = run.parent
    for log in (run / "logs").glob("actor_*.log"):
        loaded = re.search(r"Loading policy and processors from checkpoint: (\S+)", log.read_text())
        assert loaded is None or Path(loaded.group(1)).resolve() == checkpoint, f"{run.name} ran {loaded.group(1)}, not {checkpoint}"
    name = f"{run.name}-{step}"
    input_dir, run_dir = ROOT / "outputs/probe_inputs" / name, ROOT / "outputs/probe_runs" / name
    if not input_dir.exists():
        build_input(run / "inference_dataset", input_dir)
    config = write_config(input_dir, run_dir)
    subprocess.run([str(ROOT / "lerobot/scripts/run_probes.sh"), str(checkpoint), str(run_dir), str(config.relative_to(ROOT))], check=True, cwd=ROOT)
