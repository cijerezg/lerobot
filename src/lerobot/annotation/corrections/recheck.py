"""A second labelling of frames that earlier correction passes already corrected, to measure how far labels move.

  uv run python -m lerobot.annotation.corrections.recheck pick <work> <pass> [<pass> ...] [--n 32] [--seed 0]
      draws N corrected frames (pass the check, wrist depth on disk) at random over the passes, each from another
      episode. Writes one pass folder per source pass, ``<work>/<pass name>/`` with its ``pass.json``,
      ``stalls.json`` and the drawn rows of its ``frames.parquet`` (ready for ``dump``), and ``<work>/picked.parquet``
      (pass, source, episode, frame, the first labelling's folder).
  uv run python -m lerobot.annotation.corrections.recheck compare <work>
      after ``build`` on every pass folder: ``<work>/compare.csv``, one row per frame, the first labelling against
      the second (contact, contact-pose tip, tilt, gripper, the corrected chunk), and a summary on stdout.
  uv run python -m lerobot.annotation.corrections.recheck page <work>
      the built frames in the Action Inspector page (``inspector``'s), both labellings on each frame ->
      ``<work>/inspector.html``
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.corrections.arm import Arm


def pick(work: Path, passes: list[Path], n: int, seed: int) -> None:
    pool = pd.concat([pd.read_parquet(p / "corrections.parquet").assign(first=str(p), name=p.name) for p in passes], ignore_index=True)
    pool = pool[pool.ok].sample(frac=1.0, random_state=seed)
    rows, seen = [], set()
    for r in pool.itertuples():
        if (r.root, r.episode) in seen:
            continue
        if any(list(Path(r.root, "depth").glob(f"*/episode-{r.episode:06d}/frame-{f:06d}.png")) for f in (r.frame, r.frame - 1, r.frame + 1)):
            seen.add((r.root, r.episode))
            rows.append(dict(name=r.name, source=r.source, episode=r.episode, frame=r.frame, first=f"{r.first}/{r.source}/ep{r.episode:03d}/f{r.frame:06d}"))
        if len(rows) == n:
            break
    picked = pd.DataFrame(rows)
    work.mkdir(parents=True)
    picked.to_parquet(work / "picked.parquet")
    for p in passes:
        mine = picked[picked.name == p.name]
        if len(mine):
            (work / p.name).mkdir()
            shutil.copy(p / "pass.json", work / p.name / "pass.json")
            shutil.copy(p / "stalls.json", work / p.name / "stalls.json")
            frames = pd.read_parquet(p / "frames.parquet").merge(mine[["source", "episode", "frame"]])
            frames.to_parquet(work / p.name / "frames.parquet")
    print(picked.to_string())


def first_pose(arm: Arm, folder: Path) -> dict:
    spec = json.loads((folder / "waypoints.json").read_text())
    chunk = np.load(folder / "correction.npy")
    w = spec["waypoints"][0]
    if "like_frame" in w:
        tip, approach, jaw = arm.gripper.fk(np.load(folder / "manoeuvre.npy")[-1])
    else:
        tip, approach = np.asarray(w["tip"], dtype=float), np.asarray(w.get("approach", [0, 0, -1]), dtype=float)
        jaw = np.asarray(w["jaw"], dtype=float) if "jaw" in w else None
    approach = approach / np.linalg.norm(approach)
    return dict(contact=spec["contact"], tip=tip, tilt=float(np.degrees(np.arccos(-approach[2]))), approach=approach, jaw=jaw, chunk=chunk,
                end=arm.gripper.fk(chunk[-1])[0], gripper=float(chunk[-1][6]), note=spec.get("note", ""))


def compare(work: Path) -> None:
    arm, rows = Arm(), []
    for r in pd.read_parquet(work / "picked.parquet").itertuples():
        second = work / r.name / r.source / f"ep{r.episode:03d}" / f"f{r.frame:06d}"
        row = dict(name=r.name, source=r.source, episode=r.episode, frame=r.frame)
        rows.append(row)
        if not (second / "correction.npy").exists():
            row["status"] = (json.loads((second / "waypoints.json").read_text()).get("skip", "not built") if (second / "waypoints.json").exists() else "not labelled")
            continue
        a, b = first_pose(arm, Path(r.first)), first_pose(arm, second)
        d = b["tip"] - a["tip"]
        row.update(
            status="ok", contact_first=a["contact"], contact_second=b["contact"],
            tip_shift_cm=100 * float(np.linalg.norm(d)), tip_dxy_cm=100 * float(np.linalg.norm(d[:2])), tip_dz_cm=100 * float(d[2]),
            tilt_first=a["tilt"], tilt_second=b["tilt"],
            approach_angle_deg=float(np.degrees(np.arccos(np.clip(a["approach"] @ b["approach"], -1, 1)))),
            chunk_end_shift_cm=100 * float(np.linalg.norm(b["end"] - a["end"])),
            chunk_gripper_first=a["gripper"], chunk_gripper_second=b["gripper"],
            chunk_joint_rms_deg=float(np.sqrt(((a["chunk"] - b["chunk"])[:, :6] ** 2).mean())),
            note_first=a["note"], note_second=b["note"],
        )
    table = pd.DataFrame(rows)
    table.to_csv(work / "compare.csv", index=False)
    ok = table[table.status == "ok"]
    print(table.status.value_counts().to_string())
    print(f"same contact: {(ok.contact_first == ok.contact_second).sum()} of {len(ok)}")
    print(ok[["tip_shift_cm", "tip_dxy_cm", "tip_dz_cm", "approach_angle_deg", "chunk_end_shift_cm", "chunk_joint_rms_deg"]].describe().round(1).to_string())


def write_page(work: Path, cases: list[dict], names: list[str], title: str, subtitle: str) -> None:
    """``<work>/inspector.html``, the Action Inspector page: per case (label, episode, frame, npz = the
    ``action_trace_records.npz`` that holds the frame, folders = one built frame folder per name) the recorded chunk
    against the chunk of every folder."""
    from types import SimpleNamespace

    import torch

    from lerobot.annotation.corrections.arm import FPS
    from lerobot.probes.action_trace_probe import _analyse, _figure, _image_data_uri, _resolve_actuator_config, _write_dashboard_html
    from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics

    kin = RebotKinematics()
    speeds, lower, upper = _resolve_actuator_config(None, 7)
    records = []
    for c in cases:
        d = dict(np.load(c["npz"]))
        i = next(i for i, (e, f) in enumerate(zip(d["episode"], d["frame"], strict=True)) if (int(e), int(f)) == (c["episode"], c["frame"]))
        specs = [json.loads((folder / "waypoints.json").read_text()) for folder in c["folders"]]
        record = _analyse(
            kin, d["state"][i], d["gt_chunk"][i], [np.load(folder / "correction.npy") for folder in c["folders"]], 0.0,
            fps=float(FPS), motor_speeds_deg_s=speeds, joint_lower=lower, joint_upper=upper,
        )
        record.update(
            episode=c["episode"], frame=c["frame"], global_idx=int(d["global_idx"][i]), subtask=str(d["subtask"][i]) if "subtask" in d else "",
            trace_label=c["label"], sample_names=names,
            cameras=[{"label": cam, "src": _image_data_uri(torch.from_numpy(d[f"image_{cam}"][i]))} for cam in ("top", "wrist")],
            notes="\n\n".join(
                f"{name.upper()}\ncontact: {spec.get('contact')}\nintent: {spec.get('intent')}\nwrong: {spec.get('wrong')}\nplan: {spec.get('note')}\n"
                f"first waypoint: {json.dumps(spec['waypoints'][0])}"
                for name, spec in zip(names, specs, strict=True)
            ),
        )
        records.append(record)
    fig = _figure(records, SimpleNamespace(trace_table_z=0.0, trace_clearance_warn_m=0.01), fps=float(FPS))
    for trace in [*fig.data, *(t for f in fig.frames for t in f.data)]:
        if trace.name == "ground truth targets":
            trace.name = "recorded chunk"
    _write_dashboard_html(
        fig, records, str(work / "inspector.html"), page_title=title, subtitle=subtitle,
        legend_note="Black = recorded chunk; the samples are " + ", ".join(names) + ".",
        contexts=[{"label": r["trace_label"], "subtask": r["subtask"], "cameras": r["cameras"], "notes": r["notes"]} for r in records],
    )
    print("wrote", work / "inspector.html", len(records), "frames")


def page(work: Path) -> None:
    cases = []
    for r in pd.read_parquet(work / "picked.parquet").itertuples():
        second = work / r.name / r.source / f"ep{r.episode:03d}" / f"f{r.frame:06d}"
        if (second / "correction.npy").exists():
            cases.append(dict(
                label=f"{r.name} · {r.source} ep{r.episode} · frame {r.frame}", episode=r.episode, frame=r.frame,
                npz=work / r.name / r.source / "action_trace_records.npz", folders=[Path(r.first), second],
            ))
    write_page(
        work, cases, ["first labelling", "second labelling (depth)"], "Correction recheck",
        "One frame per step: the recorded 1 s chunk against the correction of the first labelling and of the second.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pick", "compare", "page"])
    parser.add_argument("work", type=Path)
    parser.add_argument("passes", type=Path, nargs="*")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "pick":
        pick(args.work, args.passes, args.n, args.seed)
    elif args.command == "compare":
        compare(args.work)
    else:
        page(args.work)


if __name__ == "__main__":
    main()
