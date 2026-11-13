"""Test the correction labeller against grasps that worked: it sees a demonstration up to a frame shortly before
the gripper closes on the object, writes the grasp, and the measured tool pose at the grasp is the answer.

  uv run python -m lerobot.annotation.corrections.grasp_test pick <work> <pass> [--n 32] [--per 3] [--lead 45] [--seed 0]
      from the demonstration roots of ``<pass>/pass.json``: the grasps that worked (``grasps``) with wrist depth
      on disk. The frame is ``lead`` frames before the gripper starts to close. At most ``per`` per object (``OBJECTS``), one per episode. Writes the pass
      folders ``<work>/coarse`` and ``<work>/dense`` (the two arms: the same frames, ``"causal": true`` in their
      ``pass.json``, for ``dump`` and ``dump --depth``) and ``<work>/key``: the answer as a pass of its own, the
      measured pose at the grasp as each frame's ``waypoints.json``, and ``picked.parquet`` (grasp frame, distance
      left at the frame).
  uv run python -m lerobot.annotation.corrections.grasp_test arm <work> <name>
      one more arm on the same frames: the pass folder ``<work>/<name>``, ready for ``dump``.
  uv run python -m lerobot.annotation.corrections.grasp_test score <work>
      after ``build`` on the pass folders: ``<work>/score.csv``, one row per frame and arm, the labelled contact
      pose against the measured one, and a summary on stdout.
  uv run python -m lerobot.annotation.corrections.grasp_test audit <out> <pass> [<pass> ...]
      built correction passes against their own episodes: ``<out>/audit.csv``, one row per corrected frame of a
      ``grasp ...`` subtask that the episode goes on to grasp (its anchor), the labelled contact pose against that
      grasp's.
  uv run python -m lerobot.annotation.corrections.grasp_test anchored <work> <pass> [<pass> ...] [--n 48] [--seed 0]
      the anchor test, after ``audit <work> <pass> ...``: ``n`` frames of ``<work>/audit.csv`` at least 3 s before
      their anchor, one per episode, a third with no closing attempt in between. Writes the pass folders
      ``<work>/label`` (not causal: for ``dump --depth`` and the labeller), ``<work>/key`` (the anchor's pose as
      each frame's ``waypoints.json``) and ``<work>/try`` (for the frames with a failed attempt: the pose where the
      last one closed), and ``<work>/key/picked.parquet``.
  uv run python -m lerobot.annotation.corrections.grasp_test page <work>
      ``<work>/inspector.html``: per frame the recorded chunk, the arms' chunks and the chunk to the real grasp.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.corrections.anchor import anchor, closings
from lerobot.annotation.corrections.arm import Arm
from lerobot.annotation.corrections.recheck import first_pose, write_page
from lerobot.annotation.corrections.source import Episode, sources

OBJECTS = ["bit", "clipper", "bottle", "cup", "sock", "shirt", "flap", "lid", "tape roll", "cap", "eraser", "sugar cube", "floss"]
LEFT_M = 0.02  # distance left at the frame


def grasps(arm: Arm, e: Episode, lead: int) -> list[dict]:
    """The worked grasps of the episode to test on: graded 4-5 from ``lead`` frames before the closing starts (the
    frame to label, in the same subtask) to the grasp, the tool still ``LEFT_M`` away at that frame, depth on disk."""
    found = []
    for close, stop in closings(arm, e):
        t = close - lead
        if t < 0 or e.subtasks[close] != e.subtasks[t] or e.grades[t : stop + 1].min() < 4:
            continue
        tip = arm.gripper.fk(e.states[stop])[0]
        left = float(np.linalg.norm(arm.gripper.fk(e.states[t])[0] - tip))
        if left >= LEFT_M and e.depth_m(t) is not None:
            found.append(dict(
                source=e.source.name, episode=e.index, frame=t, kind="demo", grade=int(e.grades[t]), subtask=str(e.subtasks[t]),
                contact=str(e.contacts[t]), precision=str(e.precisions[t]), close_frame=close, grasp_frame=stop, left_cm=100 * left,
            ))
    return found


def pick(work: Path, source_pass: Path, n: int, per: int, lead: int, seed: int) -> None:
    spec = json.loads((source_pass / "pass.json").read_text())
    spec = dict(spec, sources=[s for s in spec["sources"] if s["kind"] == "rebot"], causal=True)
    arm, found = Arm(), []
    for source in sources(spec).values():
        for index in source.bounds:
            found += grasps(arm, Episode(source, index), lead)
    pool = pd.DataFrame(found).sample(frac=1.0, random_state=seed)
    pool["object"] = [next((o for o in OBJECTS if o in subtask), subtask) for subtask in pool.subtask]
    pool = pool.drop_duplicates(["source", "episode"]).groupby("object", sort=False).head(per)
    pool = pool.assign(turn=pool.groupby("object").cumcount()).sort_values("turn", kind="stable").head(n)  # every object once first
    picked = pool.drop(columns="turn").sort_values(["source", "episode"]).reset_index(drop=True)
    stalls = (source_pass / "stalls.json").read_text()
    for name in ["coarse", "dense", "key"]:
        (work / name).mkdir(parents=True)
        (work / name / "pass.json").write_text(json.dumps(spec, indent=1))
        (work / name / "stalls.json").write_text(stalls)
        picked[["source", "episode", "frame", "kind", "grade", "subtask", "contact", "precision"]].to_parquet(work / name / "frames.parquet")
    picked.to_parquet(work / "key" / "picked.parquet")
    by_name = sources(spec, list(picked.source.unique()))
    for r in picked.itertuples():
        folder = work / "key" / r.source / f"ep{r.episode:03d}" / f"f{r.frame:06d}"
        folder.mkdir(parents=True)
        opening = float(Episode(by_name[r.source], r.episode).states[r.close_frame][6])
        (folder / "waypoints.json").write_text(json.dumps(dict(
            contact=r.contact, intent=r.subtask, wrong="nothing", note=f"the measured pose at frame {r.grasp_frame}, where the gripper stops closing",
            waypoints=[dict(like_frame=r.grasp_frame, gripper=opening, arrive="approach")],
        )))
    print(picked.drop(columns=["kind", "precision"]).to_string())
    print(f"{len(pool)} of {len(found)} grasps, {picked.object.nunique()} objects")


def audit(out: Path, passes: list[Path]) -> None:
    """Existing corrections against what the episode went on to do: for every built frame that has an anchor
    (``anchor.anchor``: a ``grasp ...`` subtask whose episode later grasps in that same stretch), the correction's
    contact pose against the measured pose at that grasp. ``tried`` = a closing attempt failed in between;
    ``try_cm`` = how far from the grasp the tool was when the last such attempt closed."""
    arm, rows = Arm(), []
    for p in passes:
        built = pd.read_parquet(p / "corrections.parquet")
        built = built[built.ok & built.subtask.astype(str).str.startswith("grasp")]
        by_name = sources(json.loads((p / "pass.json").read_text()), list(built.source.unique()))
        for (source, index), frames in built.groupby(["source", "episode"]):
            e = Episode(by_name[source], index)
            worked = closings(arm, e)
            for r in frames.itertuples():
                a = anchor(e, worked, r.frame)
                if a is None:
                    continue
                folder = p / source / f"ep{index:03d}" / f"f{r.frame:06d}"
                mine, truth, now = first_pose(arm, folder), arm.gripper.fk(e.states[a["frame"]]), arm.gripper.fk(e.states[r.frame])[0]
                d = mine["tip"] - truth[0]
                rows.append(dict(
                    name=p.name, source=source, episode=index, frame=r.frame, kind=r.kind, grade=r.grade, subtask=r.subtask, contact=mine["contact"],
                    seconds_to_grasp=(a["frame"] - r.frame) / 30, left_cm=100 * float(np.linalg.norm(now - truth[0])), tried=a["tried"],
                    try_cm=100 * float(np.linalg.norm(arm.gripper.fk(e.states[a["try_frame"]])[0] - truth[0])) if a["tried"] else np.nan,
                    tip_error_cm=100 * float(np.linalg.norm(d)), xy_error_cm=100 * float(np.linalg.norm(d[:2])), z_error_cm=100 * float(d[2]),
                    approach_error_deg=float(np.degrees(np.arccos(np.clip(mine["approach"] @ truth[1], -1, 1)))), inconsistent=r.inconsistent,
                ))
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "audit.csv", index=False)
    columns = ["seconds_to_grasp", "left_cm", "try_cm", "tip_error_cm", "xy_error_cm", "z_error_cm", "approach_error_deg"]
    print(f"{len(table)} frames\n{table[columns].describe().round(1).to_string()}")
    print(table.groupby(["kind", "tried"])[["left_cm", "tip_error_cm"]].agg(["count", "median"]).round(1).to_string())


def anchored(work: Path, passes: list[Path], n: int, seed: int) -> None:
    pool = pd.read_csv(work / "audit.csv")
    pool = pool[(pool.seconds_to_grasp >= 3) & (pool.source != "rebot_2026-11-10b-annotated-v1")].sample(frac=1.0, random_state=seed)
    pool = pool.drop_duplicates(["source", "episode"])
    specs = {p.name: json.loads((p / "pass.json").read_text()) for p in passes}
    spec = {k: v for one in specs.values() for k, v in one.items() if k not in ("done_every", "demo_grades", "demo_skip_verbs")}
    for key in ("contact", "top", "cameras"):
        spec[key] = {k: v for one in specs.values() for k, v in one.get(key, {}).items()}
    spec = dict(spec, every=30, sources=list({s["name"]: s for one in specs.values() for s in one["sources"]}.values()))
    by_name, arm = sources(spec, list(pool.source.unique())), Arm()
    rows, left = [], {False: n // 3, True: n - n // 3}
    for r in pool.itertuples():
        if left[r.tried] == 0:
            continue
        e = Episode(by_name[r.source], r.episode)
        if e.depth_m(r.frame) is None:
            continue
        a = anchor(e, closings(arm, e), r.frame)
        frame = pd.read_parquet(passes[[p.name for p in passes].index(r.name)] / "frames.parquet")
        frame = frame[(frame.source == r.source) & (frame.episode == r.episode) & (frame.frame == r.frame)].iloc[0]
        rows.append(dict(
            frame, grasp_frame=a["frame"], close_frame=a["close"], opening=a["opening"], tried=a["tried"],
            try_frame=a["try_frame"], try_cm=r.try_cm, was_cm=r.tip_error_cm,
        ))
        left[r.tried] -= 1
    picked = pd.DataFrame(rows).sort_values(["source", "episode"]).reset_index(drop=True)
    stalls = {k: v for p in passes for k, v in json.loads((p / "stalls.json").read_text()).items()}
    for name in ["label", "key", "try"]:
        (work / name).mkdir(parents=True)
        (work / name / "pass.json").write_text(json.dumps(spec, indent=1))
        (work / name / "stalls.json").write_text(json.dumps(stalls, indent=1))
        rows = picked if name != "try" else picked[picked.tried]
        rows[["source", "episode", "frame", "kind", "grade", "subtask", "contact", "precision"]].to_parquet(work / name / "frames.parquet")
        for r in rows.itertuples():
            if name != "label":
                folder = work / name / r.source / f"ep{r.episode:03d}" / f"f{r.frame:06d}"
                folder.mkdir(parents=True)
                like = r.grasp_frame if name == "key" else r.try_frame
                (folder / "waypoints.json").write_text(json.dumps(dict(
                    contact=r.contact if r.contact not in ("", "unknown", "None") else "na", intent=r.subtask, wrong="nothing",
                    note=f"the measured pose at frame {like}", waypoints=[dict(like_frame=int(like), gripper=float(r.opening), arrive="approach")],
                )))
    picked.to_parquet(work / "key" / "picked.parquet")
    print(picked.drop(columns=["precision", "contact"]).to_string())


def arms(work: Path) -> list[str]:
    return sorted(p.parent.name for p in work.glob("*/corrections.parquet") if p.parent.name != "key")


def new_arm(work: Path, name: str) -> None:
    (work / name).mkdir()
    for file in ("pass.json", "stalls.json"):
        shutil.copy(work / "key" / file, work / name / file)
    pd.read_parquet(work / "key" / "frames.parquet").to_parquet(work / name / "frames.parquet")


def score(work: Path) -> None:
    arm, rows = Arm(), []
    for r in pd.read_parquet(work / "key" / "picked.parquet").itertuples():
        case = Path(r.source) / f"ep{r.episode:03d}" / f"f{r.frame:06d}"
        truth = first_pose(arm, work / "key" / case)
        for name in arms(work):
            row = dict(arm=name, source=r.source, episode=r.episode, frame=r.frame, subtask=r.subtask, left_cm=r.left_cm)
            rows.append(row)
            if not (work / name / case / "correction.npy").exists():
                row["status"] = "not built"
                continue
            mine = first_pose(arm, work / name / case)
            jaw = truth["jaw"] if mine["jaw"] is None else mine["jaw"]
            d = mine["tip"] - truth["tip"]
            row.update(
                status="ok", contact_truth=truth["contact"], contact=mine["contact"],
                tip_error_cm=100 * float(np.linalg.norm(d)), xy_error_cm=100 * float(np.linalg.norm(d[:2])), z_error_cm=100 * float(d[2]),
                approach_error_deg=float(np.degrees(np.arccos(np.clip(mine["approach"] @ truth["approach"], -1, 1)))),
                jaw_error_deg=float(np.degrees(np.arccos(np.clip(abs(jaw @ truth["jaw"]) / np.linalg.norm(jaw), 0, 1)))),
                note=mine["note"],
            )
    table = pd.DataFrame(rows)
    table.to_csv(work / "score.csv", index=False)
    ok = table[table.status == "ok"]
    columns = ["left_cm", "tip_error_cm", "xy_error_cm", "z_error_cm", "approach_error_deg", "jaw_error_deg"]
    print(table.groupby("arm").status.value_counts().to_string())
    for name in arms(work):
        print(f"\n{name}\n{ok[ok.arm == name][columns].describe().round(1).to_string()}")
    wide = ok.pivot(index=["source", "episode", "frame", "subtask", "left_cm"], columns="arm", values="tip_error_cm").reset_index()
    print("\ntip error by frame, cm\n" + wide.round(1).to_string())


def page(work: Path) -> None:
    cases = [
        dict(
            label=f"{r.source} ep{r.episode} · frame {r.frame} · grasp at +{(r.grasp_frame - r.frame) / 30:.1f} s", episode=r.episode, frame=r.frame,
            npz=work / "key" / r.source / "action_trace_records.npz",
            folders=[work / name / r.source / f"ep{r.episode:03d}" / f"f{r.frame:06d}" for name in [*arms(work), "key"]],
        )
        for r in pd.read_parquet(work / "key" / "picked.parquet").itertuples()
    ]
    cases = [c for c in cases if all((f / "correction.npy").exists() for f in c["folders"])]
    write_page(
        work, cases, [*(f"labeller, {name}" for name in arms(work)), "to the real grasp pose"], "Grasp test",
        "One frame per step, shortly before a grasp that worked: the recorded 1 s chunk, the labeller's chunk in each arm and the chunk "
        "to the measured grasp pose, all at the builder's pace.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pick", "arm", "score", "page", "audit", "anchored"])
    parser.add_argument("work", type=Path)
    parser.add_argument("source_pass", type=Path, nargs="*")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--per", type=int, default=3)
    parser.add_argument("--lead", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "pick":
        pick(args.work, args.source_pass[0], args.n, args.per, args.lead, args.seed)
    elif args.command == "arm":
        new_arm(args.work, str(args.source_pass[0]))
    elif args.command == "audit":
        audit(args.work, args.source_pass)
    elif args.command == "anchored":
        anchored(args.work, args.source_pass, args.n, args.seed)
    elif args.command == "score":
        score(args.work)
    else:
        page(args.work)


if __name__ == "__main__":
    main()
