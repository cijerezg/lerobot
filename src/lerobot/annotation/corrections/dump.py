"""Write what the labeller reads for every frame of a correction pass.

  uv run python -m lerobot.annotation.corrections.dump <work> [--source NAME ...] [--episode N ...] [--depth]

Reads ``<work>/frames.parquet`` and ``stalls.json`` (``sample``). Writes, per episode,
  <work>/<source>/ep<NNN>/episode.txt        task, stall angles, the episode's frames in order
  <work>/<source>/ep<NNN>/f<NNNNNN>/         case.txt, wrist_a.png, wrist_b.png, wrist_c.png (the wrist clip, ``TIMES``,
                                             6 tiles each), top.png (``TOP_TIMES``; on the t = 0 tile of an episode that has a
                                             fitted top camera: red circle = tool tip, magenta = tip path of the
                                             recorded chunk)
With ``--depth`` (recordings with wrist depth) the frame folder also gets ``wrist_depth.png`` (the t = 0 wrist image
at 2x with the depth reading in cm written on a 16 x 12 grid of pixels), ``depth_map.png`` (the depth image in
colour) and ``height_map.png`` (the same coloured by world height), and ``case.txt`` lists the 16 x 12 readings
instead of the 5 x 4 ones, the same grid as world positions (the wrist camera's pose is the twin's,
``Gripper.wrist``) and the table height seen. ``look`` writes the same for any
other frame of the episode.
A pass with ``"causal": true`` in its ``pass.json`` (``grasp_test``) shows nothing after t = 0: the wrist clip is
``CAUSAL_TIMES``, the top view ``CAUSAL_TOP_TIMES``, no recorded tip path; ``look`` refuses later frames.
and for the pass ``<work>/jobs.txt``: one labelling job per line, an episode folder and its frame folders, at most
JOB frames each (a long episode is several jobs, to be run in order).
The labeller adds ``waypoints.json`` to the frame folder. A frame folder that has its ``case.txt`` (written last)
is skipped, so an interrupted dump resumes.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot.annotation.corrections.anchor import anchor, closings
from lerobot.annotation.corrections.arm import FPS, Arm, project
from lerobot.annotation.corrections.source import Episode, sources

CHUNK = 30
TIMES = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0]  # seconds from the chunk start
TOP_TIMES = [-5.0, 0.0, 1.0, 2.0, 4.0, 8.0]
CAUSAL_TIMES = [-8.0, -6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.6, -0.3, 0.0]  # a causal pass: nothing after t = 0
CAUSAL_TOP_TIMES = [-5.0, -2.0, -1.0, 0.0]
SIDE = 378
JOB = 20
DEPTH_U, DEPTH_V = [64, 192, 320, 448, 576], [60, 180, 300, 420]  # wrist-image pixels of the depth readings
DENSE_U, DENSE_V = list(range(20, 640, 40)), list(range(20, 480, 40))  # the same with --depth
INTRINSICS = (394.9832, 394.9832, 322.5604, 238.6966)  # the D405's fx, fy, cx, cy at 640 x 480
MAP_CM = (7.0, 40.0)  # range of the depth colours
LEVEL_CM = (2.5, 3.0)  # a table that reads more tilted (cm per 10 cm) or less steady in height between frames: world positions are off


def kind_text(kind: str, grade: int) -> str:
    return f"demo, grade {grade}" if kind == "demo" else kind


def tile(rgb: np.ndarray, text: str) -> np.ndarray:
    small = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    cv2.putText(small, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(small, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return small


def depth_cm(depth: np.ndarray, u: int, v: int) -> float | None:
    """The median valid reading (under 1 m) of the 13 x 13 patch at a wrist-image pixel, cm; None where it has none."""
    patch = depth[max(v - 6, 0) : v + 7, max(u - 6, 0) : u + 7]
    valid = patch[(patch > 0) & (patch < 1.0)]
    return 100 * float(np.median(valid)) if len(valid) else None


def depth_grid(depth: np.ndarray, us: list[int] = DEPTH_U, vs: list[int] = DEPTH_V, t: str = "t = 0") -> list[str]:
    """The wrist depth image as a table of readings in cm (``depth_cm`` at each pixel), "-" where there is none."""
    lines = [
        f"wrist depth at {t}, cm from the camera along its optical axis, at wrist-image pixels (640 x 480; inside a "
        "wrist tile u x 0.59, v x 0.79); - = no reading",
        "v \\ u " + " ".join(f"{u:5d}" for u in us),
    ]
    for v in vs:
        readings = [depth_cm(depth, u, v) for u in us]
        lines.append(f"{v:5d} " + " ".join("    -" if r is None else f"{r:5.1f}" for r in readings))
    return lines


def camera_xyz_cm(u: float, v: float, z_cm: float) -> tuple[float, float, float]:
    """A wrist-image pixel and its depth reading as a point in the camera frame: x right, y down, z forward."""
    fx, fy, cx, cy = INTRINSICS
    return (u - cx) * z_cm / fx, (v - cy) * z_cm / fy, z_cm


def world_point(camera: tuple, u: float, v: float, z_cm: float) -> np.ndarray:
    """A wrist-image pixel and its depth reading as a point in the world frame, metres (``camera`` = ``Gripper.wrist``)."""
    return camera[0] + camera[1] @ np.asarray(camera_xyz_cm(u, v, z_cm)) / 100


def pixel_of(camera: tuple, point: np.ndarray) -> tuple[float, float, float]:
    """A world point in the wrist image: pixel (u, v) and its distance along the optical axis, cm (<= 0: behind)."""
    fx, fy, cx, cy = INTRINSICS
    x, y, z = camera[1].T @ (np.asarray(point) - camera[0])
    return (cx + fx * x / z, cy + fy * y / z, 100 * z) if z > 1e-6 else (-1.0, -1.0, 100 * z)


def world_grid(depth: np.ndarray, camera: tuple, t: str = "t = 0") -> list[str]:
    """The surface seen at each ``DENSE_U`` x ``DENSE_V`` pixel as a world position, "x,y,z" in cm."""
    lines = [f"world position of the surface at each wrist-image pixel at {t}: x,y,z in cm (robot frame, the tips' frame); - = no reading",
             "v \\ u " + " ".join(f"{u:>10d}" for u in DENSE_U)]
    for v in DENSE_V:
        cells = []
        for u in DENSE_U:
            r = depth_cm(depth, u, v)
            cells.append("-" if r is None else "{:.0f},{:.0f},{:.1f}".format(*(100 * world_point(camera, u, v, r))))
        lines.append(f"{v:5d} " + " ".join(f"{c:>10s}" for c in cells))
    return lines


def table_cm(depth: np.ndarray, camera: tuple) -> float | None:
    """Height of the largest level surface in the wrist depth image (normally the table), world z in cm."""
    fx, fy, cx, cy = INTRINSICS
    v, u = np.mgrid[0:480:6, 0:640:6]
    z = depth[v, u]
    ok = (z > 0.08) & (z < 0.6)
    if ok.sum() < 500:
        return None
    height = 100 * (np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)[ok] @ camera[1][2] + camera[0][2])
    count, edges = np.histogram(height, bins=np.arange(-10.0, 30.0, 0.4))
    return float(edges[count.argmax()] + 0.2)


def tilt_cm(depth: np.ndarray, camera: tuple) -> float | None:
    """How far from level the table of ``table_cm`` reads: the slope of the plane through it, cm of height per 10 cm."""
    fx, fy, cx, cy = INTRINSICS
    v, u = np.mgrid[0:480:6, 0:640:6]
    z = depth[v, u]
    ok = (z > 0.08) & (z < 0.6)
    if ok.sum() < 500:
        return None
    p = 100 * (np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)[ok] @ camera[1].T + camera[0])
    off, width = p[:, 2] - table_cm(depth, camera), 2.5
    for _ in range(3):
        near = np.abs(off) < width
        if near.sum() < 200:
            return None
        plane = np.linalg.lstsq(np.column_stack([p[near, :2], np.ones(near.sum())]), p[near, 2], rcond=None)[0]
        off, width = p[:, 2] - np.column_stack([p[:, :2], np.ones(len(p))]) @ plane, 1.0
    return 10 * float(np.hypot(plane[0], plane[1]))


def world_warning(arm: Arm, e: Episode, t: int) -> str | None:
    """A line for the labeller when the world positions of this recording cannot be trusted around frame ``t``: the
    table reads tilted, or at heights that differ between t, t - 1.5 s and t - 3 s (seen on rebot_main_additions,
    where the depth image does not line up with the colour image)."""
    seen = []
    for f in (t, max(t - 45, 0), max(t - 90, 0)):
        depth = e.depth_m(f)
        camera = arm.gripper.wrist(e.states[f])
        if depth is not None and table_cm(depth, camera) is not None and tilt_cm(depth, camera) is not None:
            seen.append((table_cm(depth, camera), tilt_cm(depth, camera)))
    if seen and (max(s[1] for s in seen) > LEVEL_CM[0] or np.ptp([s[0] for s in seen]) > LEVEL_CM[1]):
        return ("WARNING: the world positions of this recording are off around this frame (the table reads at "
                + ", ".join(f"{z:.1f} cm tilted {tilt:.1f} cm per 10 cm" for z, tilt in seen)
                + " at t = 0, -1.5 and -3 s). Do not place the pose from them or from preview: use rule 7, the measured rows and the images.")
    return None


def height_image(depth: np.ndarray, camera: tuple, text: str) -> np.ndarray:
    """The wrist depth image coloured by world height: blue -2 cm to red 16 cm and above, black = no reading."""
    fx, fy, cx, cy = INTRINSICS
    v, u = np.mgrid[0:480, 0:640]
    height = 100 * (np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], axis=-1) @ camera[1][2] + camera[0][2])
    colour = cv2.applyColorMap(np.uint8(np.clip((height + 2) / 18, 0, 1) * 255), cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    colour[(depth <= 0) | (depth >= 1.0)] = 0
    label = f"{text}  height above the robot base plane: blue -2 cm -> red 16 cm and above, black = no reading"
    cv2.putText(colour, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
    cv2.putText(colour, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return colour


def depth_images(rgb: np.ndarray, depth: np.ndarray, text: str) -> tuple[np.ndarray, np.ndarray]:
    """(the wrist image at 2x with the ``DENSE_U`` x ``DENSE_V`` readings written on it in cm and pixel rulers on
    the top and left edges, the depth image in colour: ``MAP_CM`` blue-near to red-far, black = no reading)."""
    big = cv2.resize(rgb, (1280, 960), interpolation=cv2.INTER_CUBIC)

    def write(image, label, at, scale=0.42):
        cv2.putText(image, label, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(image, label, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1)

    for v in DENSE_V:
        for u in DENSE_U:
            r = depth_cm(depth, u, v)
            cv2.circle(big, (2 * u, 2 * v), 2, (255, 0, 255), -1)
            write(big, "-" if r is None else f"{r:.1f}", (2 * u - 16, 2 * v + 16))
    for u in range(80, 640, 80):
        write(big, f"u{u}", (2 * u - 14, 12), 0.4)
    for v in range(80, 480, 80):
        write(big, f"v{v}", (2, 2 * v + 4), 0.4)
    write(big, text, (40, 30), 0.5)
    cm = 100 * depth
    level = np.uint8(np.clip((cm - MAP_CM[0]) / (MAP_CM[1] - MAP_CM[0]), 0, 1) * 255)
    colour = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    colour[(depth <= 0) | (depth >= 1.0)] = 0
    write(colour, f"{text}  depth: blue {MAP_CM[0]:.0f} cm -> red {MAP_CM[1]:.0f} cm and beyond, black = no reading", (6, 16), 0.4)
    return big, colour


def dump_frame(arm: Arm, e: Episode, row, stalls: str, folder: Path, dense: bool = False, causal: bool = False, worked: list | None = None) -> None:
    t = row.frame
    times, top_times = (CAUSAL_TIMES, CAUSAL_TOP_TIMES) if causal else (TIMES, TOP_TIMES)
    frames = [int(np.clip(t + round(s * FPS), 0, len(e.states) - 1)) for s in times]
    folder.mkdir(parents=True, exist_ok=True)

    wrist = [tile(img, f"t={(f - t) / FPS:+.1f}s") for f, img in zip(frames, e.images("wrist", frames), strict=True)]
    cv2.imwrite(str(folder / "wrist_a.png"), np.vstack([np.hstack(wrist[0:3]), np.hstack(wrist[3:6])])[:, :, ::-1])
    cv2.imwrite(str(folder / "wrist_b.png"), np.vstack([np.hstack(wrist[6:9]), np.hstack(wrist[9:12])])[:, :, ::-1])
    if len(wrist) > 12:
        cv2.imwrite(str(folder / "wrist_c.png"), np.vstack([np.hstack(wrist[12:15]), np.hstack(wrist[15:18])])[:, :, ::-1])
    top_frames = [int(np.clip(t + round(s * FPS), 0, len(e.states) - 1)) for s in top_times]
    tops = e.images("top", top_frames)
    if e.top is not None:
        now = tops[top_times.index(0.0)] = tops[top_times.index(0.0)].copy()
        if not causal:
            recorded = [project(e.top, arm.gripper.fk(a)[0]) for a in e.actions[t : t + CHUNK : 3]]
            cv2.polylines(now, [np.int32(np.round(recorded))], False, (255, 0, 255), 2)
        cv2.circle(now, tuple(np.int32(np.round(project(e.top, arm.gripper.fk(e.states[t])[0])))), 8, (255, 0, 0), 2)
    tops = [tile(img, f"t={(f - t) / FPS:+.1f}s") for f, img in zip(top_frames, tops, strict=True)]
    cv2.imwrite(str(folder / "top.png"), np.hstack(tops)[:, :, ::-1])

    tip, approach, jaw = arm.gripper.fk(e.states[t])
    lines = [
        f"case {e.source.name}/ep{e.index:03d}/f{t:06d} | task: {e.task} | subtask: {row.subtask} | contact: {row.contact}"
        f" | precision: {row.precision} | frame kind: {kind_text(row.kind, row.grade)}",
        f"chunk start = frame {t} of the episode, {FPS} fps; "
        + ("causal pass: nothing after t = 0 is shown" if causal else f"the recorded chunk is the {CHUNK} actions from t = 0 to 1.0 s"),
        f"tool at t = 0: tip {tip.round(3).tolist()} approach {approach.round(2).tolist()} jaw {jaw.round(2).tolist()} gripper {e.states[t][6]:.0f}",
        stalls,
        "",
        "time_s frame | measured: tip xyz m, approach, gripper deg | recorded action: tip xyz m, gripper deg"
        " (| operator = a human was in control at that frame)",
    ]
    for f in frames:
        tip, approach, _ = arm.gripper.fk(e.states[f])
        lines.append(
            f"{(f - t) / FPS:+.1f} {f} | {tip.round(3).tolist()} {approach.round(2).tolist()} {e.states[f][6]:.0f}"
            f" | {arm.gripper.fk(e.actions[f])[0].round(3).tolist()} {e.actions[f][6]:.0f}" + (" | operator" if e.kinds[f] == "operator" else "")
        )
    if not causal:
        end = t + int(np.argmax(np.append(e.subtasks[t:] != e.subtasks[t], True))) - 1
        tip, approach, _ = arm.gripper.fk(e.states[end])
        lines += ["", f"this stretch of the subtask ends at frame {end} ({(end - t) / FPS:+.1f} s): measured tip {tip.round(3).tolist()} approach "
                  f"{approach.round(2).tolist()} gripper {e.states[end][6]:.0f}; then: {e.subtasks[end + 1] if end + 1 < len(e.states) else 'the episode ends'}"]
        a = anchor(e, worked or [], t)
        if a is not None:
            tip, approach, jaw = arm.gripper.fk(e.states[a["frame"]])
            lines += [
                f"ANCHOR: the episode goes on to grasp this object and the grasp works: the gripper starts closing at frame {a['close']} "
                f"({(a['close'] - t) / FPS:+.1f} s) and stops at frame {a['frame']}; measured tool there: tip {tip.round(3).tolist()} approach "
                f"{approach.round(2).tolist()} jaw {jaw.round(2).tolist()}, gripper {a['opening']:.0f} before closing, {a['stall']:.0f} on the object.",
                f'  as the contact pose: {{"like_frame": {a["frame"]}, "gripper": {a["opening"]:.0f}, "arrive": "approach"}}, then close to {min(a["stall"] + 10, 0):.0f}'
                + (f" | CHECK FIRST: a closing attempt between this frame and that grasp failed (it stopped at frame {a['try_frame']}), so the "
                   "object may lie elsewhere at this frame than where it was grasped (rubric, Anchors)" if a["tried"] else "")
                + (f" | that grasp is in a stretch graded {a['grade']}: the Objects table outranks it" if a["grade"] in (1, 2, 3) else ""),
            ]
    depth = e.depth_m(t)
    if depth is not None and dense:
        fx, _, cx, cy = INTRINSICS
        lines += [
            "", *depth_grid(depth, DENSE_U, DENSE_V),
            f"a pixel (u, v) that reads z cm is at x = (u - {cx:.0f}) z / {fx:.0f} cm to the right, y = (v - {cy:.0f}) z / {fx:.0f} cm down"
            " and z cm forward of the wrist camera (the camera frame; the camera is fixed to the hand)",
        ]
        camera = arm.gripper.wrist(e.states[t])
        table = table_cm(depth, camera)
        lines += ["", *world_grid(depth, camera), "largest level surface in view (normally the table): " + ("none" if table is None else f"z = {table:.1f} cm")]
        lines += [w for w in [world_warning(arm, e, t)] if w is not None]
        big, colour = depth_images(e.images("wrist", [t])[0], depth, "t=+0.0s")
        cv2.imwrite(str(folder / "wrist_depth.png"), big[:, :, ::-1])
        cv2.imwrite(str(folder / "depth_map.png"), colour[:, :, ::-1])
        cv2.imwrite(str(folder / "height_map.png"), height_image(depth, camera, "t=+0.0s")[:, :, ::-1])
    elif depth is not None:
        lines += ["", *depth_grid(depth)]
    (folder / "case.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--episode", type=int, nargs="+")
    parser.add_argument("--depth", action="store_true")
    args = parser.parse_args()
    frames = pd.read_parquet(args.work / "frames.parquet")
    jobs = []
    for (name, index), rows in frames.groupby(["source", "episode"], sort=False):
        for part in np.array_split(rows.frame.to_numpy(), -(-len(rows) // JOB)):
            jobs.append(f"{name}/ep{index:03d} f{part[0]:06d}..f{part[-1]:06d} {len(part)} frames")
    (args.work / "jobs.txt").write_text("\n".join(jobs) + "\n")
    if args.source is not None:
        frames = frames[frames.source.isin(args.source)]
    if args.episode is not None:
        frames = frames[frames.episode.isin(args.episode)]
    stalls = json.loads((args.work / "stalls.json").read_text())
    arm, spec = Arm(), json.loads((args.work / "pass.json").read_text())
    for name, source in sources(spec, list(frames.source.unique())).items():
        for index, rows in frames[frames.source == name].groupby("episode"):
            e = Episode(source, index)
            worked = closings(arm, e)
            folder = args.work / name / f"ep{index:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            stall_line = "gripper stalls on a held object in the pass's recordings (measured deg): " + (
                "; ".join(f"{s}: {stalls[s]}" for s in dict.fromkeys(e.subtasks) if s in stalls) or "none recorded"
            )
            lines = [
                f"task: {e.task}",
                f"{name} episode {index}: {len(e.states)} frames at {FPS} fps, {len(rows)} frames to label",
                stall_line,
                "",
                "folder | time_s | subtask | contact | frame kind",
                *(f"f{r.frame:06d} | {r.frame / FPS:.1f} | {r.subtask} | {r.contact} | {kind_text(r.kind, r.grade)}" for r in rows.itertuples()),
            ]
            (folder / "episode.txt").write_text("\n".join(lines) + "\n")
            for row in rows.itertuples():
                if not (folder / f"f{row.frame:06d}" / "case.txt").exists():
                    dump_frame(arm, e, row, stall_line, folder / f"f{row.frame:06d}", args.depth, spec.get("causal", False), worked)
            print("wrote", folder, len(rows), "frames")


if __name__ == "__main__":
    main()
