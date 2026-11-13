"""Show the labeller where its ``waypoints.json`` puts the fingers, before it hands the frame in.

  uv run python -m lerobot.annotation.corrections.preview <work> <source> <episode> <frame> [--on F]

Reads ``<work>/<source>/ep<NNN>/f<NNNNNN>/waypoints.json``, expands it as ``build`` does and writes into that folder
  preview.png        (``preview_f<F>.png`` with ``--on F``) the wrist image of frame F (default: the frame itself) at 2x with the contact pose (the first
                     waypoint) drawn in: the two finger pads (yellow circles, joined by the jaw line), the tip
                     (magenta cross), the fingers' axis back from the tip (magenta line) and the later waypoints'
                     tips (numbered); in grey the same marks for the tool where it is now
  preview_top.png    the top image with the waypoints' tips, when the episode has a fitted top camera
and prints the chunk check, and for the tip and the two pads of the contact pose: the pixel, the surface the depth
sees there (world position) and how far the mark is above (+) or below (-) that surface, next to the table height.
A mark hidden behind a surface reads negative: for a pinch from above, pads at or just above the table beside the
object and the tip below the object's top. ``--on F`` draws on another frame of the episode, one where the camera
sees the place better (a causal pass refuses a later frame).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot.annotation.corrections.arm import FPS, MAX_SPEED_DEG_S, Arm, check, project
from lerobot.annotation.corrections.build import CHUNK, smooth
from lerobot.annotation.corrections.dump import depth_cm, pixel_of, table_cm, world_point, world_warning
from lerobot.annotation.corrections.source import Episode, sources

GAP_M, GAP_M_PER_DEG = 0.002, 0.000223  # gap between the finger tips as the wrist images show it: 2 mm shut, 4.9 cm at -210 deg


def pads(tip: np.ndarray, approach: np.ndarray, jaw: np.ndarray, grip: float) -> tuple[np.ndarray, np.ndarray]:
    jaw = jaw - (jaw @ approach) * approach / (approach @ approach)
    half = 0.5 * (GAP_M + GAP_M_PER_DEG * max(-grip, 0.0)) * jaw / np.linalg.norm(jaw)
    return tip - half, tip + half


def draw_tool(image: np.ndarray, camera: tuple, tip, approach, jaw, grip, colour, cross) -> None:
    at = lambda p: tuple(int(round(2 * c)) for c in pixel_of(camera, p)[:2])  # noqa: E731
    if pixel_of(camera, tip)[2] <= 1:
        return
    a, b = pads(tip, approach, jaw, grip)
    cv2.line(image, at(a), at(b), colour, 2)
    for p in (a, b):
        cv2.circle(image, at(p), 9, colour, 2)
    cv2.line(image, at(tip), at(tip - 0.05 * approach / np.linalg.norm(approach)), cross, 2)
    cv2.drawMarker(image, at(tip), cross, cv2.MARKER_CROSS, 22, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("source")
    parser.add_argument("episode", type=int)
    parser.add_argument("frame", type=int)
    parser.add_argument("--on", type=int)
    args = parser.parse_args()
    spec = json.loads((args.work / "pass.json").read_text())
    on = args.frame if args.on is None else args.on
    if spec.get("causal"):
        frames = pd.read_parquet(args.work / "frames.parquet")
        last = int(frames[(frames.source == args.source) & (frames.episode == args.episode)].frame.max())
        if on > last:
            parser.error(f"this pass shows no frame after {last}")
    e, arm = Episode(sources(spec, [args.source])[args.source], args.episode), Arm()
    folder = args.work / args.source / f"ep{args.episode:03d}" / f"f{args.frame:06d}"
    full, poses = smooth(arm, e, args.frame, json.loads((folder / "waypoints.json").read_text()), MAX_SPEED_DEG_S)
    result = check(arm, e.states[args.frame], np.vstack([full[:CHUNK], np.repeat(full[-1:], max(0, CHUNK - len(full)), axis=0)]))
    print(f"chunk check: {'ok' if result['ok'] else 'NOT OK ' + json.dumps(result)}; the manoeuvre takes {len(full) / FPS:.1f} s at the builder's pace")

    camera, depth = arm.gripper.wrist(e.states[on]), e.depth_m(on)
    image = cv2.resize(e.images("wrist", [on])[0], (1280, 960), interpolation=cv2.INTER_CUBIC)
    now = arm.gripper.fk(e.states[args.frame])
    draw_tool(image, camera, *now, e.states[args.frame][6], (150, 150, 150), (150, 150, 150))
    tip, approach, jaw, grip, _ = poses[0]
    draw_tool(image, camera, tip, approach, jaw, grip, (255, 255, 0), (255, 0, 255))
    for i, pose in enumerate(poses[1:], start=2):
        u, v, z = pixel_of(camera, pose[0])
        if z > 1:
            cv2.circle(image, (int(2 * u), int(2 * v)), 5, (0, 255, 255), -1)
            cv2.putText(image, str(i), (int(2 * u) + 8, int(2 * v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"frame {on}: yellow = finger pads at the contact pose, magenta = tip and finger axis, grey = the tool now", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    file = folder / ("preview.png" if args.on is None else f"preview_f{on:06d}.png")
    cv2.imwrite(str(file), image[:, :, ::-1])
    print("wrote", file)

    table = None if depth is None else table_cm(depth, camera)
    print(f"frame {on}: table (largest level surface in view): " + ("not seen" if table is None else f"z = {table / 100:.3f}"))
    if world_warning(arm, e, on) is not None:
        print(world_warning(arm, e, on))
    a, b = pads(tip, approach, jaw, grip)
    print(f"the pads are drawn where the finger TIPS are at gripper {grip:.0f}: {100 * np.linalg.norm(b - a):.1f} cm apart (the tips open less than the "
          "0.42 mm per degree of the middle of the fingers, where a wide object is held)")
    for name, p in (("tip", tip), ("pad 1", a), ("pad 2", b)):
        u, v, z = pixel_of(camera, p)
        line = f"{name}: world {np.round(p, 3).tolist()}"
        if z <= 1 or not (0 <= u < 640 and 0 <= v < 480):
            print(line + " | outside the wrist image of this frame: draw it on an earlier frame with --on, one where the camera is further back")
            continue
        line += f" | pixel ({u:.0f}, {v:.0f}), {z:.1f} cm from the camera"
        seen = None if depth is None else depth_cm(depth, int(round(u)), int(round(v)))
        if seen is None or seen < 8.0:
            print(line + " | no usable depth reading there (none, or nearer than the sensor reads)")
            continue
        surface = world_point(camera, u, v, seen)
        print(line + f" | surface seen there: world {np.round(surface, 3).tolist()} | the mark is {seen - z:+.1f} cm from that surface along the view "
              f"(+ = in front of it, - = behind it), {100 * (p[2] - surface[2]):+.1f} cm in height")
    if e.top is not None:
        top = e.images("top", [on])[0].copy()
        for i, pose in enumerate(poses, start=1):
            u, v = project(e.top, pose[0])
            cv2.circle(top, (int(u), int(v)), 6, (255, 0, 255), 2)
            cv2.putText(top, str(i), (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imwrite(str(folder / "preview_top.png"), top[:, :, ::-1])
        print("wrote", folder / "preview_top.png")


if __name__ == "__main__":
    main()
