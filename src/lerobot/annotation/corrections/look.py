"""Show the labeller another frame of an episode: images, tool pose and wrist depth.

  uv run python -m lerobot.annotation.corrections.look <work> <source> <episode> <frame> [<frame> ...] [--at U,V ...]

For each frame (episode-local, any frame of the episode) writes ``<work>/<source>/ep<NNN>/look/f<NNNNNN>/``
  wrist.png, top.png    the two cameras, 640 x 480
  wrist_depth.png       the wrist image at 2x with the depth readings in cm on a 16 x 12 grid (``dump --depth``)
  depth_map.png         the depth image in colour
  height_map.png        the depth image coloured by world height
and prints the measured tool pose, the recorded action, the table height seen and, without ``--at``, the depth table
and the same grid as world positions. ``--at U,V`` (wrist-image pixels) prints, per pixel, the reading and the world
position of the surface there instead of the tables. Depth is stored for
every third frame; the nearest one within a frame is used. In a causal pass (``"causal": true`` in ``pass.json``)
frames after the episode's last frame to label are refused.
"""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from lerobot.annotation.corrections.arm import FPS, Arm
from lerobot.annotation.corrections.dump import DENSE_U, DENSE_V, depth_cm, depth_grid, depth_images, height_image, table_cm, world_grid, world_point, world_warning
from lerobot.annotation.corrections.source import Episode, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("source")
    parser.add_argument("episode", type=int)
    parser.add_argument("frames", type=int, nargs="+")
    parser.add_argument("--at", nargs="+", action="extend", default=[])
    args = parser.parse_args()
    spec = json.loads((args.work / "pass.json").read_text())
    if spec.get("causal"):  # a causal pass shows nothing after the episode's last frame to label
        frames = pd.read_parquet(args.work / "frames.parquet")
        last = int(frames[(frames.source == args.source) & (frames.episode == args.episode)].frame.max())
        if max(args.frames) > last:
            parser.error(f"this pass shows no frame after {last}")
    source = sources(spec, [args.source])[args.source]
    e, arm = Episode(source, args.episode), Arm()
    print(f"{args.source} episode {args.episode}: {len(e.states)} frames at {FPS} fps")
    for f, wrist, top in zip(args.frames, e.images("wrist", args.frames), e.images("top", args.frames), strict=True):
        folder = args.work / args.source / f"ep{args.episode:03d}" / "look" / f"f{f:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "wrist.png"), wrist[:, :, ::-1])
        cv2.imwrite(str(folder / "top.png"), top[:, :, ::-1])
        tip, approach, jaw = arm.gripper.fk(e.states[f])
        print(f"\nframe {f} ({f / FPS:.1f} s) | subtask: {e.subtasks[f]} | {e.kinds[f]} | {folder}")
        print(f"measured: tip {tip.round(3).tolist()} approach {approach.round(2).tolist()} jaw {jaw.round(2).tolist()} gripper {e.states[f][6]:.0f}")
        print(f"recorded action: tip {arm.gripper.fk(e.actions[f])[0].round(3).tolist()} gripper {e.actions[f][6]:.0f}")
        depth = e.depth_m(f)
        if depth is None:
            print("no wrist depth at this frame")
            continue
        big, colour = depth_images(wrist, depth, f"frame {f}")
        cv2.imwrite(str(folder / "wrist_depth.png"), big[:, :, ::-1])
        cv2.imwrite(str(folder / "depth_map.png"), colour[:, :, ::-1])
        camera = arm.gripper.wrist(e.states[f])
        cv2.imwrite(str(folder / "height_map.png"), height_image(depth, camera, f"frame {f}")[:, :, ::-1])
        table = table_cm(depth, camera)
        if not args.at:
            print("\n".join(depth_grid(depth, DENSE_U, DENSE_V, f"frame {f}") + world_grid(depth, camera, f"frame {f}")))
        print("largest level surface in view (normally the table): " + ("none" if table is None else f"z = {table:.1f} cm"))
        if world_warning(arm, e, f) is not None:
            print(world_warning(arm, e, f))
        for at in args.at:
            u, v = (int(x) for x in at.split(","))
            z = depth_cm(depth, u, v)
            where = "no reading" if z is None else "{:.1f} cm from the camera, world x {:.3f} y {:.3f} z {:.3f} m".format(z, *world_point(camera, u, v, z))
            print(f"pixel ({u}, {v}): {where}")


if __name__ == "__main__":
    main()
