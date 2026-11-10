"""Fit the top camera of every episode of a correction pass.

  uv run python -m lerobot.annotation.corrections.fit_top <work> [--source NAME ...]

The top camera was moved between sessions, so each episode of ``<work>/frames.parquet`` gets its own camera pose:
the one that puts the arm, computed from the measured joints, on the arm's lime-green pixels. A fit takes N random
frames, samples points on the model's lime parts and minimises two distances in pixels: each projected point to the
nearest green pixel, and each green pixel to the nearest projected point. Both are clipped, because the scene has
other green things; the clip shrinks over three rounds, and each round keeps the model points that the camera of
the round before sees. Only position and rotation are fitted: the arm does not tell the focal length from the
camera's distance, so the focal length stays the start's.

The arm's green shows little of the camera's position (a shift and a turn move the image alike), so a fit is held
to its start, in two stages:
  source    frames from all its episodes, from the source's camera in ``pass.json["top"]``: free turn, position
            held (HOLD_SOURCE)
  episode   frames of the episode, from the source's fit: turn and position held (HOLD_EPISODE)

Writes ``pass.json["cameras"]["<source>/ep<NNN>"]`` (``px`` = the fit's mean clipped distance; 8 = nothing matches)
and one overlay tile per episode (green = the mask's outline, magenta = the fitted arm points, red circle = the
tool tip) in ``<work>/top_fit/<source>.png``.
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lerobot.annotation.corrections.arm import Arm
from lerobot.annotation.corrections.source import Episode, Source, sources

N, PER_GEOM, PER_MASK = 24, 250, 400
CLIPS_PX = (60.0, 20.0, 8.0)
GREEN_LO, GREEN_HI = (38, 50, 150), (55, 255, 255)  # OpenCV HSV of the lime links
# a fit is held to its start: being this far from it (m, deg) costs 1 px rms
HOLD_SOURCE, HOLD_EPISODE = (0.01, np.inf), (0.01, 2.0)


class Fitter:
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.arm = Arm()
        self.model, self.data = self.arm.gripper.m, self.arm.gripper.d
        self.lime = []  # (geom, points in the geom frame) sampled on the faces of the model's lime meshes
        m = self.model
        for g in range(m.ngeom):
            if m.geom_matid[g] == m.material("lime").id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]
                vertices = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                triangles = vertices[m.mesh_face[m.mesh_faceadr[k] : m.mesh_faceadr[k] + m.mesh_facenum[k]]]
                area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
                pick = self.rng.choice(len(triangles), PER_GEOM, p=area / area.sum())
                self.lime.append((g, np.einsum("ij,ijk->ik", self.rng.dirichlet(np.ones(3), PER_GEOM), triangles[pick])))

    def frames(self, picks: list[tuple[Episode, list[int]]]) -> list[dict]:
        """The frames (episode, times) that show enough green, each with its mask and the arm's state."""
        found = []
        for e, times in picks:
            for t, image in zip(times, e.images("top", times), strict=True):
                mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), GREEN_LO, GREEN_HI)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                green = np.argwhere(mask > 0)[:, ::-1].astype(float)
                if len(green) >= 200:
                    found.append(dict(
                        image=image, mask=mask, state=e.states[t], tip=self.arm.gripper.fk(e.states[t])[0],
                        far=cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(float),
                        green=green[self.rng.choice(len(green), min(PER_MASK, len(green)), replace=False)],
                    ))
        return found

    def look(self, cam: dict, frames: list[dict]) -> None:
        """Set each frame's ``points``: the lime points the camera sees (no visual geom of the model, group 2, in
        front of them)."""
        hit, visual = np.zeros(1, dtype=np.int32), np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
        for f in frames:
            self.data.qpos[self.arm.gripper.q0 : self.arm.gripper.q0 + 6] = np.deg2rad(f["state"][:6])
            mujoco.mj_forward(self.model, self.data)
            keep = []
            for g, local in self.lime:
                for p in local @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]:
                    far = np.linalg.norm(p - cam["pos"])
                    d = mujoco.mj_ray(self.model, self.data, cam["pos"], (p - cam["pos"]) / far, visual, 1, -1, hit)
                    if d < 0 or d > far - 0.003:
                        keep.append(p)
            f["points"] = np.array(keep)

    def fit(self, start: dict, frames: list[dict], hold: tuple[float, float]) -> tuple[dict, float]:
        """The camera near ``start`` that matches the frames, and its mean clipped distance in pixels."""
        scale = np.array([hold[0]] * 3 + [np.deg2rad(hold[1])] * 3)
        x = np.zeros(6)
        for clip in CLIPS_PX:
            self.look(camera(start, x), frames)
            n = len(distances(start, frames, clip))
            x = least_squares(
                lambda x, clip=clip, n=n: np.append(distances(camera(start, x), frames, clip), np.sqrt(n) * x / scale),
                x, x_scale=0.02, diff_step=1e-2,
            ).x
        fitted = camera(start, x)
        self.look(fitted, frames)
        return fitted, float(distances(fitted, frames, CLIPS_PX[-1]).mean())


def camera(start: dict, x: np.ndarray) -> dict:
    return dict(start, pos=start["pos"] + x[:3], mat=Rotation.from_rotvec(x[3:]).as_matrix() @ start["mat"])


def pixels(cam: dict, points: np.ndarray) -> np.ndarray:
    local = (points - cam["pos"]) @ cam["mat"]
    return np.column_stack([cam["cx"] + cam["f"] * local[:, 0] / -local[:, 2], cam["cy"] - cam["f"] * local[:, 1] / -local[:, 2]])


def distances(cam: dict, frames: list[dict], clip: float) -> np.ndarray:
    """Clipped pixels from each projected arm point to the nearest green pixel, and from each sampled green pixel to
    the nearest projected point, over all frames."""
    found = []
    for f in frames:
        uv = pixels(cam, f["points"])
        found += [map_coordinates(f["far"], [uv[:, 1], uv[:, 0]], order=1, mode="nearest"), cKDTree(uv).query(f["green"])[0]]
    return clip * np.tanh(np.concatenate(found) / clip)


def fit_source(fitter: Fitter, source: Source, indices: list[int]) -> tuple[dict, list[np.ndarray]]:
    """{episode: camera with ``px``} of the source's episodes, and their overlay tiles."""
    rng = fitter.rng
    episodes = {i: Episode(source, i) for i in indices}
    picks = rng.choice(indices, 2 * N)
    shared = fitter.frames([(episodes[i], sorted(rng.choice(len(episodes[i].states), int((picks == i).sum()), replace=False).tolist())) for i in indices])
    base, px = fitter.fit(source.top, shared, HOLD_SOURCE)
    print(f"{source.name}: {len(shared)} frames, px {px:.2f}, pos {base['pos'].round(3).tolist()}", flush=True)
    cameras, tiles = {}, []
    for i, e in episodes.items():
        frames = fitter.frames([(e, sorted(rng.choice(len(e.states), N, replace=False).tolist()))])
        fitted, px = fitter.fit(base, frames, HOLD_EPISODE)
        cameras[i] = dict(fitted, px=px)
        moved, turned = 1e3 * np.linalg.norm(fitted["pos"] - base["pos"]), np.rad2deg(Rotation.from_matrix(fitted["mat"] @ base["mat"].T).magnitude())
        print(f"  ep{i:03d}: {len(frames)} frames, px {px:.2f}, {moved:.0f} mm and {turned:.1f} deg from the source's", flush=True)
        f = frames[len(frames) // 2]
        tile = f["image"].copy()
        cv2.drawContours(tile, cv2.findContours(f["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, (0, 255, 0), 1)
        for u, v in pixels(fitted, f["points"][::2]):
            cv2.circle(tile, (round(u), round(v)), 1, (255, 0, 255), -1)
        cv2.circle(tile, tuple(np.int32(np.round(pixels(fitted, f["tip"][None])[0]))), 6, (255, 0, 0), 1)
        cv2.putText(tile, f"ep{i:03d} {px:.1f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(tile)
    return cameras, tiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--source", nargs="+")
    args = parser.parse_args()
    spec = json.loads((args.work / "pass.json").read_text())
    episodes = pd.read_parquet(args.work / "frames.parquet")[["source", "episode"]].drop_duplicates()
    if args.source is not None:
        episodes = episodes[episodes.source.isin(args.source)]
    fitter = Fitter()
    (args.work / "top_fit").mkdir(exist_ok=True)
    for name, source in sources(spec, list(episodes.source.unique())).items():
        cameras, tiles = fit_source(fitter, source, episodes[episodes.source == name].episode.tolist())
        for i, c in cameras.items():
            spec.setdefault("cameras", {})[f"{name}/ep{i:03d}"] = dict(c, pos=c["pos"].tolist(), mat=c["mat"].tolist())
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
        cv2.imwrite(str(args.work / "top_fit" / f"{name}.png"), np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])[:, :, ::-1])
        (args.work / "pass.json").write_text(json.dumps(spec, indent=1))
    print("wrote", args.work / "pass.json", "and", args.work / "top_fit")


if __name__ == "__main__":
    main()
