"""Play the Action Inspector's chunks on the MuJoCo reBot and write one video per anchor.

Input is ``action_trace_records.npz``, written beside ``action_trace.html`` by
``action_trace_probe``: per anchor, the measured state, the demonstrated chunk, every
filtered flow sample, and the follower states the demonstration actually produced.

    uv run python -m lerobot.probes.action_trace_sim --records <run>/validation/step_00001200/action_trace/action_trace_records.npz

writes ``anchor000_ep0003_fr001234.mp4`` per anchor plus ``summary.json`` into ``--out``
(default ``action_trace_sim/`` beside the records). Every video frame is one tick of the
chunk: a panel per arm, the demonstrated chunk in grey first and then each flow sample in
its Action Inspector colour, each with its own end-effector path and the demo's path for
reference (and the FAST decode's path on an ``action_mode=both`` checkpoint); the
dataset's camera frames over the same window when the dataset root recorded in the npz
exists on this machine, the anchor's still frames otherwise; and the joint curves, demo
against every sample, with a cursor at the current tick.

By default everything drawn is a commanded action: no physics, no servo lag, what the
policy asked for against what the demonstrator asked for. ``--execute`` instead drives
the MuJoCo position actuators through each chunk from the measured demo state, one
target per tick, and poses the sample arms from what the plant did. The demo panel then
poses the arm from the real follower's measured states, every panel draws its command
target ahead of the arm and its achieved trail behind it, the joint curves add the
simulated joints next to the commands and the real follower next to the demo command,
and the caption counts the ticks with table contact. The scene is the Menagerie model
retargeted to our DM geometry (``mjcf/retarget_to_dm.py``): end effector exact to a
micron, meshes and inertials still RobStride-shaped, actuator gains inherited from the
lighter RS arm and not identified against the DM motors. Read contact and the shape of
the lag from it, not millimetres of lag.

Chunks are one second long; ``--time-scale 2`` writes the video at half the chunk's fps
so it plays at half speed. Rendering is offscreen through EGL (``MUJOCO_GL`` defaults
to ``egl`` here).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from lerobot.probes.action_trace_probe import FAST_COLOR, SAMPLE_COLORS  # noqa: E402
from lerobot.rl.inference_video import VideoWriter  # noqa: E402
from lerobot.robots.rebot_b601_follower.kinematics import ARM_DOF  # noqa: E402

ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "rebot_b601_follower"
SCENE = ROBOT_DIR / "mjcf" / "scene_dm.xml"
MODEL = ROBOT_DIR / "mjcf" / "rebot_b601_dm.xml"
EE_BODY = "gripper_end"
FLOOR_GEOM = "floor"
ARM_BODY_RE = r'(<body name="(?:link[1-6]|gripper_(?:end|left|right))")'
GRIPPER_IDX = 6

# Dataset gripper units are degrees in [-238.8, 0] with 0 = closed; the MJCF drives two
# coupled slide joints in [0, 0.05] m, 0 = closed (same mapping as mouse_teleop_sim.py).
GRIPPER_DEG_OPEN = -238.8
GRIPPER_SLIDE_OPEN = 0.05

# The MJCF ranges are the RobStride arm's. The DM arm is driven past them in the data
# (elbow to -196 deg against the model's -180), so joint and actuator ranges are widened
# to the soft limits the runtimes and the probe use, where those are wider.
SOFT_LIMITS_DEG = {
    "joint1": (-145.0, 145.0),
    "joint2": (-170.0, 1.0),
    "joint3": (-200.0, 1.0),
    "joint4": (-80.0, 90.0),
    "joint5": (-90.0, 90.0),
    "joint6": (-90.0, 90.0),
}

DEMO_COLOR = "#6b7280"
MEASURED_COLOR = "#111827"

# Frame layout, pixels. One square arm panel per arm in a row, the camera frames stacked
# at the right end of that row, the joint strip across the full width underneath.
PANEL = 360
CAM_W, CAM_H = 240, 180
TITLE_H = 36
CAP_H = 34
PLOT_H = 260
FONT = ImageFont.load_default(size=13)


def hex_rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def rgb255(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb)


def load_model(table_z: float, gravcomp: bool) -> mujoco.MjModel:
    """The DM scene with the floor at ``table_z``, widened ranges and compiled-in gravity compensation.

    Edited copies are written beside the originals: the scene includes the model and the
    model references 115 meshes by relative path, none of which resolve from a string.
    Gravity compensation must be in the XML; ``body_gravcomp`` assigned after load does
    nothing.
    """
    model_src = MODEL.read_text()
    if gravcomp:
        model_src = re.sub(ARM_BODY_RE, r'\1 gravcomp="1"', model_src)
    for joint, (lo, hi) in SOFT_LIMITS_DEG.items():
        match = re.search(rf'<joint name="{joint}"[^>]*range="([-0-9.e]+) ([-0-9.e]+)"', model_src)
        lo = min(np.deg2rad(lo), float(match.group(1)))
        hi = max(np.deg2rad(hi), float(match.group(2)))
        widened = f'{lo:.4f} {hi:.4f}"'
        model_src = re.sub(
            rf'(<joint name="{joint}"[^>]*range=")[^"]*"', lambda m: m.group(1) + widened, model_src
        )
        model_src = re.sub(
            rf'(<position [^>]*name="{joint}"[^>]*ctrlrange=")[^"]*"',
            lambda m: m.group(1) + widened,
            model_src,
        )
    scene_src = SCENE.read_text()
    scene_src = scene_src.replace('name="floor"', f'name="floor" pos="0 0 {table_z:.4f}"')

    tmp_model = MODEL.with_name(f"_sim_model_{os.getpid()}.xml")
    tmp_scene = SCENE.with_name(f"_sim_scene_{os.getpid()}.xml")
    tmp_model.write_text(model_src)
    tmp_scene.write_text(scene_src.replace(MODEL.name, tmp_model.name))
    try:
        return mujoco.MjModel.from_xml_path(str(tmp_scene))
    finally:
        tmp_model.unlink()
        tmp_scene.unlink()


def to_qpos(action_deg: np.ndarray) -> np.ndarray:
    """Dataset action (6 joints deg + gripper deg) to the 8-slot MJCF configuration."""
    slide = float(action_deg[GRIPPER_IDX]) / GRIPPER_DEG_OPEN * GRIPPER_SLIDE_OPEN
    slide = float(np.clip(slide, 0.0, GRIPPER_SLIDE_OPEN))
    return np.concatenate([np.deg2rad(action_deg[:ARM_DOF]), [slide, slide]])


def from_qpos(qpos: np.ndarray) -> np.ndarray:
    gripper = float(qpos[ARM_DOF]) / GRIPPER_SLIDE_OPEN * GRIPPER_DEG_OPEN
    return np.concatenate([np.rad2deg(qpos[:ARM_DOF]), [gripper]])


def anchor_camera(points: np.ndarray, table_z: float) -> mujoco.MjvCamera:
    """A camera framed on this anchor's end-effector paths, still for the whole chunk.

    ``points`` are every path drawn in the anchor (demo, samples, measured, achieved).
    The camera looks at their centre, backs off until they fit with the forearm around
    them, and sits at a three-quarter angle to the base-to-gripper direction (the base
    is at the origin) so the arm never stands between the camera and the gripper. A
    free camera's azimuth is the direction it looks along, so ``theta + 130`` puts it
    off the gripper's far side.
    """
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = float(np.linalg.norm(points - centre, axis=1).max())
    theta = float(np.degrees(np.arctan2(centre[1], centre[0])))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # The paths are fingertip paths with the gripper body above them and the arm reaching in
    # from the base, so the look-at point sits above the paths and a little back towards the base.
    camera.lookat[:] = (0.85 * centre[0], 0.85 * centre[1], max(centre[2], table_z + 0.03) + 0.09)
    camera.distance = float(np.clip(2.6 * radius + 0.45, 0.6, 0.95))
    camera.azimuth = theta + 130.0
    camera.elevation = -18.0
    return camera


def plain_lighting(model: mujoco.MjModel) -> None:
    """A flat light floor and a brighter headlight: the scene file's dark checker is for the live viewer."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    material = int(model.geom_matid[floor])
    model.mat_texid[material, :] = -1
    model.mat_rgba[material] = (0.86, 0.85, 0.82, 1.0)
    model.mat_reflectance[material] = 0.0
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    model.vis.headlight.diffuse[:] = (0.7, 0.7, 0.7)


class Arm:
    """The model's one arm: posed kinematically and tinted per panel.

    The visual meshes lose their material at construction so ``tint`` can recolour them
    through ``geom_rgba``, keeping each part's light/dark contrast and taking the panel's hue.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self.visual = [
            geom
            for geom in range(model.ngeom)
            if model.geom_group[geom] == 2 and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        self.luminance = []
        for geom in self.visual:
            material = int(model.geom_matid[geom])
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
            self.luminance.append(0.35 + 0.65 * float(rgba[:3].mean()))
            model.geom_matid[geom] = -1

    def tint(self, rgb: tuple[float, float, float]) -> None:
        for geom, luminance in zip(self.visual, self.luminance, strict=True):
            self.model.geom_rgba[geom, :3] = np.asarray(rgb) * luminance
            self.model.geom_rgba[geom, 3] = 1.0

    def pose(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)

    def ee_path(self, chunk_deg: np.ndarray) -> np.ndarray:
        points = []
        for action in chunk_deg:
            self.pose(to_qpos(action))
            points.append(self.data.xpos[self.ee_body].copy())
        return np.stack(points)


def execute(
    model: mujoco.MjModel, data: mujoco.MjData, state_deg: np.ndarray, chunk_deg: np.ndarray, substeps: int
):
    """Drive the position actuators through ``chunk_deg`` from rest at ``state_deg``.

    Returns the configuration after every tick, the end-effector position, and whether
    any arm geom touched the floor during that tick.
    """
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = to_qpos(state_deg)
    data.ctrl[:] = data.qpos
    mujoco.mj_forward(model, data)

    qpos = np.zeros((len(chunk_deg), model.nq))
    ee = np.zeros((len(chunk_deg), 3))
    touched = np.zeros(len(chunk_deg), dtype=bool)
    for step, action in enumerate(chunk_deg):
        data.ctrl[:] = to_qpos(action)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            for contact in data.contact[: data.ncon]:
                if floor in (contact.geom1, contact.geom2):
                    touched[step] = True
        qpos[step] = data.qpos
        ee[step] = data.xpos[ee_body]
    return qpos, ee, touched


class ChunkVideo:
    """The dataset's camera frames over a chunk window, as uint8 arrays per step.

    Only built when the dataset root the probe recorded exists here; otherwise the
    anchor still frames from the npz are shown instead.
    """

    def __init__(self, repo_id: str, root: str, fps: float):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(repo_id=repo_id, root=root)
        self.fps = fps

    def frames(self, episode: int, frame: int, horizon: int, label: str) -> list[np.ndarray]:
        from lerobot.datasets.video_utils import decode_video_frames

        key = f"observation.images.{label}"
        meta = self.dataset.meta
        if (
            key not in meta.video_keys
        ):  # npz labels are canonical roles; a raw rig root (inference) stores top / wrist
            from lerobot.datasets.diverse_actor_selection import CAMERA_ROLE_MAP

            raw = {role: name for name, role in CAMERA_ROLE_MAP["rebot"].items()}
            key = f"observation.images.{raw[label]}"
        start = float(meta.episodes[episode][f"videos/{key}/from_timestamp"])
        end = float(meta.episodes[episode][f"videos/{key}/to_timestamp"])
        timestamps = [start + (frame + step) / self.fps for step in range(horizon)]
        timestamps = [t for t in timestamps if t < end - 0.5 / self.fps]
        decoded = decode_video_frames(
            self.dataset.root / meta.get_video_file_path(episode, key),
            timestamps,
            self.dataset.tolerance_s,
            backend=self.dataset._video_backend,
            return_uint8=True,
        )
        return list(decoded.permute(0, 2, 3, 1).numpy())


def draw_path(
    scene: mujoco.MjvScene, points: np.ndarray, rgb: tuple[float, float, float], radius: float
) -> None:
    """Append a polyline of capsules to a scene already filled by ``update_scene``."""
    rgba = np.array([*rgb, 1.0], dtype=np.float32)
    for a, b in zip(points[:-1], points[1:], strict=True):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        scene.ngeom += 1


def draw_marker(
    scene: mujoco.MjvScene, point: np.ndarray, rgb: tuple[float, float, float], radius: float
) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(point, dtype=np.float64),
        np.eye(3).ravel(),
        np.array([*rgb, 1.0], dtype=np.float32),
    )
    scene.ngeom += 1


def render_panel(
    renderer: mujoco.Renderer,
    arm: Arm,
    camera: mujoco.MjvCamera,
    qpos: np.ndarray,
    paths: list,
    markers: list,
) -> np.ndarray:
    """One arm panel: the arm at ``qpos`` plus ``(points, rgb, radius)`` paths and ``(point, rgb, radius)`` markers."""
    arm.pose(qpos)
    renderer.update_scene(arm.data, camera)
    for points, rgb, radius in paths:
        draw_path(renderer.scene, points, rgb, radius)
    for point, rgb, radius in markers:
        draw_marker(renderer.scene, point, rgb, radius)
    return renderer.render().copy()


def joint_strip(
    width: int,
    joint_names: list[str],
    fps: float,
    gt_chunk: np.ndarray,
    samples: list[np.ndarray],
    sample_colors: list,
    measured: np.ndarray | None,
    achieved: list[np.ndarray] | None,
) -> tuple[np.ndarray, list[tuple[float, float, int, int]]]:
    """The joint curves of one anchor, drawn once, with the pixel geometry of every axis.

    Returns the RGB strip and, per axis, ``(x0, x_per_s, y_top, y_bottom)`` so the cursor
    at any tick is a column overlaid on the array: no second matplotlib draw per frame.
    """
    fig, axes = plt.subplots(1, len(joint_names), figsize=(width / 100, PLOT_H / 100), dpi=100)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.2, top=0.86, wspace=0.32)
    horizon = len(gt_chunk)
    time = np.arange(horizon) / fps
    for j, (ax, name) in enumerate(zip(axes, joint_names, strict=True)):
        ax.plot(time, gt_chunk[:, j], color=DEMO_COLOR, linewidth=2.0)
        if measured is not None and len(measured):
            ax.plot(time[: len(measured)], measured[:, j], color=MEASURED_COLOR, linewidth=2.5)
        for k, chunk in enumerate(samples):
            ax.plot(time[: len(chunk)], chunk[:, j], color=sample_colors[k], linewidth=1.5)
            if achieved is not None:
                ax.plot(
                    time[: len(achieved[k])],
                    achieved[k][:, j],
                    color=sample_colors[k],
                    linewidth=0.8,
                    linestyle="--",
                )
        ax.set_title(name, fontsize=8)
        ax.set_xlim(0.0, max(time[-1], 1.0 / fps))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("deg", fontsize=7)
    axes[len(axes) // 2].set_xlabel("s", fontsize=7)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    height = image.shape[0]
    geometry = []
    for ax in axes:
        x0 = ax.transData.transform((0.0, 0.0))[0]
        x1 = ax.transData.transform((1.0, 0.0))[0]
        bbox = ax.get_window_extent()
        geometry.append((x0, x1 - x0, int(height - bbox.y1), int(height - bbox.y0)))
    plt.close(fig)
    return image, geometry


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((width, height), Image.BILINEAR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--records", required=True, help="action_trace_records.npz from the probe")
    parser.add_argument(
        "--out", default=None, help="output directory; default action_trace_sim/ beside the records"
    )
    parser.add_argument("--anchors", default=None, help="comma-separated anchor indices; default all")
    parser.add_argument("--episodes", default=None, help="comma-separated episode indices; default all")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="video seconds per chunk second; 2 plays the chunks at half speed",
    )
    parser.add_argument(
        "--execute", action="store_true", help="run the chunks through MuJoCo physics instead of posing them"
    )
    parser.add_argument("--substeps", type=int, default=17, help="physics steps per action tick (--execute)")
    parser.add_argument(
        "--no-gravcomp", action="store_true", help="let the arm sag under its own weight (--execute)"
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="dataset to play video from; default the root recorded in the npz",
    )
    parser.add_argument("--no-video", action="store_true", help="show the anchor still frames only")
    args = parser.parse_args()

    records = np.load(args.records)
    fps = float(records["fps"])
    table_z = float(records["table_z"])
    joint_names = [str(name) for name in records["joint_names"]]
    camera_labels = [key[len("image_") :] for key in records.files if key.startswith("image_")]
    has_fast = "fast_chunk" in records.files
    n_anchors, n_samples = records["sample_chunks"].shape[:2]
    chosen = list(range(n_anchors))
    if args.anchors:
        chosen = [int(a) for a in args.anchors.split(",")]
    if args.episodes:
        wanted = {int(e) for e in args.episodes.split(",")}
        chosen = [a for a in chosen if int(records["episode"][a]) in wanted]

    video = None
    dataset_root = args.dataset_root or (
        str(records["dataset_root"]) if "dataset_root" in records.files else ""
    )
    if not args.no_video and dataset_root and Path(dataset_root).is_dir():
        video = ChunkVideo(str(records["dataset_repo_id"]), dataset_root, fps)
        print(f"playing video from {dataset_root}")
    elif not args.no_video:
        print(f"dataset root {dataset_root!r} not found here; showing anchor still frames")

    model = load_model(table_z, gravcomp=not args.no_gravcomp)
    model.opt.timestep = 1.0 / fps / args.substeps
    data = mujoco.MjData(model)
    arm = Arm(model)
    plain_lighting(model)
    renderer = mujoco.Renderer(model, PANEL, PANEL)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    demo_rgb = hex_rgb(DEMO_COLOR)
    measured_rgb = hex_rgb(MEASURED_COLOR)
    fast_rgb = hex_rgb(FAST_COLOR)
    sample_colors = [hex_rgb(SAMPLE_COLORS[k % len(SAMPLE_COLORS)]) for k in range(n_samples)]
    panel_colors = [demo_rgb, *sample_colors]

    out = Path(args.out) if args.out else Path(args.records).with_name("action_trace_sim")
    out.mkdir(parents=True, exist_ok=True)
    width = (n_samples + 1) * PANEL + CAM_W
    height = TITLE_H + PANEL + CAP_H + PLOT_H
    video_fps = int(round(fps / args.time_scale))
    legend = "grey = demo command, colours = flow samples"
    if args.execute:
        legend = "grey = demo command, black = real follower, colours = flow samples (dashed = sim achieved), small sphere = command target"
    if has_fast:
        legend += f", {FAST_COLOR} path = FAST decode"

    anchors_summary = []
    for anchor in chosen:
        episode = int(records["episode"][anchor])
        frame = int(records["frame"][anchor])
        subtask = str(records["subtask"][anchor]) or "-"
        state = np.asarray(records["state"][anchor], dtype=np.float64)[: len(joint_names)]
        gt_chunk = records["gt_chunk"][anchor]
        gt_chunk = gt_chunk[~np.isnan(gt_chunk[:, 0])]
        measured = records["measured_states"][anchor]
        measured = measured[~np.isnan(measured[:, 0])][:, : len(joint_names)]
        samples = [chunk[~np.isnan(chunk[:, 0])] for chunk in records["sample_chunks"][anchor]]
        horizon = len(gt_chunk)
        command_paths = [arm.ee_path(chunk) for chunk in samples]
        demo_path = arm.ee_path(gt_chunk)
        measured_path = arm.ee_path(measured) if len(measured) else demo_path[:1]
        fast_path = None
        if has_fast:
            fast_chunk = records["fast_chunk"][anchor]
            fast_chunk = fast_chunk[~np.isnan(fast_chunk[:, 0])]
            if len(fast_chunk):
                fast_path = arm.ee_path(fast_chunk)
        runs = (
            [execute(model, data, state, chunk, args.substeps) for chunk in samples] if args.execute else None
        )
        clips = (
            {label: video.frames(episode, frame, horizon, label) for label in camera_labels} if video else {}
        )
        drawn = [demo_path, measured_path, *command_paths]
        if fast_path is not None:
            drawn.append(fast_path)
        if args.execute:
            drawn += [ee for _, ee, _ in runs]
        camera = anchor_camera(np.concatenate(drawn), table_z)

        final_spread = max(
            (
                np.linalg.norm(command_paths[i][-1] - command_paths[j][-1])
                for i in range(n_samples)
                for j in range(i + 1, n_samples)
            ),
            default=0.0,
        )
        per_sample = []
        captions = [("demo: real follower" if args.execute else "demo command", "")]
        for k, path in enumerate(command_paths):
            row = {
                "clearance_mm": float(records["sample_clearance"][anchor][k] * 1000),
                "final_vs_demo_mm": float(np.linalg.norm(path[-1] - demo_path[-1]) * 1000),
                "path_mm": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * 1000),
            }
            if args.execute:
                qpos, ee, touched = runs[k]
                row["contact_ticks"] = int(touched.sum())
                row["max_lag_mm"] = float(np.linalg.norm(ee - path, axis=1).max() * 1000)
                row["final_vs_demo_mm"] = float(np.linalg.norm(ee[-1] - demo_path[-1]) * 1000)
                second = f"table contact {row['contact_ticks']} ticks   max lag {row['max_lag_mm']:.0f} mm"
            else:
                second = f"path {row['path_mm']:.0f} mm"
            captions.append(
                (
                    f"sample {k}   clearance {row['clearance_mm']:.0f} mm   final vs demo {row['final_vs_demo_mm']:.0f} mm",
                    second,
                )
            )
            per_sample.append(row)
        title = f"anchor {anchor}   episode {episode}   frame {frame}   subtask: {subtask}   final EE spread across samples: {final_spread * 1000:.0f} mm"

        achieved = [np.stack([from_qpos(q) for q in qpos]) for qpos, _, _ in runs] if args.execute else None
        strip, geometry = joint_strip(
            width,
            joint_names,
            fps,
            gt_chunk,
            samples,
            sample_colors,
            measured if args.execute else None,
            achieved,
        )
        stills = {label: fit(records[f"image_{label}"][anchor], CAM_W, CAM_H) for label in camera_labels}

        name = f"anchor{anchor:03d}_ep{episode:04d}_fr{frame:06d}.mp4"
        writer = VideoWriter(out / name, video_fps, width, height)
        for step in range(horizon):
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
            # Arm panels: demo first, then every sample.
            for p in range(n_samples + 1):
                paths = [(demo_path, demo_rgb, 0.002)]
                if fast_path is not None:
                    paths.append((fast_path, fast_rgb, 0.0015))
                if p == 0:
                    if args.execute:
                        paths.append((measured_path, measured_rgb, 0.002))
                        pose_step = min(step, len(measured) - 1)
                        qpos = to_qpos(measured[pose_step]) if len(measured) else to_qpos(gt_chunk[step])
                        markers = [
                            (demo_path[step], demo_rgb, 0.005),
                            (measured_path[pose_step], measured_rgb, 0.007),
                        ]
                    else:
                        qpos = to_qpos(gt_chunk[step])
                        markers = [(demo_path[step], demo_rgb, 0.007)]
                else:
                    k = p - 1
                    paths.append((command_paths[k], sample_colors[k], 0.0015))
                    if args.execute:
                        qpos_run, ee, _ = runs[k]
                        qpos = qpos_run[step]
                        paths.append((ee[: step + 1], sample_colors[k], 0.003))
                        markers = [(command_paths[k][step], sample_colors[k], 0.005)]
                    else:
                        qpos = to_qpos(samples[k][step])
                        markers = [(command_paths[k][step], sample_colors[k], 0.007)]
                arm.tint(panel_colors[p])
                x = p * PANEL
                canvas[TITLE_H : TITLE_H + PANEL, x : x + PANEL] = render_panel(
                    renderer, arm, camera, qpos, paths, markers
                )
                canvas[TITLE_H : TITLE_H + 4, x : x + PANEL] = rgb255(panel_colors[p])
            # Cameras, stacked at the right end of the arm row.
            x = (n_samples + 1) * PANEL
            for c, label in enumerate(camera_labels):
                frames = clips.get(label)
                image = fit(frames[step], CAM_W, CAM_H) if frames and step < len(frames) else stills[label]
                canvas[TITLE_H + c * CAM_H : TITLE_H + (c + 1) * CAM_H, x : x + CAM_W] = image
            # Joint strip with the cursor.
            plot_y = TITLE_H + PANEL + CAP_H
            canvas[plot_y:, :] = strip
            for x0, x_per_s, y_top, y_bottom in geometry:
                cursor = int(round(x0 + x_per_s * step / fps))
                canvas[plot_y + y_top : plot_y + y_bottom, cursor - 1 : cursor + 1] = (30, 30, 30)

            image = Image.fromarray(canvas)
            draw = ImageDraw.Draw(image)
            draw.text((8, 4), title, fill=(20, 20, 20), font=FONT)
            draw.text((8, 20), legend, fill=(90, 90, 90), font=FONT)
            for p, (first, second) in enumerate(captions):
                draw.text((p * PANEL + 8, TITLE_H + PANEL + 2), first, fill=(20, 20, 20), font=FONT)
                draw.text((p * PANEL + 8, TITLE_H + PANEL + 17), second, fill=(90, 90, 90), font=FONT)
            for c, label in enumerate(camera_labels):
                draw.text((x + 6, TITLE_H + c * CAM_H + 4), label, fill=(255, 255, 255), font=FONT)
            writer.write(np.asarray(image))
        writer.close()

        anchors_summary.append(
            {
                "anchor": int(anchor),
                "episode": episode,
                "frame": frame,
                "subtask": subtask,
                "file": name,
                "final_spread_mm": float(final_spread * 1000),
                "samples": per_sample,
            }
        )
        print(
            f"anchor {anchor}: ep{episode} frame {frame}  final spread {final_spread * 1000:.0f} mm  -> {name}"
        )

    renderer.close()
    summary = {
        "execute": bool(args.execute),
        "time_scale": args.time_scale,
        "n_anchors": len(anchors_summary),
        "n_samples": int(n_samples),
        "final_spread_mm_mean": float(np.mean([a["final_spread_mm"] for a in anchors_summary]))
        if anchors_summary
        else None,
        "anchors": anchors_summary,
    }
    if args.execute and anchors_summary:
        summary["contact_anchor_fraction"] = float(
            np.mean([any(s["contact_ticks"] > 0 for s in a["samples"]) for a in anchors_summary])
        )
        summary["max_lag_mm_mean"] = float(
            np.mean([s["max_lag_mm"] for a in anchors_summary for s in a["samples"]])
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(anchors_summary)} videos and summary.json to {out}")


if __name__ == "__main__":
    main()
