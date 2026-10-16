"""Play the Action Inspector's chunks on the MuJoCo reBot and record a rerun log.

Input is ``action_trace_records.npz``, written beside ``action_trace.html`` by
``action_trace_probe``: per anchor, the measured state, the demonstrated chunk, every
filtered flow sample, and the follower states the demonstration actually produced.

    uv run python -m lerobot.probes.action_trace_sim --records <run>/validation/step_00001200/action_trace/action_trace_records.npz
    rerun <run>/validation/step_00001200/action_trace/action_trace_sim.rrd

By default everything drawn is a commanded action. Each flow sample is an arm posed
straight from its chunk, tick by tick, next to the demonstrated chunk's end-effector path
and marker, with the dataset's camera video playing over the same frames when the dataset
root recorded in the npz exists on this machine. No physics, no servo lag: what the policy
asked for, against what the demonstrator asked for.

``--execute`` instead drives the MuJoCo position actuators through each chunk from the
measured demo state, one target per tick, and poses the arms from what the plant did.
That adds the achieved trails, the real follower's path, the sim-vs-command joint curves,
end-effector lag, and a table-contact flag. The scene is the Menagerie model retargeted
to our DM geometry (``mjcf/retarget_to_dm.py``): end effector exact to a micron, meshes
and inertials still RobStride-shaped, actuator gains inherited from the lighter RS arm
and not identified against the DM motors. Read contact and the shape of the lag from
it, not millimetres of lag.

One ``time`` timeline runs through the anchors in order, with a pause between them; the
``anchor`` timeline jumps between anchors. Chunks are one second long, so slow the
viewer's playback speed down to read them.
"""

from __future__ import annotations

import argparse
import io
import os
import re
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from lerobot.probes.action_trace_probe import FAST_COLOR, SAMPLE_COLORS
from lerobot.robots.rebot_b601_follower.kinematics import ARM_DOF

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


def hex_rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


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
        model_src = re.sub(rf'(<joint name="{joint}"[^>]*range=")[^"]*"', lambda m: m.group(1) + widened, model_src)
        model_src = re.sub(
            rf'(<position [^>]*name="{joint}"[^>]*ctrlrange=")[^"]*"', lambda m: m.group(1) + widened, model_src
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


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


class ArmMeshes:
    """The arm's visual geoms: logged once as static meshes under a prefix, posed per step."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.geoms = []
        for geom in range(model.ngeom):
            if model.geom_group[geom] != 2 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = int(model.geom_dataid[geom])
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
            name = f"{body}/{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh)}"
            vert = model.mesh_vertadr[mesh]
            face = model.mesh_faceadr[mesh]
            vertices = model.mesh_vert[vert : vert + model.mesh_vertnum[mesh]].astype(np.float32)
            faces = model.mesh_face[face : face + model.mesh_facenum[mesh]].astype(np.uint32)
            material = int(model.geom_matid[geom])
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
            self.geoms.append((geom, name, vertices, faces, vertex_normals(vertices, faces), rgba[:3].copy()))
        self.scratch = mujoco.MjData(model)
        self.ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)

    def log_static(self, prefix: str, tint: tuple[float, float, float] | None = None, alpha: float = 1.0) -> None:
        for _, name, vertices, faces, normals, rgb in self.geoms:
            color = rgb
            if tint is not None:
                # Keep the material's light/dark contrast, take the sample's hue.
                luminance = 0.35 + 0.65 * float(rgb.mean())
                color = np.asarray(tint) * luminance
            rr.log(
                f"{prefix}/{name}",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_normals=normals,
                    albedo_factor=(float(color[0]), float(color[1]), float(color[2]), alpha),
                ),
                static=True,
            )

    def log_pose(self, prefix: str, qpos: np.ndarray) -> None:
        """Pose the meshes at ``qpos`` kinematically."""
        self.scratch.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.scratch)
        for geom, name, *_ in self.geoms:
            rr.log(
                f"{prefix}/{name}",
                rr.Transform3D(
                    translation=self.scratch.geom_xpos[geom], mat3x3=self.scratch.geom_xmat[geom].reshape(3, 3)
                ),
            )

    def ee_path(self, chunk_deg: np.ndarray) -> np.ndarray:
        points = []
        for action in chunk_deg:
            self.scratch.qpos[:] = to_qpos(action)
            mujoco.mj_kinematics(self.model, self.scratch)
            points.append(self.scratch.xpos[self.ee_body].copy())
        return np.stack(points)


def execute(model: mujoco.MjModel, data: mujoco.MjData, state_deg: np.ndarray, chunk_deg: np.ndarray, substeps: int):
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
    """The dataset's camera frames over a chunk window, as half-res JPEG bytes per step.

    Only built when the dataset root the probe recorded exists here; otherwise the
    anchor still frames from the npz are shown instead.
    """

    def __init__(self, repo_id: str, root: str, fps: float):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(repo_id=repo_id, root=root)
        self.fps = fps

    def frames(self, episode: int, frame: int, horizon: int, label: str) -> list[bytes]:
        from PIL import Image

        from lerobot.datasets.video_utils import decode_video_frames

        key = f"observation.images.{label}"
        meta = self.dataset.meta
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
        encoded = []
        for image in decoded.permute(0, 2, 3, 1).numpy()[:, ::2, ::2]:
            buffer = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="JPEG", quality=85)
            encoded.append(buffer.getvalue())
        return encoded


def blueprint(camera_labels: list[str], joint_names: list[str], execute_mode: bool) -> rrb.Blueprint:
    window = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-1.5),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=1.5),
    )
    joint_tabs = rrb.Tabs(
        *[rrb.TimeSeriesView(origin=f"/plots/joints/{name}", name=name, time_ranges=window) for name in joint_names]
    )
    right = [joint_tabs]
    shares = [1]
    if execute_mode:
        right += [
            rrb.TimeSeriesView(origin="/plots/ee_lag_mm", name="EE lag, achieved vs commanded (mm)", time_ranges=window),
            rrb.TimeSeriesView(origin="/plots/table_contact", name="table contact", time_ranges=window),
        ]
        shares = [2, 1, 0.7]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", name="actions", contents=["+ $origin/**", "- /plots/**", "- /cameras/**", "- /info"]),
            rrb.Vertical(
                rrb.Tabs(*[rrb.Spatial2DView(origin=f"/cameras/{label}", name=label) for label in camera_labels]),
                rrb.TextDocumentView(origin="/info", name="anchor"),
                row_shares=[2, 1.3],
            ),
            rrb.Vertical(*right, row_shares=shares),
            column_shares=[3, 1.3, 1.7],
        ),
        collapse_panels=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", required=True, help="action_trace_records.npz from the probe")
    parser.add_argument("--out", default=None, help=".rrd path; default action_trace_sim.rrd beside the records")
    parser.add_argument("--spawn", action="store_true", help="open the rerun viewer live instead of writing the .rrd")
    parser.add_argument("--anchors", default=None, help="comma-separated anchor indices; default all")
    parser.add_argument("--episodes", default=None, help="comma-separated episode indices; default all")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds of timeline between anchors")
    parser.add_argument("--execute", action="store_true", help="run the chunks through MuJoCo physics instead of posing them")
    parser.add_argument("--substeps", type=int, default=17, help="physics steps per action tick (--execute)")
    parser.add_argument("--no-gravcomp", action="store_true", help="let the arm sag under its own weight (--execute)")
    parser.add_argument("--demo-arm", action="store_true", help="also draw the demo command as a translucent ghost arm")
    parser.add_argument("--dataset-root", default=None, help="dataset to play video from; default the root recorded in the npz")
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
    dataset_root = args.dataset_root or (str(records["dataset_root"]) if "dataset_root" in records.files else "")
    if not args.no_video and dataset_root and Path(dataset_root).is_dir():
        video = ChunkVideo(str(records["dataset_repo_id"]), dataset_root, fps)
        print(f"playing video from {dataset_root}")
    elif not args.no_video:
        print(f"dataset root {dataset_root!r} not found here; showing anchor still frames")

    model = load_model(table_z, gravcomp=not args.no_gravcomp)
    model.opt.timestep = 1.0 / fps / args.substeps
    data = mujoco.MjData(model)
    arm = ArmMeshes(model)
    sample_colors = [hex_rgb(SAMPLE_COLORS[k % len(SAMPLE_COLORS)]) for k in range(n_samples)]

    out = args.out or str(Path(args.records).with_name("action_trace_sim.rrd"))
    layout = blueprint(camera_labels, joint_names, args.execute)
    rr.init("action_trace_sim", spawn=args.spawn, default_blueprint=layout)
    if not args.spawn:
        rr.save(out, default_blueprint=layout)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "scene/table",
        rr.Boxes3D(centers=[[0.25, 0.0, table_z - 0.01]], half_sizes=[[0.75, 0.75, 0.01]], colors=[(0.82, 0.80, 0.76)], fill_mode="solid"),
        static=True,
    )
    if args.demo_arm:
        arm.log_static("demo/arm", tint=hex_rgb(DEMO_COLOR), alpha=0.35)
    for k in range(n_samples):
        arm.log_static(f"sample{k}/arm", tint=sample_colors[k])
    for name in joint_names:
        rr.log(f"plots/joints/{name}/demo_cmd", rr.SeriesLines(colors=[hex_rgb(DEMO_COLOR)], widths=[2.0], names=["demo command"]), static=True)
        for k in range(n_samples):
            rr.log(f"plots/joints/{name}/sample{k}_cmd", rr.SeriesLines(colors=[sample_colors[k]], widths=[2.0], names=[f"sample {k} command"]), static=True)
        if args.execute:
            rr.log(f"plots/joints/{name}/demo_meas", rr.SeriesLines(colors=[hex_rgb(MEASURED_COLOR)], widths=[2.5], names=["real follower"]), static=True)
            for k in range(n_samples):
                rr.log(f"plots/joints/{name}/sample{k}_sim", rr.SeriesLines(colors=[sample_colors[k]], widths=[1.0], names=[f"sample {k} sim"]), static=True)
    if args.execute:
        for k in range(n_samples):
            rr.log(f"plots/ee_lag_mm/sample{k}", rr.SeriesLines(colors=[sample_colors[k]], names=[f"sample {k}"]), static=True)
            rr.log(f"plots/table_contact/sample{k}", rr.SeriesLines(colors=[sample_colors[k]], names=[f"sample {k}"]), static=True)

    clock = 0.0
    for anchor in chosen:
        episode = int(records["episode"][anchor])
        frame = int(records["frame"][anchor])
        state = np.asarray(records["state"][anchor], dtype=np.float64)[: len(joint_names)]
        gt_chunk = records["gt_chunk"][anchor]
        gt_chunk = gt_chunk[~np.isnan(gt_chunk[:, 0])]
        measured = records["measured_states"][anchor]
        measured = measured[~np.isnan(measured[:, 0])][:, : len(joint_names)]
        samples = [chunk[~np.isnan(chunk[:, 0])] for chunk in records["sample_chunks"][anchor]]
        horizon = len(gt_chunk)
        command_paths = [arm.ee_path(chunk) for chunk in samples]
        demo_path = arm.ee_path(gt_chunk)
        runs = [execute(model, data, state, chunk, args.substeps) for chunk in samples] if args.execute else None
        clips = {label: video.frames(episode, frame, horizon, label) for label in camera_labels} if video else {}

        rr.set_time("anchor", sequence=int(anchor))
        rr.set_time("time", duration=clock)
        final_spread = max(
            (np.linalg.norm(command_paths[i][-1] - command_paths[j][-1]) for i in range(n_samples) for j in range(i + 1, n_samples)),
            default=0.0,
        )
        lines = [
            f"**anchor {anchor}**  episode {episode}  frame {frame}",
            "",
            f"subtask: {records['subtask'][anchor] or '-'}",
            "",
            f"final EE spread across samples: {final_spread * 1000:.0f} mm",
            "",
        ]
        if args.execute:
            lines += ["| sample | probe clearance (mm) | sim table contact (ticks) | max EE lag (mm) | final EE vs demo (mm) |", "|---|---|---|---|---|"]
            for k, (qpos, ee, touched) in enumerate(runs):
                lag = np.linalg.norm(ee - command_paths[k], axis=1)
                lines.append(
                    f"| {k} | {records['sample_clearance'][anchor][k] * 1000:.0f} | {int(touched.sum())} | {lag.max() * 1000:.0f} | {np.linalg.norm(ee[-1] - demo_path[-1]) * 1000:.0f} |"
                )
        else:
            lines += ["| sample | probe clearance (mm) | final EE vs demo (mm) | path length (mm) |", "|---|---|---|---|"]
            for k, path in enumerate(command_paths):
                length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                lines.append(
                    f"| {k} | {records['sample_clearance'][anchor][k] * 1000:.0f} | {np.linalg.norm(path[-1] - demo_path[-1]) * 1000:.0f} | {length * 1000:.0f} |"
                )
        rr.log("info", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))
        for label in camera_labels:
            if label not in clips:
                rr.log(f"cameras/{label}", rr.Image(records[f"image_{label}"][anchor]))
        rr.log("demo/command_path", rr.LineStrips3D([demo_path], colors=[hex_rgb(DEMO_COLOR)], radii=[0.002]))
        for k in range(n_samples):
            rr.log(f"sample{k}/command_path", rr.LineStrips3D([command_paths[k]], colors=[sample_colors[k]], radii=[0.0015]))
        if has_fast:
            fast_chunk = records["fast_chunk"][anchor]
            fast_chunk = fast_chunk[~np.isnan(fast_chunk[:, 0])]
            if len(fast_chunk):
                rr.log("fast/command_path", rr.LineStrips3D([arm.ee_path(fast_chunk)], colors=[hex_rgb(FAST_COLOR)], radii=[0.0015]))
        if args.execute:
            rr.log("demo/measured_path", rr.LineStrips3D([arm.ee_path(measured)], colors=[hex_rgb(MEASURED_COLOR)], radii=[0.002]))

        for step in range(horizon):
            rr.set_time("time", duration=clock + step / fps)
            for label, frames in clips.items():
                if step < len(frames):
                    rr.log(f"cameras/{label}", rr.EncodedImage(contents=frames[step], media_type="image/jpeg"))
            if args.demo_arm:
                arm.log_pose("demo/arm", to_qpos(gt_chunk[step]))
            rr.log("demo/ee", rr.Points3D([demo_path[step]], colors=[hex_rgb(DEMO_COLOR)], radii=[0.006]))
            for j, name in enumerate(joint_names):
                rr.log(f"plots/joints/{name}/demo_cmd", rr.Scalars(float(gt_chunk[step, j])))
                if args.execute and step < len(measured):
                    rr.log(f"plots/joints/{name}/demo_meas", rr.Scalars(float(measured[step, j])))
            for k in range(n_samples):
                if args.execute:
                    qpos, ee, touched = runs[k]
                    arm.log_pose(f"sample{k}/arm", qpos[step])
                    rr.log(f"sample{k}/target", rr.Points3D([command_paths[k][step]], colors=[sample_colors[k]], radii=[0.005]))
                    rr.log(f"sample{k}/achieved_path", rr.LineStrips3D([ee[: step + 1]], colors=[sample_colors[k]], radii=[0.003]))
                    achieved = from_qpos(qpos[step])
                    for j, name in enumerate(joint_names):
                        rr.log(f"plots/joints/{name}/sample{k}_sim", rr.Scalars(float(achieved[j])))
                    rr.log(f"plots/ee_lag_mm/sample{k}", rr.Scalars(float(np.linalg.norm(ee[step] - command_paths[k][step]) * 1000)))
                    rr.log(f"plots/table_contact/sample{k}", rr.Scalars(float(touched[step])))
                else:
                    arm.log_pose(f"sample{k}/arm", to_qpos(samples[k][step]))
                    rr.log(f"sample{k}/ee", rr.Points3D([command_paths[k][step]], colors=[sample_colors[k]], radii=[0.005]))
                for j, name in enumerate(joint_names):
                    rr.log(f"plots/joints/{name}/sample{k}_cmd", rr.Scalars(float(samples[k][step, j])))
        clock += horizon / fps + args.pause
        print(f"anchor {anchor}: ep{episode} frame {frame}  final spread {final_spread * 1000:.0f} mm")
    if not args.spawn:
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
