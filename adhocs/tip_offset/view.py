"""Rerun recording of the tip-offset targets of one episode: the arm, the table as seen by the wrist depth camera,
the tip path and the arrow from the tip to its target, on a frame slider. Run build.py first."""
import argparse, glob, os
import av, cv2, mujoco, numpy as np, pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from lerobot.annotation.corrections.arm import Arm, frame
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import ARM_BODIES, GRIPPER_DEG_OPEN, GRIPPER_SLIDE_OPEN

FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966  # D405 at 640 x 480
STRIDE = 4  # every 4th pixel of the depth image, both ways
VOXEL = 0.003  # the episode map: one point per 3 mm cell that at least MAP_FRAMES depth frames saw
MAP_FRAMES = 5
WHITE = 245
COLORS = {"grasp": (150, 70, 200), "release": (30, 100, 225)}  # the event points
IN_VIEW, OUT_OF_VIEW = (30, 160, 70), (215, 40, 40)  # the arrow: the target is inside / outside the wrist image
AXES = [(215, 40, 40), (40, 160, 70), (30, 100, 225)]

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--episodes", default="0", help="comma list, or 'all'")
args = ap.parse_args()
root = args.root.rstrip("/")
out = f"outputs/analysis/tip_offset/{os.path.basename(root)}"

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/data/*/*.parquet"))]).sort_values("index").reset_index(drop=True)
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/meta/episodes/*/*.parquet"))]).set_index("episode_index")
targets = pd.read_parquet(f"lerobot/annotation_data/tip_offset/{os.path.basename(root)}/targets.parquet")
arm = Arm()
m, d = arm.gripper.m, arm.gripper.d
vv, uu = np.mgrid[0:480:STRIDE, 0:640:STRIDE]
rays = np.stack([(uu - CX) / FX, (vv - CY) / FY, np.ones_like(uu, dtype=float)], -1)

# the arm's visual meshes, in their own frames
bodies = {m.body(b).id for b in ("base_link", *ARM_BODIES)}
meshes = {}
for g in range(m.ngeom):
    if m.geom_bodyid[g] in bodies and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        f = m.mesh_face[m.mesh_faceadr[k] : m.mesh_faceadr[k] + m.mesh_facenum[k]]
        normal = np.zeros_like(v)
        face_normal = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for c in range(3):
            np.add.at(normal, f[:, c], face_normal)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        rgb = 255 * (0.25 + 0.75 * m.mat_rgba[m.geom_matid[g], :3])  # the materials are near black; lifted to read on screen
        meshes[g] = rr.Mesh3D(vertex_positions=v, triangle_indices=f, vertex_normals=normal, albedo_factor=rgb.astype(np.uint8))


def video(key, e):
    """Frames of one episode's video, in order (rgb)."""
    r = eps.loc[e]
    c = av.open(f"{root}/videos/{key}/chunk-{r[f'videos/{key}/chunk_index']:03d}/file-{r[f'videos/{key}/file_index']:03d}.mp4")
    st = c.streams.video[0]
    t0 = r[f"videos/{key}/from_timestamp"]
    c.seek(int(max(t0 - 1, 0) / st.time_base), stream=st)
    for fr in c.decode(st):
        if float(fr.pts * st.time_base) >= t0 - 1e-3:
            yield fr.to_ndarray(format="rgb24")


def jpeg(rgb):
    return rr.EncodedImage(contents=cv2.imencode(".jpg", rgb[..., ::-1])[1].tobytes(), media_type="image/jpeg")


for e in sorted(df.episode_index.unique()) if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]:
    g = df[df.episode_index == e]
    t = targets.loc[g.index].reset_index(drop=True)
    state = np.stack(g["observation.state"])
    n = len(g)

    # pass 1: poses of every frame, cloud of every depth frame
    tip, rot, geoms, cams, clouds = np.zeros((n, 3)), np.zeros((n, 3, 3)), {}, {}, {}
    wrist, top = video("observation.images.wrist", e), video("observation.images.top", e)
    for i, q in enumerate(state):
        rgb, top_rgb = next(wrist), next(top)
        d.qpos[arm.gripper.q0 + 6 : arm.gripper.q0 + 8] = q[6] / GRIPPER_DEG_OPEN * GRIPPER_SLIDE_OPEN
        tip[i], a, j = arm.gripper.fk(q)
        rot[i] = frame(a, j)
        f = f"{root}/depth/wrist.depth/episode-{e:06d}/frame-{i:06d}.png"
        if not os.path.exists(f):  # depth every 3rd frame
            continue
        geoms[i] = {k: (d.geom_xpos[k].copy(), d.geom_xmat[k].reshape(3, 3).copy()) for k in meshes}
        cam_p, cam_r = arm.gripper.wrist(q)
        depth = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-4
        kernel = np.ones((5, 5), np.uint8)
        ok = (depth > 0.065) & (depth < 0.6) & (cv2.dilate(depth, kernel) - cv2.erode(depth, kernel) < 0.015)  # no holes, no edges
        ok, depth = ok[::STRIDE, ::STRIDE], depth[::STRIDE, ::STRIDE]
        points = (rays[ok] * depth[ok][:, None]) @ cam_r.T + cam_p
        near = (points[:, 2] > -0.03) & (points[:, 2] < 0.3)  # the table and what stands on it, not the room behind
        clouds[i] = (points[near].astype(np.float32), rgb[::STRIDE, ::STRIDE][ok][near], depth[ok][near] * STRIDE / FX * 0.6)
        cams[i] = (cam_p, cam_r, rgb, top_rgb)
    world = tip + np.einsum("nij,nj->ni", rot, t[["x", "y", "z"]].fillna(0).to_numpy())  # the event point of each frame

    # the episode map: cells seen by enough depth frames, mean color, faded
    cells, frames_seen, color = [], [], []
    for i, (p, c, _) in clouds.items():
        key, first = np.unique(np.floor(p / VOXEL).astype(np.int32), axis=0, return_index=True)
        cells.append(key); color.append(c[first])
    key, inverse, count = np.unique(np.concatenate(cells), axis=0, return_inverse=True, return_counts=True)
    color = np.stack([np.bincount(inverse.ravel(), np.concatenate(color)[:, c], len(key)) for c in range(3)], 1) / count[:, None]
    seen = count >= MAP_FRAMES
    map_points, map_colors = (key[seen] + 0.5) * VOXEL, (0.55 * color[seen] + 0.45 * WHITE).astype(np.uint8)

    rr.init(f"tip offset, episode {e}", recording_id=f"tip_offset_{os.path.basename(root)}_{e}")
    rr.save(f"{out}/episode_{e:03d}.rrd")
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="scene", background=[WHITE, WHITE, WHITE], line_grid=rrb.LineGrid3D(True, spacing=0.1, stroke_width=0.6, color=[200, 200, 200])),
        rrb.Vertical(rrb.TextDocumentView(origin="subtask", name="subtask"), rrb.Spatial2DView(origin="wrist", name="wrist camera"),
                     rrb.Spatial2DView(origin="top", name="top camera"),
                     rrb.TimeSeriesView(origin="offset", name="offset in the tool frame (cm)", plot_legend=rrb.Corner2D.RightTop), row_shares=[1, 4, 4, 4]),
        column_shares=[5, 3]), collapse_panels=True))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/map", rr.Points3D(map_points, colors=map_colors, radii=0.6 * VOXEL), static=True)
    rr.log("world/tip_path", rr.LineStrips3D([tip], colors=(110, 110, 110), radii=0.0007), static=True)
    events = t[t.kind != "none"].drop_duplicates("event_index")
    rr.log("world/events", rr.Points3D(tip[(events.event_index - g.index[0]).astype(int)], colors=[COLORS[k] for k in events.kind], radii=0.0035), static=True)
    for k, mesh in meshes.items():
        rr.log(f"world/arm/{k}", mesh, static=True)
    for c, color in zip(["x", "y", "z", "distance"], [*AXES, (60, 60, 60)]):
        rr.log(f"offset/{c}", rr.SeriesLines(colors=color, names=c, widths=1.5), static=True)

    for i, (points, colors, radii) in clouds.items():
        rr.set_time("frame", sequence=i)
        cam_p, cam_r, rgb, top_rgb = cams[i]
        for k, (p, r) in geoms[i].items():
            rr.log(f"world/arm/{k}", rr.Transform3D(translation=p, mat3x3=r))
        rr.log("world/cloud", rr.Points3D(points, colors=colors, radii=radii))
        rr.log("world/camera", rr.Transform3D(translation=cam_p, mat3x3=cam_r), rr.Pinhole(focal_length=[FX, FY], principal_point=[CX, CY], resolution=[640, 480], image_plane_distance=0.02, color=(120, 120, 120)))
        rr.log("wrist", jpeg(rgb))
        rr.log("top", jpeg(top_rgb))
        rr.log("world/tool", rr.Arrows3D(origins=[tip[i]] * 3, vectors=0.015 * rot[i].T, colors=AXES, radii=0.0006))
        k = t.kind[i]
        if k == "none":
            rr.log("world/target", rr.Clear(recursive=True))
            rr.log("wrist/target", rr.Clear(recursive=False))
            rr.log("subtask", rr.TextDocument(f"**{t.subtask[i]}**: no target", media_type="text/markdown"))
            continue
        color = IN_VIEW if t.in_view[i] else OUT_OF_VIEW
        rr.log("world/target/arrow", rr.Arrows3D(origins=[tip[i]], vectors=[world[i] - tip[i]], colors=color, radii=0.0022))
        rr.log("world/target/point", rr.Points3D([world[i]], colors=color, radii=0.006))
        rr.log("wrist/target", rr.Points2D([[t.u[i], t.v[i]]], colors=color, radii=9) if t.in_view[i] else rr.Clear(recursive=False))
        for c in "xyz":
            rr.log(f"offset/{c}", rr.Scalars(100 * t[c][i]))
        rr.log("offset/distance", rr.Scalars(100 * t.distance[i]))
        rr.log("subtask", rr.TextDocument(
            f"**{t.subtask[i]}**: {k} in {t.seconds_to_event[i]:.1f} s, {100 * t.distance[i]:.1f} cm away" + ("" if t.in_view[i] else ", not in the wrist view"), media_type="text/markdown"))
    print("wrote", f"{out}/episode_{e:03d}.rrd")
