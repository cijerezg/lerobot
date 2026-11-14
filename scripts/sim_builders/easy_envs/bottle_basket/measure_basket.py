"""Metric basket measurement from the 2026-10-21 teleop wrist depth (D405) + sim FK camera pose.
Frames: ep 0 frames 6000 and 6030 of outputs/rebot_teleop_2026-10-21-cut-v1 (basket fully in the wrist view).
Run from the workspace root: MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/easy_envs/bottle_basket/measure_basket.py
"""
import json, sys
from pathlib import Path
import cv2, mujoco, numpy as np, pandas as pd
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos, DEPTH_UNITS_M

WS = Path(__file__).resolve().parents[5]
DS = WS / "outputs/rebot_teleop_2026-10-21-cut-v1"
OUT = Path(__file__).resolve().parents[1] / "basket_reference"
SCENE = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/clippers_only/scene.xml"
FRAMES = [6030, 6060, 6000]

m = mujoco.MjModel.from_xml_path(str(SCENE)); d = mujoco.MjData(m)
q0 = int(m.jnt_qposadr[m.joint("joint1").id]); cam = m.camera("wrist").id
states = pd.read_parquet(DS / "data/chunk-000/file-000.parquet", columns=["frame_index", "observation.state"])
H, W = 480, 640; f = 394.9832; cx, cy = 322.5604, 238.6966  # factory z16 intrinsics (memory: D405, 2026-07-02)
report = {}
for fr in FRAMES:
    st = np.array(states.iloc[fr]["observation.state"]); assert int(states.iloc[fr]["frame_index"]) == fr
    d.qpos[q0:q0 + 8] = to_qpos(st); mujoco.mj_forward(m, d)
    Rc = d.cam_xmat[cam].reshape(3, 3); t = d.cam_xpos[cam]
    depth = cv2.imread(str(DS / f"depth/wrist.depth/episode-000000/frame-{fr:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) * DEPTH_UNITS_M
    assert depth.shape == (H, W), depth.shape
    v, u = np.mgrid[0:H, 0:W]; Z = depth; ok = Z > 0.05
    pc = np.stack([(u - cx) / f * Z, -(v - cy) / f * Z, -Z], -1)  # MuJoCo camera: looks down -z, y up
    world = pc @ Rc.T + t
    # table plane from the pixels below the basket (bottom-centre of the image, between the finger silhouettes)
    tbl = ok & (v > 380) & (v < 430) & (u > 220) & (u < 420)
    z_table = float(np.median(world[tbl, 2]))  # FK table-height check only (frame 6000 looks at the rim there)
    basket = ok & (world[..., 2] > 0.03) & (world[..., 2] < 0.25) & (v < 440)
    rim_band = basket & (world[..., 2] > 0.17)
    xy = world[rim_band][:, :2]
    # principal axes of the rim ring
    c = xy.mean(0); evec = np.linalg.eigh(np.cov((xy - c).T))[1][:, ::-1]; p = (xy - c) @ evec
    L = np.percentile(p[:, 0], 99) - np.percentile(p[:, 0], 1); Wd = np.percentile(p[:, 1], 99) - np.percentile(p[:, 1], 1)
    yaw = np.degrees(np.arctan2(evec[1, 0], evec[0, 0]))
    z_rim = np.percentile(world[basket, 2], 97)
    report[fr] = dict(table_z_fk=float(z_table), cam_pos=t.tolist(), rim_height_m=float(z_rim), rim_outer_length_m=float(L), rim_outer_width_m=float(Wd),
                      rim_centre_xy=c.tolist(), long_axis_yaw_deg=float(yaw), n_rim_pts=int(rim_band.sum()))
    print(fr, json.dumps(report[fr]))
    # top-down scatter for a sanity look
    img = np.full((600, 600, 3), 255, np.uint8)
    sel = ok & (v < 400) & (world[..., 2] > -0.02) & (world[..., 2] < 0.3)
    for (x, y, z) in world[sel][::7]:
        px = int(300 + (y - c[1]) * 1000); py = int(300 - (x - c[0]) * 1000)
        if 0 <= px < 600 and 0 <= py < 600: img[py, px] = (int(255 * min(z / 0.15, 1)), 60, int(255 * (1 - min(z / 0.15, 1))))
    cv2.circle(img, (300, 300), 3, (0, 180, 0), -1); cv2.putText(img, f"frame {fr}: {L:.3f} x {Wd:.3f} m, rim z {z_rim:.3f}; 1 px = 1 mm, +x up", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
    cv2.imwrite(str(OUT / f"depth_topdown_{fr}.png"), img)
    # side profile (z vs principal axis) to see wall height / flare
    prof = np.full((300, 600, 3), 255, np.uint8); pp = (world[basket][:, :2] - c) @ evec
    for (a, b), z in zip(pp[::5], world[basket][::5, 2]):
        px = int(300 + a * 1000); py = int(280 - z * 1000)
        if 0 <= px < 600 and 0 <= py < 300: prof[py, px] = (120, 0, 0)
    cv2.imwrite(str(OUT / f"depth_profile_{fr}.png"), prof)
(OUT / "basket_depth_measurement.json").write_text(json.dumps(report, indent=1))
