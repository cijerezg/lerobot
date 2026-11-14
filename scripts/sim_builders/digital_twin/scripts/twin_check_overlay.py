"""Project the twin's bodies (footprint AABB corners of their collision geoms, z = body bottom) through the CALIBRATED camera
plus the measured pixel offset and draw them on the real top frame: a placement check that does not depend on the aligned render camera.
Run from the LeRobot root: MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/digital_twin/scripts/twin_check_overlay.py"""
import json, pathlib, numpy as np, cv2, mujoco
ROOT = pathlib.Path("lerobot/scripts/sim_builders/digital_twin"); lay = json.loads((ROOT / "twin/layout.json").read_text()); off = np.array(lay["pixel_offset_real_minus_calibrated"])
mc = mujoco.MjModel.from_xml_path("lerobot/src/lerobot/robots/rebot_b601_sim/scenes/clippers_only/scene.xml"); dc = mujoco.MjData(mc); mujoco.mj_forward(mc, dc)
cid = mc.camera("ext").id; f = 0.5 * 480 / np.tan(np.deg2rad(mc.cam_fovy[cid]) / 2); R = dc.cam_xmat[cid].reshape(3, 3); t = dc.cam_xpos[cid]
def proj(p):
    pc = R.T @ (np.asarray(p) - t); return (np.array([320 + f * pc[0] / -pc[2], 240 - f * pc[1] / -pc[2]]) + off)
m = mujoco.MjModel.from_xml_path(str(ROOT / "twin/scene_twin.xml")); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
im = cv2.imread(str(ROOT / "room/renders/real_top_f0.png"))
for name in lay["poses"]:
    b = m.body(name).id; ids = [b] + [c for c in range(m.nbody) if m.body_parentid[c] == b]
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ids and m.geom_contype[g] | m.geom_conaffinity[g]:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]; V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                s = m.geom_size[g]; V = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
            pts.append((d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g])
    P = np.vstack(pts); z = P[:, 2].min(); hull = cv2.convexHull(P[:, :2].astype(np.float32)).reshape(-1, 2)
    poly = np.array([proj([x, y, z]) for x, y in hull]).astype(np.int32)
    cv2.polylines(im, [poly], True, (0, 255, 0), 1); c = proj([*d.xpos[b][:2], z]).astype(int)
    cv2.putText(im, name, (c[0] - 30, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    print(f"{name:15s} footprint px centre {c}  table xy {d.xpos[b][:2].round(3)}  footprint bottom z {z*1e3:+.1f} mm")
cv2.imwrite(str(ROOT / "twin/renders/placement_check_real_top_f0.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92]); print("wrote placement_check_real_top_f0.jpg")
