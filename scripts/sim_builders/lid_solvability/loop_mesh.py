"""Shape of the loop handle mesh (hinged_nail_ep2_21) in the headphone_case body frame; proposal for a collision box set."""
import mujoco, numpy as np, cv2
from pathlib import Path
WS = Path(__file__).resolve().parents[4]; OUT = Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml")); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
case = m.body("headphone_case").id
g = m.geom("hinged_nail_ep2_21").id; mid = m.geom_dataid[g]
v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]          # world
loc = (w - d.xpos[case]) @ d.xmat[case].reshape(3, 3)             # case body frame
print("verts", len(v), "world extents", w.min(0).round(4), w.max(0).round(4))
print("case-local extents", loc.min(0).round(4), loc.max(0).round(4))
# the case geoms use quat_z(ANGLE) within the body: the 'canonical' frame of the builder. Express in that frame too.
gb = m.geom("headphone_case_block_a").id
Rc = d.geom_xmat[gb].reshape(3, 3); oc = d.geom_xpos[gb]
can = (w - oc) @ Rc
print("canonical (block_a frame) extents", can.min(0).round(4), can.max(0).round(4))
# histogram of the loop in the canonical xy plane (1 mm cells) and xz
for a, b, n in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
    lo = can[:, [a, b]].min(0); hi = can[:, [a, b]].max(0)
    H, xe, ye = np.histogram2d(can[:, a], can[:, b], bins=[np.arange(lo[0], hi[0] + 0.002, 0.002), np.arange(lo[1], hi[1] + 0.002, 0.002)])
    print(f"--- {n} occupancy (2 mm cells; rows = {'xyz'[a]} from {lo[0]:.3f}, cols = {'xyz'[b]} from {lo[1]:.3f}) ---")
    for i in range(H.shape[0]):
        print(f"{xe[i]:+.3f} " + "".join("#" if H[i, j] > 0 else "." for j in range(H.shape[1])))
# faces: how many, to see if it is a closed band
print("faces", m.mesh_facenum[mid])
# close-up render of the loop from above and from the side with collision geoms
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = w.mean(0); cam.distance = 0.18
opt = mujoco.MjvOption()
ims = []
for az, el in ((180, -89), (180, -20), (90, -20)):
    cam.azimuth, cam.elevation = az, el
    for grp in ({2}, {2, 3}):
        for k in range(6): opt.geomgroup[k] = 1 if k in grp else 0
        r.update_scene(d, cam, opt); ims.append(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT / "frames/loop_closeup.jpg"), np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, 6, 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
