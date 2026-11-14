"""Finger geometry in the gripper_end frame: slide axis, tip axis, pad gap vs slide, finger length, palm extent."""
import mujoco, numpy as np
from pathlib import Path
WS = Path(__file__).resolve().parents[4]
m = mujoco.MjModel.from_xml_path(str(WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml")); d = mujoco.MjData(m)
q0 = int(m.jnt_qposadr[m.joint("joint1").id]); ge = m.body("gripper_end").id
def pts(body):
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == m.body(body).id and (m.geom_contype[g] | m.geom_conaffinity[g]):
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(out)
def local(P): return (P - d.xpos[ge]) @ d.xmat[ge].reshape(3, 3)
mujoco.mj_forward(m, d)
for s in (0.0, 0.025, 0.05):
    d.qpos[q0 + 6:q0 + 8] = s; mujoco.mj_forward(m, d)
    L, R, E = local(pts("gripper_left")), local(pts("gripper_right")), local(pts("gripper_end"))
    print(f"slide={s}: left  x[{L[:,0].min():.4f},{L[:,0].max():.4f}] y[{L[:,1].min():.4f},{L[:,1].max():.4f}] z[{L[:,2].min():.4f},{L[:,2].max():.4f}]")
    print(f"           right x[{R[:,0].min():.4f},{R[:,0].max():.4f}] y[{R[:,1].min():.4f},{R[:,1].max():.4f}] z[{R[:,2].min():.4f},{R[:,2].max():.4f}]")
    print(f"           palm  x[{E[:,0].min():.4f},{E[:,0].max():.4f}] y[{E[:,1].min():.4f},{E[:,1].max():.4f}] z[{E[:,2].min():.4f},{E[:,2].max():.4f}]")
    print("           finger body pos (gripper_end frame): left", local(d.xpos[m.body('gripper_left').id][None])[0].round(4), "right", local(d.xpos[m.body('gripper_right').id][None])[0].round(4))
# which local axis is the tool/tip axis: the finger extent beyond the palm
d.qpos[q0 + 6:q0 + 8] = 0.0; mujoco.mj_forward(m, d)
L, R, E = local(pts("gripper_left")), local(pts("gripper_right")), local(pts("gripper_end"))
for ax, n in enumerate("xyz"):
    print(n, "palm", E[:, ax].min().round(4), E[:, ax].max().round(4), "fingers", min(L[:, ax].min(), R[:, ax].min()).round(4), max(L[:, ax].max(), R[:, ax].max()).round(4))
# pad geometry near the tips (last 15 mm along the tip axis, assume -x is the tip direction if fingers extend past palm in -x)
ext = {n: (min(L[:, a].min(), R[:, a].min()), max(L[:, a].max(), R[:, a].max())) for a, n in enumerate("xyz")}
print("extents", ext)
# gap between inner faces along the slide axis for the tip region, closed and open
for s in (0.0, 0.01, 0.02, 0.05):
    d.qpos[q0 + 6:q0 + 8] = s; mujoco.mj_forward(m, d)
    L, R = local(pts("gripper_left")), local(pts("gripper_right"))
    tip_ax = 0; tipL = L[L[:, tip_ax] < np.percentile(L[:, tip_ax], 25)]; tipR = R[R[:, tip_ax] < np.percentile(R[:, tip_ax], 25)]
    for a in (1, 2):
        gap = max(tipR[:, a].min() - tipL[:, a].max(), tipL[:, a].min() - tipR[:, a].max())
        print(f"slide={s} tip-region gap along axis {'xyz'[a]}: {gap:.4f}")
print("wrist camera pos in gripper_end frame: see xml (-0.0547, -0.0029, 0.0506)")
print("---- slices along the tip axis x (gripper_end frame), y-extent of each finger ----")
for s in (0.0, 0.02, 0.05):
    d.qpos[q0 + 6:q0 + 8] = s; mujoco.mj_forward(m, d)
    L, R = local(pts("gripper_left")), local(pts("gripper_right"))
    print(f"slide={s}")
    for lo in np.arange(-0.09, 0.0, 0.01):
        sl = L[(L[:, 0] >= lo) & (L[:, 0] < lo + 0.01)]; sr = R[(R[:, 0] >= lo) & (R[:, 0] < lo + 0.01)]
        if len(sl) and len(sr):
            print(f"  x[{lo:.2f},{lo+0.01:.2f}] left y[{sl[:,1].min():.4f},{sl[:,1].max():.4f}] z[{sl[:,2].min():.4f},{sl[:,2].max():.4f}] | right y[{sr[:,1].min():.4f},{sr[:,1].max():.4f}]  inner gap(right.min-left.max)={sr[:,1].min()-sl[:,1].max():.4f}")
