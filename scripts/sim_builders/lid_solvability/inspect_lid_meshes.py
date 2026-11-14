"""Per-part AABB of the lid meshes (22-31) + base/hinge parts in the lid body frame and in the world at the open pose; find the loop handle."""
import sys
import mujoco, numpy as np
from pathlib import Path
WS = Path(__file__).resolve().parents[4]
SCENE = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml"
m = mujoco.MjModel.from_xml_path(str(SCENE)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
lid = m.body("headphone_lid").id; case = m.body("headphone_case").id
Rl = d.xmat[lid].reshape(3, 3); ol = d.xpos[lid]
print("lid body world pos", ol.round(4), "xipos (COM)", d.xipos[lid].round(4))
for g in range(m.ngeom):
    b = m.geom_bodyid[g]
    if b not in (lid, case): continue
    name = m.geom(g).name
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    else:
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        local = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        w = local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    loc = (w - ol) @ Rl  # lid frame
    print(f"{name:28s} body={'lid ' if b==lid else 'case'} n={len(w):6d} world x[{w[:,0].min():.4f},{w[:,0].max():.4f}] y[{w[:,1].min():.4f},{w[:,1].max():.4f}] z[{w[:,2].min():.4f},{w[:,2].max():.4f}]  lidframe x[{loc[:,0].min():.4f},{loc[:,0].max():.4f}] y[{loc[:,1].min():.4f},{loc[:,1].max():.4f}] z[{loc[:,2].min():.4f},{loc[:,2].max():.4f}]")
