"""mj_step 1 s on a scene: free-body displacement, contacts involving room_* or robot geoms. usage: sim_check.py SCENE_XML"""
import sys, numpy as np, mujoco
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]; names = [m.body(m.jnt_bodyid[j]).id for j in free]
x0 = {b: d.xpos[b].copy() for b in names}
room = {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("room_")}
robot_bodies = set(); b = m.body("base_link").id if "base_link" in [m.body(i).name for i in range(m.nbody)] else None
print("timestep", m.opt.timestep, "ngeom", m.ngeom, "room geoms", len(room), "free bodies", [m.body(b).name for b in names])
print("room geoms contype/conaffinity:", sorted({(int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in room}), "mass sum", float(sum(m.geom(g).id >= 0 for g in room)) and float(m.body_mass[0]))
pairs = {}; n = int(round(1.0 / m.opt.timestep))
for k in range(n):
    mujoco.mj_step(m, d)
    for i in range(d.ncon):
        c = d.contact[i]; key = (m.geom(c.geom1).name or f"g{c.geom1}", m.geom(c.geom2).name or f"g{c.geom2}"); pairs[key] = pairs.get(key, 0) + 1
    if k == 0 or k == n - 1:
        print(f"step {k}: ncon {d.ncon}")
print("after 1 s: displacement mm per free body:", {m.body(b).name: float(np.linalg.norm(d.xpos[b] - x0[b]) * 1e3).__round__(1) for b in names})
room_pairs = {k: v for k, v in pairs.items() if any(s.startswith("room_") for s in k)}
print("contact pairs involving room_* geoms:", room_pairs)
rob = {k: v for k, v in pairs.items() if not any(s.startswith("room_") for s in k) and not all(("clipper" in s or "table_contact" in s or s.startswith("g")) for s in k)}
print("other contact pairs (not clipper/case/table):", rob)
print("all pairs (geom name, name): count over", n, "steps:"); [print("  ", k, v) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]]
