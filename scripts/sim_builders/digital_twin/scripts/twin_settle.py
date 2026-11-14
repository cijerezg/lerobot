"""Settle twin/scene_twin.xml for 1 s: per-free-body displacement / tilt, penetrations at the start, object-object contacts.
Run from the LeRobot root: MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/digital_twin/scripts/twin_settle.py"""
import sys, numpy as np, mujoco
import json, pathlib
args = [a for a in sys.argv[1:] if not a.startswith("--")]; write_rest = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--write-rest=")]
xml = args[0] if args else "lerobot/scripts/sim_builders/digital_twin/twin/scene_twin.xml"
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
free = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
def gname(g): return m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}/g{g}"
def bname(g): return m.body(m.geom_bodyid[g]).name
print("timestep", m.opt.timestep, "free bodies", [m.body(b).name for b in free])
print("start contacts with penetration > 1 mm:")
for i in range(d.ncon):
    c = d.contact[i]
    if c.dist < -1e-3: print(f"  {bname(c.geom1)}:{gname(c.geom1)} x {bname(c.geom2)}:{gname(c.geom2)} dist {c.dist*1e3:+.1f} mm")
print("start ncon", d.ncon)
x0 = {b: d.xpos[b].copy() for b in free}; q0 = {b: d.xquat[b].copy() for b in free}
pairs = {}; n = int(round(1.0 / m.opt.timestep))
for k in range(n):
    mujoco.mj_step(m, d)
    for i in range(d.ncon):
        c = d.contact[i]; a, b = bname(c.geom1), bname(c.geom2)
        if a != b and "world" not in (a, b): pairs[tuple(sorted((a, b)))] = pairs.get(tuple(sorted((a, b))), 0) + 1
print(f"after 1 s ({n} steps) ncon {d.ncon}")
print(f"{'body':16s} {'dxy mm':>7s} {'dz mm':>7s} {'tilt deg':>8s}  z0 -> z1 (m)")
for b in free:
    dp = d.xpos[b] - x0[b]; q = np.zeros(4); qi = np.zeros(4); mujoco.mju_negQuat(qi, q0[b]); mujoco.mju_mulQuat(q, d.xquat[b], qi); tilt = np.degrees(2 * np.arccos(min(1.0, abs(q[0]))))
    print(f"{m.body(b).name:16s} {np.linalg.norm(dp[:2])*1e3:7.1f} {dp[2]*1e3:+7.1f} {tilt:8.2f}  {x0[b][2]:.4f} -> {d.xpos[b][2]:.4f}")
print("body-body contact pairs over the second (robot/table excluded):", {k: v for k, v in pairs.items() if not any(s in ("table", "robot") for s in k)})
# --write-rest=NAME: store the settled pose of NAME in twin/layout.json "rest_overrides" (twin_build.py starts the body there)
if write_rest:
    lp = pathlib.Path(xml).parent / "layout.json"; lay = json.loads(lp.read_text()); ro = lay.setdefault("rest_overrides", {})
    for name in write_rest:
        b = m.body(name).id; ro[name] = {"pos": d.xpos[b].round(4).tolist(), "quat": d.xquat[b].round(5).tolist(), "note": "settled pose after 1 s, used as the start pose"}
    lp.write_text(json.dumps(lay, indent=1)); print("rest overrides written:", {k: ro[k]["pos"] for k in write_rest})
