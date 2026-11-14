"""Pull every free body farther than --limit m (xy, from the base at the origin) radially in to that distance, then slide the
moved bodies along their arc away from overlapping neighbours (bounding circles from geom_rbound), on the table. Edits the body pos
in scene_nail.xml and scene_nail_photo.xml of each layout (backups *.pre_pull.xml) and objects.json xy.
usage: uv run python scripts/pull_objects_in.py LAYOUT_DIR [...] [--limit 0.50]"""
import argparse, json, pathlib, re, mujoco, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("layouts", nargs="+"); ap.add_argument("--limit", type=float, default=0.50, help="the real arm reached 0.66 m in the tidy-up episodes; 0.15 m margin for the bent top-pinch pose (user 2026-10-17)"); ap.add_argument("--gap", type=float, default=0.015); a = ap.parse_args()
for L in a.layouts:
    L = pathlib.Path(L); model = mujoco.MjModel.from_xml_path(str(L / "scene_nail.xml"))
    free = [b for b in range(model.nbody) if model.body_jntnum[b] > 0 and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]
    pos = {b: model.body_pos[b].copy() for b in free}
    rad = {b: max(np.linalg.norm(model.geom_pos[g][:2]) + model.geom_rbound[g] for g in range(model.ngeom) if model.geom_bodyid[b] == b or model.geom_bodyid[g] == b) for b in free}
    moved = [b for b in free if np.linalg.norm(pos[b][:2]) > a.limit]
    for b in moved: pos[b][:2] *= a.limit / np.linalg.norm(pos[b][:2])
    def rot(v, deg):
        c, s_ = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg)); return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])
    for _ in range(200):  # crowded: slide the moved body along its arc away from the neighbour (same distance from the base)
        for i in moved:
            for j in free:
                if i == j: continue
                d = pos[i][:2] - pos[j][:2]; n = np.linalg.norm(d); need = rad[i] + rad[j] + a.gap
                if n < need:
                    side = np.sign(np.cross(pos[j][:2], pos[i][:2])) or 1.0; pos[i][:2] = rot(pos[i][:2], 1.5 * side)
                    if j in moved: pos[j][:2] = rot(pos[j][:2], -1.5 * side)
            pos[i][1] = np.clip(pos[i][1], -0.44, 0.49); pos[i][0] = max(pos[i][0], 0.10)
    for b in moved: print(f"{L.name}: {model.body(b).name:26s} {model.body_pos[b][:2].round(3)} (d {np.linalg.norm(model.body_pos[b][:2]):.3f}) -> {pos[b][:2].round(3)} (d {np.linalg.norm(pos[b][:2]):.3f}), r {rad[b]:.3f}")
    for xml in ("scene_nail.xml", "scene_nail_photo.xml"):
        p = L / xml; s = p.read_text(); (L / xml.replace(".xml", ".pre_pull.xml")).write_text(s)
        for b in moved:
            name = model.body(b).name; pat = re.compile(rf'(<body name="{name}" pos=")[^"]*(")')
            assert pat.search(s), (xml, name); s = pat.sub(lambda m: f'{m.group(1)}{pos[b][0]:.4f} {pos[b][1]:.4f} {pos[b][2]:.4f}{m.group(2)}', s, count=1)
        p.write_text(s)
    oj = L / "objects.json"
    if oj.exists():
        o = json.loads(oj.read_text())
        for b in moved:
            name = model.body(b).name
            if name in o.get("objects", {}) and "xy" in o["objects"][name]: o["objects"][name]["xy"] = [float(pos[b][0]), float(pos[b][1])]
        oj.write_text(json.dumps(o, indent=1))
