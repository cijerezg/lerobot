"""mesh_object.ply -> one rigid MuJoCo body (largest cluster standing on the sheet) + optional primitive boxes.
Body origin = footprint centre at table level. Positions in the XML = scene_pos + (table-frame xy - sheet centre).
usage: build_rigid_asset.py work_dir asset_dir name --scene_pos X Y [--mass_kg 0.05] [--hulls 32] [--threshold 0.03]
       [--box NAME L W H X Y YAW_DEG MASS]...   (metres, table-frame xy, repeatable)"""
import argparse, json, pathlib, numpy as np, trimesh, coacd, open3d as o3d
p = argparse.ArgumentParser(); p.add_argument("work"); p.add_argument("asset"); p.add_argument("name"); p.add_argument("--scene_pos", nargs=2, type=float, required=True)
p.add_argument("--mass_kg", type=float, default=0.05); p.add_argument("--hulls", type=int, default=32); p.add_argument("--threshold", type=float, default=0.03)
p.add_argument("--box", nargs=8, action="append", default=[]); a = p.parse_args()
W = pathlib.Path(a.work); A = pathlib.Path(a.asset); A.mkdir(parents=True, exist_ok=True); sh = np.load(W / "sheet.npz"); scale = float(sh["scale"])
box = sh["box"] * scale; sc = box.mean(0); e0 = box[1] - box[0]; e1 = box[3] - box[0]; L0, L1 = np.linalg.norm(e0), np.linalg.norm(e1); e0 /= L0; e1 /= L1
def in_sheet(xy): r = xy - sc; return (np.abs(r @ e0) < L0 / 2) & (np.abs(r @ e1) < L1 / 2)
m = o3d.io.read_triangle_mesh(str(W / "mesh_object.ply")); V = np.asarray(m.vertices); T = np.asarray(m.triangles)
cl, cnt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); cnt = np.asarray(cnt); pick = None
for k in np.argsort(-cnt):
    tri = T[cl == k]; cen = V[tri].mean(1); ok = cnt[k] > 1500 and in_sheet(cen[:, :2]).mean() > 0.9 and np.ptp(cen[:, :2], axis=0).max() < 0.25
    print("cluster %d tris %d on_sheet %.2f size %s -> %s" % (k, cnt[k], in_sheet(cen[:, :2]).mean(), (np.ptp(cen, axis=0) * 1000).round(0), "PICK" if ok and pick is None else "skip"))
    if ok and pick is None: pick = k
m.remove_triangles_by_mask(cl != pick); m.remove_unreferenced_vertices(); V = np.asarray(m.vertices)
origin = np.r_[(V[:, :2].min(0) + V[:, :2].max(0)) / 2, 0.0]; tm = trimesh.Trimesh(V - origin, np.asarray(m.triangles), process=False)
tm.export(A / f"{a.name}_visual.obj"); parts = coacd.run_coacd(coacd.Mesh(tm.vertices, tm.faces), threshold=a.threshold, max_convex_hull=a.hulls, preprocess_mode="auto", seed=0)
hulls = []
for i, (v, f) in enumerate(parts):
    h = trimesh.Trimesh(v, f, process=False).convex_hull; fn = f"{a.name}_hull{i:02d}.obj"; h.export(A / fn); hulls.append((fn, h.volume))
vol = sum(v for _, v in hulls); dens = a.mass_kg / vol; print("body size mm", (np.ptp(V, axis=0) * 1000).round(0), "hulls", len(hulls), "hull volume %.2e m3" % vol)
sp = np.array(a.scene_pos); pos = sp + (origin[:2] - sc)
L = ["<mujoco>", "  <asset>", f'    <mesh name="{a.name}_visual" file="{a.name}_visual.obj"/>'] + [f'    <mesh name="{fn[:-4]}" file="{fn}"/>' for fn, _ in hulls]
L += ["  </asset>", "  <default>", f'    <default class="{a.name}_col"><geom type="mesh" group="3" condim="4" friction="0.8 0.005 0.0001"/></default>',
      f'    <default class="{a.name}_vis"><geom type="mesh" group="2" contype="0" conaffinity="0" mass="0" rgba="0.55 0.55 0.58 1"/></default>', "  </default>", "  <worldbody>",
      f'    <body name="{a.name}" pos="{pos[0]:.4f} {pos[1]:.4f} 0.001">', "      <freejoint/>", f'      <geom class="{a.name}_vis" mesh="{a.name}_visual"/>']
L += [f'      <geom class="{a.name}_col" mesh="{fn[:-4]}" mass="{a.mass_kg * v / vol:.5f}"/>' for fn, v in hulls] + ["    </body>"]
for name, bl, bw, bh, bx, by, yaw, mass in a.box:
    bl, bw, bh, bx, by, yaw, mass = map(float, (bl, bw, bh, bx, by, yaw, mass)); q = np.array([np.cos(np.deg2rad(yaw) / 2), np.sin(np.deg2rad(yaw) / 2)]); bp = sp + (np.array([bx, by]) - sc)
    L += [f'    <body name="{name}" pos="{bp[0]:.4f} {bp[1]:.4f} {bh / 2 + 0.002:.4f}" quat="{q[0]:.5f} 0 0 {q[1]:.5f}">', "      <freejoint/>",
          f'      <geom type="box" size="{bl / 2:.4f} {bw / 2:.4f} {bh / 2:.4f}" mass="{mass}" rgba="0.75 0.75 0.78 1" condim="4" friction="0.6 0.005 0.0001"/>', "    </body>"]
L += ["  </worldbody>", "</mujoco>"]; (A / f"{a.name}.xml").write_text("\n".join(L) + "\n")
json.dump({"origin_table_m": origin.tolist(), "sheet_centre_m": sc.tolist(), "scene_pos": pos.tolist(), "hull_volume_m3": vol, "mass_kg": a.mass_kg, "size_m": np.ptp(V, axis=0).tolist()}, open(A / f"{a.name}_meta.json", "w"), indent=1)
print("wrote", A / f"{a.name}.xml", "case scene pos", pos.round(3))
