"""mesh_object.ply (table frame, metres, colours) -> rigid MuJoCo body: largest connected cluster, textured visual OBJ (per-face 2x2 texel
blocks from the vertex colours, case_texture.py approach), CoACD hulls, <name>.xml at scene pos 0 0, meta json.
usage: build_asset.py work_dir asset_dir name [--mass_kg 0.05] [--hulls 32] [--threshold 0.03] [--crop_mm X Y] (crop around the mesh centre)"""
import argparse, json, pathlib, numpy as np, trimesh, coacd, open3d as o3d, cv2
p = argparse.ArgumentParser(); p.add_argument("work"); p.add_argument("asset"); p.add_argument("name"); p.add_argument("--mass_kg", type=float, default=0.05)
p.add_argument("--hulls", type=int, default=32); p.add_argument("--threshold", type=float, default=0.03); p.add_argument("--crop_mm", nargs=2, type=float, default=None); a = p.parse_args()
W = pathlib.Path(a.work); A = pathlib.Path(a.asset); A.mkdir(parents=True, exist_ok=True); m = o3d.io.read_triangle_mesh(str(W / "mesh_object.ply"))
if a.crop_mm:
    V = np.asarray(m.vertices); c = np.median(V[:, :2], axis=0); keep = (np.abs(V[:, 0] - c[0]) < a.crop_mm[0] / 2000) & (np.abs(V[:, 1] - c[1]) < a.crop_mm[1] / 2000); m = m.select_by_index(np.nonzero(keep)[0])
cl, cnt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); cnt = np.asarray(cnt); k = int(np.argmax(cnt)); m.remove_triangles_by_mask(cl != k); m.remove_unreferenced_vertices()
V = np.asarray(m.vertices); F = np.asarray(m.triangles); C = (np.asarray(m.vertex_colors) * 255).astype(np.uint8); origin = np.r_[(V[:, :2].min(0) + V[:, :2].max(0)) / 2, 0.0]; V = V - origin
n = len(F); side = int(np.ceil(np.sqrt(n))); tex = np.zeros((2 * side, 2 * side, 3), np.uint8); uv = np.zeros((3 * n, 2)); corners = np.array([[0, 0], [1, 0], [0, 1]])
for i, f in enumerate(F):
    bx, by = (i % side) * 2, (i // side) * 2; cols = C[f]
    for c, (dx, dy) in enumerate(corners): tex[by + dy, bx + dx] = cols[c]; uv[3 * i + c] = [(bx + dx + 0.5) / (2 * side), 1 - (by + dy + 0.5) / (2 * side)]
    tex[by + 1, bx + 1] = cols.mean(0)
verts = V[F.reshape(-1)]; faces = np.arange(3 * n).reshape(n, 3)
with open(A / f"{a.name}_visual.obj", "w") as fh:
    fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts); fh.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
    fh.writelines(f"f {a0 + 1}/{a0 + 1} {b0 + 1}/{b0 + 1} {c0 + 1}/{c0 + 1}\n" for a0, b0, c0 in faces)
cv2.imwrite(str(A / f"{a.name}_colors.png"), tex[:, :, ::-1])
tm = trimesh.Trimesh(V, F, process=False); parts = coacd.run_coacd(coacd.Mesh(tm.vertices, tm.faces), threshold=a.threshold, max_convex_hull=a.hulls, preprocess_mode="auto", seed=0); hulls = []
for i, (v, f) in enumerate(parts):
    h = trimesh.Trimesh(v, f, process=False).convex_hull; fn = f"{a.name}_hull{i:02d}.obj"; h.export(A / fn); hulls.append((fn, h.volume))
vol = sum(v for _, v in hulls); size = np.ptp(V, axis=0)
L = ["<mujoco>", "  <asset>", f'    <texture name="{a.name}_tex" type="2d" file="{a.name}_colors.png"/>', f'    <material name="{a.name}_mat" texture="{a.name}_tex" specular="0.1" shininess="0.2"/>',
     f'    <mesh name="{a.name}_visual" file="{a.name}_visual.obj"/>'] + [f'    <mesh name="{fn[:-4]}" file="{fn}"/>' for fn, _ in hulls]
L += ["  </asset>", "  <default>", f'    <default class="{a.name}_col"><geom type="mesh" group="3" condim="4" friction="0.8 0.005 0.0001"/></default>',
      f'    <default class="{a.name}_vis"><geom type="mesh" group="2" contype="0" conaffinity="0" mass="0" material="{a.name}_mat"/></default>', "  </default>", "  <worldbody>",
      f'    <body name="{a.name}" pos="0 0 0.001">', "      <freejoint/>", f'      <geom class="{a.name}_vis" mesh="{a.name}_visual"/>']
L += [f'      <geom class="{a.name}_col" mesh="{fn[:-4]}" mass="{a.mass_kg * v / vol:.5f}"/>' for fn, v in hulls] + ["    </body>", "  </worldbody>", "</mujoco>"]
(A / f"{a.name}.xml").write_text("\n".join(L) + "\n")
meta = dict(origin_table_m=origin.tolist(), size_mm=(1e3 * size).round(1).tolist(), faces=int(n), hulls=len(hulls), hull_volume_m3=vol, mass_kg=a.mass_kg, texture_px=int(2 * side), mesh_cluster_tris=int(cnt[k]), clusters_in_mesh=int(len(cnt)))
json.dump(meta, open(A / f"{a.name}_meta.json", "w"), indent=1); print(json.dumps(meta))
