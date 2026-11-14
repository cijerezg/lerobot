"""Bake the scan's vertex colours (work/nail/mesh_object.ply, TSDF-fused RGB) onto the clipper case visual mesh as a MuJoCo
texture: every face gets its own 2x2 texel block (three corners = the vertex colours, fourth = their mean) and per-face UVs,
so the renderer's interpolation reproduces Gouraud shading of the scan colours. The asset's visual OBJ is the largest cluster of
the scan shifted to its XY bbox centre (build_rigid_asset.py); the cluster is re-selected here by vertex count.
usage: uv run python scripts/case_texture.py work/nail/mesh_object.ply assets/clipper_case/clipper_case_visual.obj
writes <visual>_textured.obj and <visual>_colors.png beside the OBJ."""
import argparse, pathlib, numpy as np, open3d as o3d, trimesh, cv2
ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("obj"); a = ap.parse_args()
vis = trimesh.load(a.obj, process=False); m = o3d.io.read_triangle_mesh(a.ply); cl, cnt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl)
for k in np.argsort(-np.asarray(cnt)):
    sub = o3d.geometry.TriangleMesh(m); sub.remove_triangles_by_mask(cl != k); sub.remove_unreferenced_vertices()
    if len(sub.vertices) == len(vis.vertices): break
V = np.asarray(sub.vertices); F = np.asarray(sub.triangles); C = (np.asarray(sub.vertex_colors) * 255).astype(np.uint8)
shift = (V[:, :2].min(0) + V[:, :2].max(0)) / 2; assert np.abs(V[:, :2] - shift - vis.vertices[:, :2]).max() < 1e-4 and len(F) == len(vis.faces), "cluster does not match the OBJ"
n = len(F); side = int(np.ceil(np.sqrt(n))); tex = np.zeros((2 * side, 2 * side, 3), np.uint8); uv = np.zeros((3 * n, 2))
corners = np.array([[0, 0], [1, 0], [0, 1]])  # texel offsets inside the block for the three face corners
for i, f in enumerate(F):
    bx, by = (i % side) * 2, (i // side) * 2; cols = C[f]
    for c, (dx, dy) in enumerate(corners): tex[by + dy, bx + dx] = cols[c]; uv[3 * i + c] = [(bx + dx + 0.5) / (2 * side), 1 - (by + dy + 0.5) / (2 * side)]
    tex[by + 1, bx + 1] = cols.mean(0)
verts = vis.vertices[F.reshape(-1)]; faces = np.arange(3 * n).reshape(n, 3)
out_obj = pathlib.Path(a.obj).with_name(pathlib.Path(a.obj).stem + "_textured.obj"); out_png = pathlib.Path(a.obj).with_name(pathlib.Path(a.obj).stem + "_colors.png")
with open(out_obj, "w") as fh:
    fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts); fh.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
    fh.writelines(f"f {a0 + 1}/{a0 + 1} {b0 + 1}/{b0 + 1} {c0 + 1}/{c0 + 1}\n" for a0, b0, c0 in faces)
cv2.imwrite(str(out_png), tex[:, :, ::-1]); print(f"{n} faces -> {out_obj} + {out_png} ({tex.shape[1]}x{tex.shape[0]})")
