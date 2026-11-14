"""Photo textures for the static clutter primitives of the nail scene (bottles, pouch, headphone case, bits box, mic knob): the real
top-camera frame with the robot parked is projected onto each primitive. The primitives were placed by back-projecting this very
image, so the alignment is exact; the top camera gives 2 mm per pixel, about the wrist camera's resolution on these objects, which
it only sees from afar. Each primitive becomes a mesh (cylinder / box / ellipsoid) in WORLD coordinates with a top-down planar
atlas (texel = world x, y over the footprint; 3D point = first hit from above), texture in top-camera colours (the wrist camera
config applies its colour gain). Writes OUT/<geom>_photo.obj + .png; photo_scene.py --object_tex adds them (primitive -> group 3).
usage: uv run python scripts/clutter_texture.py SCENE_XML TOP_FRAME OUT_DIR [--res 1000]"""
import argparse, pathlib, cv2, mujoco, numpy as np, open3d as o3d, trimesh
from scipy import ndimage
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("top_frame"); ap.add_argument("out"); ap.add_argument("--res", type=int, default=1000); a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); H, W = 480, 640
model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); mujoco.mj_forward(model, data); top = cv2.imread(a.top_frame)
cid = model.camera("ext").id; Rc = data.cam_xmat[cid].reshape(3, 3); pc = data.cam_xpos[cid]; fpx = (H / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
SKIP = {"table", "floor", "table_top"}
for g in range(model.ngeom):
    name = model.geom(g).name
    if model.geom_bodyid[g] != 0 or not name or name in SKIP: continue
    size = model.geom_size[g]; t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER: mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=64)
    elif t == mujoco.mjtGeom.mjGEOM_BOX: mesh = trimesh.creation.box(extents=2 * size)
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID: mesh = trimesh.creation.icosphere(subdivisions=3); mesh.apply_scale(size)
    else: print("skip", name, t); continue
    T = np.eye(4); T[:3, :3] = data.geom_xmat[g].reshape(3, 3); T[:3, 3] = data.geom_xpos[g]; mesh.apply_transform(T)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces); res = a.res
    x0, y0 = V[:, :2].min(0) - 2 / res; x1, y1 = V[:, :2].max(0) + 2 / res; nx, ny = int(np.ceil((x1 - x0) * res)), int(np.ceil((y1 - y0) * res))
    xs = x0 + (np.arange(nx) + 0.5) / res; ys = y1 - (np.arange(ny) + 0.5) / res; X, Y = np.meshgrid(xs, ys)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V, o3d.core.float32), o3d.core.Tensor(F, o3d.core.int32)))
    origins = np.stack([X, Y, np.full_like(X, V[:, 2].max() + 0.01)], -1).reshape(-1, 3); rays = np.concatenate([origins, np.tile([0, 0, -1.0], (len(origins), 1))], 1)
    hit = scene.cast_rays(o3d.core.Tensor(rays, o3d.core.float32)); th = hit["t_hit"].numpy(); ok = np.isfinite(th)
    P = origins + np.where(ok, th, 0)[:, None] * np.array([0, 0, -1.0]); c = (P - pc) @ Rc; zc = -c[:, 2]
    U = (W / 2 + fpx * c[:, 0] / zc).reshape(ny, nx).astype(np.float32); Vv = (H / 2 - fpx * c[:, 1] / zc).reshape(ny, nx).astype(np.float32)
    tex = cv2.remap(top, U, Vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); okm = ok.reshape(ny, nx)
    _, inds = ndimage.distance_transform_edt(~okm, return_indices=True); tex = tex[inds[0], inds[1]]  # outside the silhouette: nearest hit texel
    uv = np.stack([(V[F.reshape(-1), 0] - x0) * res / nx, 1 - (y1 - V[F.reshape(-1), 1]) * res / ny], 1)
    verts = V[F.reshape(-1)]; faces = np.arange(3 * len(F)).reshape(-1, 3); obj = out / f"{name}_photo.obj"
    with open(obj, "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts); fh.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
        fh.writelines(f"f {a0 + 1}/{a0 + 1} {b0 + 1}/{b0 + 1} {c0 + 1}/{c0 + 1}\n" for a0, b0, c0 in faces)
    cv2.imwrite(str(out / f"{name}_photo.png"), tex)
    print(f"{name}: {len(F)} faces, texture {nx}x{ny} ({1e3 / res:.0f} mm/texel), {okm.mean() * 100:.0f}% of the footprint hit, mean BGR {tex[okm].mean(0).round(0)} -> {obj}")
