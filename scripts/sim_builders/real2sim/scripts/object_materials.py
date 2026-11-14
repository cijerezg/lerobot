"""Object appearance from first principles: one flat colour per material, hard boundaries from the GEOMETRY (not from projected
pixels), textures authored in the WRIST camera's colour space (measure_materials.py; the top camera gets an affine map
wrist -> top on its render, fitted on the materials it renders: config color_gain/offset).

Case (scan mesh, body frame): planar atlas (texel = x, y of the footprint, 0.5 mm; 3D point = first hit from above) with a
class per texel from the height map and the scan's own brightness: rim = within --rim_mm of the footprint edge, slot = below
--slot_z mm, black end = the rows whose scan brightness is dark (threshold at the midpoint of the two row-brightness modes),
foam = the rest; wall faces (nz < 0.3) and the new skirt use the rim colour through a constant patch. The mesh is shifted up by
--case_dz (the scan lost the bottom: real foam 17-19 mm, rim 20-21 mm above the table (depth sidecar); mesh foam 15-16.5, rim
17-18.5) and closed by a skirt extruded from the footprint outline down to z = 0 (also written as a convex collision hull, so the
case rests on the table instead of settling 5 mm). Two visual meshes: clipper_case (foam/slot/black, matte) and
clipper_case_metal (rim, walls, skirt; specular in photo_scene.py).
Clippers: textured box, top face = body colour with a dark head over the last --head_frac of the length at the end that the real
top frame shows dark (large: +u image side, small: +v; the sign of the body x axis is read off the scene's ext camera); sides =
body colour. Clutter: the primitive as a mesh in its own frame with a uniform texture (the real top colour mapped to wrist
space through the inverse of the top affine). Also writes OUT/case_atlas_review.png and OUT/materials.json (the colours used).
usage: MUJOCO_GL=egl uv run python scripts/object_materials.py SCENE_XML OUT_DIR [--case_dz 0.0025]"""
import argparse, json, pathlib, cv2, mujoco, numpy as np, open3d as o3d, trimesh
from scipy import ndimage
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("--case_dz", type=float, default=0.0025)
ap.add_argument("--rim_mm", type=float, default=3.5); ap.add_argument("--slot_z", type=float, default=13.0, help="mm, below = slot (mesh frame, before the shift)")
ap.add_argument("--res", type=int, default=2000, help="texels per metre"); ap.add_argument("--head_frac", type=float, default=0.25)
ap.add_argument("--case_obj", default=str(pathlib.Path(__file__).resolve().parents[1] / "assets/clipper_case/clipper_case_visual.obj"))
ap.add_argument("--ply", default=str(pathlib.Path(__file__).resolve().parents[1] / "work/nail/mesh_object.ply"))
ap.add_argument("--top_frame", required=True, help="real top frame with the robot parked and every object at the scene pose: clutter colours = median inside the eroded sim silhouette of each free body (ext camera)"); a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

# ---- colours, BGR 0-255, WRIST camera space (measure_materials.py on 10-16 ep 5; one exposure: frame 334 unless noted) ----
WRIST = dict(
    foam=(193, 189, 198),        # f334 window std 4; f1002 lit strip (190, 190, 221) agrees within the frame's exposure
    black_end=(54, 63, 63),      # f334 clean window (std 9)
    rim=(165, 170, 190),         # f1002 left rim strip (169, 169, 194) seen lit: brushed steel reflecting the ceiling
    clipper_body=(120, 126, 135),  # brushed steel reflects the ceiling: real wrist pooled over lit frames (138, 144, 157) (object_color_check), top
                                   # camera sees it darker (120, 123, 126 -> 79, 90, 97 in wrist space): view-dependent, compromise toward the wrist
    clipper_head=(62, 70, 76),     # the pivot end: f334 (51, 67, 79), f2005 (51, 64, 79)
)
SLOT_FACTOR = 0.8                # slot floor = foam shaded by its walls: real wrist 0.65, real top 0.9 of the foam
# top affine (wrist -> top, per channel, BGR) through the foam and the black end measured in both cameras:
TOP = dict(foam=(253, 250, 250), black_end=(91, 89, 85))
A = np.array([(TOP["foam"][c] - TOP["black_end"][c]) / (WRIST["foam"][c] - WRIST["black_end"][c]) for c in range(3)])
B = np.array([TOP["black_end"][c] - A[c] * WRIST["black_end"][c] for c in range(3)])
def to_wrist(top_bgr): return np.clip((np.asarray(top_bgr, float) - B) / A, 4, 255)
TASK_BODIES = ("clipper_case", "clipper_large", "clipper_small")
print(f"top affine wrist->top BGR: gain {A.round(3)} offset {B.round(1)}  (yaml RGB: color_gain {A[::-1].round(3).tolist()} color_offset {B[::-1].round(1).tolist()})")

H, W = 480, 640; model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
# clutter: every free body except the task objects; colour = median of the real top frame inside its eroded sim silhouette (ext camera)
rend = mujoco.Renderer(model, H, W); rend.enable_segmentation_rendering(); rend.update_scene(data, "ext"); seg = cv2.medianBlur(rend.render()[:, :, 0].astype(np.float32), 3).astype(np.int64); rend.disable_segmentation_rendering()  # 1 px edge ids
top_img = cv2.imread(a.top_frame); CLUTTER_TOP = {}
for g in range(model.ngeom):
    b = model.geom_bodyid[g]; name = model.geom(g).name
    if b == 0 or model.body(b).name in TASK_BODIES or model.body_jntnum[b] == 0 or not name: continue
    m_ = cv2.erode((seg == g).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if m_.sum() < 20: m_ = seg == g
    CLUTTER_TOP[name] = tuple(np.median(top_img[m_].reshape(-1, 3), 0).round(0).tolist())


def write_obj(path, V, F, uv):
    verts = V[F.reshape(-1)]; faces = np.arange(3 * len(F)).reshape(-1, 3)
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts); fh.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
        fh.writelines(f"f {a0 + 1}/{a0 + 1} {b0 + 1}/{b0 + 1} {c0 + 1}/{c0 + 1}\n" for a0, b0, c0 in faces)


def planar(V, F, res, pad_rows=16):
    """Top-down planar atlas over the xy bbox: z (first hit from above, nan outside), face id, texel centres, and a constant patch
    region (the last pad_rows rows) for faces that get one colour. Returns dict."""
    x0, y0 = V[:, :2].min(0) - 1 / res; x1, y1 = V[:, :2].max(0) + 1 / res; nx, ny = int(np.ceil((x1 - x0) * res)), int(np.ceil((y1 - y0) * res))
    S_w, S_h = nx, ny + pad_rows
    xs = x0 + (np.arange(nx) + 0.5) / res; ys = y1 - (np.arange(ny) + 0.5) / res; X, Y = np.meshgrid(xs, ys)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V, o3d.core.float32), o3d.core.Tensor(F, o3d.core.int32)))
    org = np.stack([X, Y, np.full_like(X, V[:, 2].max() + 0.01)], -1).reshape(-1, 3); rays = np.concatenate([org, np.tile([0, 0, -1.0], (len(org), 1))], 1)
    hit = sc.cast_rays(o3d.core.Tensor(rays, o3d.core.float32)); t = hit["t_hit"].numpy(); fid = hit["primitive_ids"].numpy().astype(np.int64); ok = np.isfinite(t)
    z = np.where(ok, org[:, 2] - t, np.nan).reshape(ny, nx); fid = np.where(ok, fid, -1).reshape(ny, nx)
    def uv_of(P):  # 3D points (n, 3) -> uv in the planar region
        return np.stack([(P[:, 0] - x0) * res / S_w, 1 - (y1 - P[:, 1]) * res / S_h], 1)
    patch_uv = np.array([0.5, 0.5 * pad_rows / S_h])  # centre of the constant patch (bottom rows)
    return dict(z=z, fid=fid, nx=nx, ny=ny, S_w=S_w, S_h=S_h, uv_of=uv_of, patch_uv=patch_uv, pad_rows=pad_rows)


def fill_nearest(tex, valid):
    _, inds = ndimage.distance_transform_edt(~valid, return_indices=True); return tex[inds[0], inds[1]]


# ------------------------------------------------------------------ case
tm = trimesh.load(a.case_obj, process=False); V = np.asarray(tm.vertices).copy(); F = np.asarray(tm.faces)
m = o3d.io.read_triangle_mesh(a.ply); cl, cnt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl)
for k in np.argsort(-np.asarray(cnt)):
    sub = o3d.geometry.TriangleMesh(m); sub.remove_triangles_by_mask(cl != k); sub.remove_unreferenced_vertices()
    if len(sub.vertices) == len(V): break
C = (np.asarray(sub.vertex_colors) * 255)[:, ::-1]
P = planar(V, F, a.res); z, fid, nx, ny = P["z"], P["fid"], P["nx"], P["ny"]; ok = fid >= 0
# classes: 0 foam, 1 slot, 2 rim, 3 black end
dist_mm = ndimage.distance_transform_edt(ok) * 1e3 / a.res
rim = ok & (dist_mm <= a.rim_mm); slot = ok & ~rim & (z * 1e3 < a.slot_z)
k = np.ones((5, 5), np.uint8); slot = cv2.morphologyEx(cv2.morphologyEx(slot.astype(np.uint8), cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k) > 0  # 1 mm clean-up of the scan's ragged slot contour
bright = C[F[np.clip(fid, 0, None)]].mean(-1).mean(-1)  # per texel: mean scan brightness of its face (ny, nx)
rows = np.array([np.median(bright[j][ok[j] & ~rim[j] & ~slot[j]]) if (ok[j] & ~rim[j] & ~slot[j]).sum() > 10 else np.nan for j in range(ny)])
lo, hi = np.nanpercentile(rows, 10), np.nanpercentile(rows, 90); mid = (lo + hi) / 2; dark_rows = rows < mid
# the black end is one contiguous block of rows at one end
first_dark = np.flatnonzero(dark_rows & ~np.isnan(rows)); top_end = first_dark.mean() < ny / 2 if len(first_dark) else True
edge = (first_dark.max() + 1) if top_end else first_dark.min(); black = np.zeros_like(ok); black[:edge] = True if top_end else False
if not top_end: black[edge:] = True
black &= ok & ~rim & ~slot
foam = ok & ~rim & ~slot & ~black
print(f"case atlas {nx}x{ny} texels at {1e3 / a.res:.2f} mm: foam {foam.sum()}, slot {slot.sum()}, rim {rim.sum()}, black end {black.sum()} "
      f"(row brightness modes {lo:.0f}/{hi:.0f}, boundary row {edge} of {ny}, black at the {'+y' if top_end else '-y'} end)")
tex = np.zeros((P["S_h"], P["S_w"], 3), np.float32)
tex[:ny, :nx][foam] = WRIST["foam"]; tex[:ny, :nx][slot] = np.array(WRIST["foam"]) * SLOT_FACTOR; tex[:ny, :nx][rim] = WRIST["rim"]; tex[:ny, :nx][black] = WRIST["black_end"]
# slots inside the black end keep the slot colour (the slots run into it); outside the footprint: nearest colour (mip safety)
tex[:ny, :nx] = fill_nearest(tex[:ny, :nx], ok); tex[ny:] = WRIST["rim"]
# review image of the classes
cls = np.zeros((ny, nx, 3), np.uint8); cls[foam] = (230, 230, 230); cls[slot] = (120, 200, 120); cls[rim] = (200, 120, 60); cls[black] = (40, 40, 40)
cv2.imwrite(str(out / "case_atlas_review.png"), np.hstack([cv2.resize(cls, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), cv2.resize(tex[:ny, :nx].astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)]))
# skirt: footprint outline (convex hull of the hit mask) extruded from z = 0 to the shifted mesh bottom (+0.5 mm overlap)
cnts, _ = cv2.findContours(ok.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); hull = cv2.convexHull(max(cnts, key=cv2.contourArea))[:, 0, :]
x0 = V[:, 0].min() - 1 / a.res; y1 = V[:, 1].max() + 1 / a.res
poly = np.stack([x0 + (hull[:, 0] + 0.5) / a.res, y1 - (hull[:, 1] + 0.5) / a.res], 1)
V[:, 2] += a.case_dz; z_bottom = V[:, 2].min() + 0.0005
n_p = len(poly); Vs = np.concatenate([np.c_[poly, np.zeros(n_p)], np.c_[poly, np.full(n_p, z_bottom)]])  # convex outline: fan caps + side quads
Fs = [[i, (i + 1) % n_p, n_p + i] for i in range(n_p)] + [[(i + 1) % n_p, n_p + (i + 1) % n_p, n_p + i] for i in range(n_p)]
Fs += [[0, i + 1, i] for i in range(1, n_p - 1)] + [[n_p, n_p + i, n_p + i + 1] for i in range(1, n_p - 1)]; Fs = np.array(Fs)
skirt = trimesh.Trimesh(Vs, Fs, process=False); skirt.fix_normals(); Fs = np.asarray(skirt.faces); skirt.export(out / "clipper_case_skirt.obj")  # convex: collision hull
# split faces: metal = walls (nz < 0.3) + top faces whose centroid texel is rim; matte = the rest
fn = trimesh.Trimesh(V, F, process=False).face_normals; cen = V[F].mean(1)
ci = np.clip(((cen[:, 0] - x0) * a.res).astype(int), 0, nx - 1); cj = np.clip(((y1 - cen[:, 1]) * a.res).astype(int), 0, ny - 1)
wall = fn[:, 2] < 0.3; metal = wall | rim[cj, ci]
uv = np.zeros((3 * len(F), 2)); top_uv = P["uv_of"](V[F.reshape(-1)]); uv[:] = top_uv
uv[np.repeat(wall, 3)] = P["patch_uv"]
write_obj(out / "clipper_case_photo.obj", V, F[~metal], uv.reshape(-1, 3, 2)[~metal].reshape(-1, 2))
Vm = np.concatenate([V, Vs]); Fm = np.concatenate([F[metal], Fs + len(V)]); uvm = np.concatenate([uv.reshape(-1, 3, 2)[metal].reshape(-1, 2), np.tile(P["patch_uv"], (3 * len(Fs), 1))])
write_obj(out / "clipper_case_metal_photo.obj", Vm, Fm, uvm)
cv2.imwrite(str(out / "clipper_case_photo.png"), tex.astype(np.uint8)); cv2.imwrite(str(out / "clipper_case_metal_photo.png"), tex.astype(np.uint8))
print(f"case: {(~metal).sum()} matte faces, {metal.sum()} metal faces + skirt {len(Fs)} faces (z 0..{z_bottom * 1e3:.1f} mm), mesh shifted {a.case_dz * 1e3:.1f} mm, top now {V[:, 2].max() * 1e3:.1f} mm")

# ------------------------------------------------------------------ clippers
cid = model.camera("ext").id; Rc = data.cam_xmat[cid].reshape(3, 3); pc = data.cam_xpos[cid]; fpx = (H / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
def project_ext(Pw):
    c = (Pw - pc) @ Rc; zc = -c[:, 2]; return np.stack([W / 2 + fpx * c[:, 0] / zc, H / 2 - fpx * c[:, 1] / zc], 1)
HEAD_IMAGE_DIR = dict(clipper_large=np.array([1.0, 0.0]), clipper_small=np.array([0.0, 1.0]))  # real top frame 0: the dark pivot end
for name in ("clipper_large", "clipper_small"):
    b = model.body(name).id; g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == b][0]; size = model.geom_size[g].copy()
    Rb = data.xmat[b].reshape(3, 3); pb = data.xpos[b]
    d_img = np.diff(project_ext(np.stack([pb, pb + 0.03 * Rb[:, 0]])), axis=0)[0]; head_sign = 1.0 if d_img @ HEAD_IMAGE_DIR[name] > 0 else -1.0
    box = trimesh.creation.box(extents=2 * size); Vb, Fb = np.asarray(box.vertices), np.asarray(box.faces); fnb = box.face_normals
    nxb, nyb = int(round(2 * size[0] * a.res)), int(round(2 * size[1] * a.res)); pad = 8
    tb = np.zeros((nyb + pad, nxb, 3), np.float32); tb[:] = WRIST["clipper_body"]
    nh = int(round(a.head_frac * nxb)); tb[:nyb, -nh:] = WRIST["clipper_head"] if head_sign > 0 else tb[:nyb, -nh:]
    if head_sign < 0: tb[:nyb, :nh] = WRIST["clipper_head"]
    top = fnb[:, 2] > 0.9; uvb = np.zeros((3 * len(Fb), 2))
    Pv = Vb[Fb.reshape(-1)]; uvb[:] = np.stack([(Pv[:, 0] + size[0]) / (2 * size[0]), 1 - (size[1] - Pv[:, 1]) / (2 * size[1]) * nyb / (nyb + pad)], 1)
    uvb[np.repeat(~top, 3)] = [0.5, 0.5 * pad / (nyb + pad)]
    write_obj(out / f"{name}_photo.obj", Vb, Fb, uvb); cv2.imwrite(str(out / f"{name}_photo.png"), tb.astype(np.uint8))
    print(f"{name}: box {2 * size[0] * 1e3:.0f} x {2 * size[1] * 1e3:.0f} mm, head at the {'+x' if head_sign > 0 else '-x'} end ({nh * 1e3 / a.res:.0f} mm), image dir of +x {d_img.round(1)}")

# ------------------------------------------------------------------ clutter: world-frame primitive meshes with one colour
for g in range(model.ngeom):
    name = model.geom(g).name
    if name not in CLUTTER_TOP: continue
    size = model.geom_size[g]; t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER: mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=48)
    elif t == mujoco.mjtGeom.mjGEOM_BOX: mesh = trimesh.creation.box(extents=2 * size)
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID: mesh = trimesh.creation.icosphere(subdivisions=3); mesh.apply_scale(size)
    else: continue
    Vc, Fc = np.asarray(mesh.vertices), np.asarray(mesh.faces); col = to_wrist(CLUTTER_TOP[name])  # in the primitive's own frame: photo_scene.py gives the mesh the primitive's pos/quat
    write_obj(out / f"{name}_photo.obj", Vc, Fc, np.full((3 * len(Fc), 2), 0.5)); cv2.imwrite(str(out / f"{name}_photo.png"), np.full((4, 4, 3), col, np.uint8))
    print(f"{name}: top {CLUTTER_TOP[name]} -> wrist {col.round(0).tolist()}")
json.dump(dict(wrist_bgr=WRIST, slot_factor=SLOT_FACTOR, top_affine_bgr=dict(gain=A.tolist(), offset=B.tolist()), yaml_top_rgb=dict(color_gain=A[::-1].round(4).tolist(), color_offset=B[::-1].round(2).tolist()),
               clutter_top_bgr=CLUTTER_TOP, case_dz=a.case_dz, rim_mm=a.rim_mm, slot_z_mm=a.slot_z, head_frac=a.head_frac), open(out / "materials.json", "w"), indent=1)
print("wrote", out)
