"""Photo textures for the free objects of the wrist view: the real wrist pixels of a recorded episode projected onto the
object meshes (the wrist-camera analogue of the top camera's real-background composite).

Per object and per depth-sidecar frame in its frame range (the object lying untouched on the table):
1. real table plane from the depth sidecar (RANSAC outside the object's sim mask); real object mask = points 4-40 mm above it,
   the component nearest the sim projection (clippers: also metal-coloured, low saturation, and sized like the box);
2. alignment in the camera frame, independent of the wrist calibration (its 4 cm bias) and of the start-pose error:
   case = centroid shift + point-to-plane ICP of the depth points against the scan mesh; clipper = PCA of the points on the plane
   (centre, long axis), top face at the box height, texture stretched to the measured footprint;
3. accepted when the projected mesh and the real mask agree (IoU), every texel of the mesh (per-face KxK texel blocks, see
   `atlas`) is projected into the frame; kept when it faces the camera, lies inside the real mask and matches the real depth;
   weighted mean over frames, divided by the render's lighting factor for the texel normal, min(0.6 + 0.65 nz + 0.25 n.l2, 1):
   the renderer clamps the lit white material at 1 before the texture multiplies it, so up-facing texels keep the real colour and
   walls are brightened. Textures are written in TOP-camera colours (wrist bake / --to_top_gain) so the camera config's color_gain
   makes them wrist colours again; the case's planar region takes its low frequencies from --top_frame (exact top alignment).
Unseen case texels fall back to the scan colours (case_texture.py), brightness-matched; clipper side faces take the top face mean.
Writes OUT/<object>_photo.obj + .png + _frames.json and a review sheet OUT/<object>_align.jpg.
usage: MUJOCO_GL=egl uv run python scripts/object_photo_texture.py ROOT EP SCENE_XML OUT_DIR
           --objects clipper_case:250:740 clipper_large:250:600 clipper_small:250:2090 [--every 6] [--ply work/nail/mesh_object.ply]"""
import argparse, json, pathlib, sys, cv2, mujoco, numpy as np, open3d as o3d, trimesh
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scene_wrist as Wr
from scene_frames import episode_states, decode_frames, WRIST
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("ep", type=int); ap.add_argument("scene"); ap.add_argument("out")
ap.add_argument("--objects", nargs="+", default=["clipper_case:250:740", "clipper_large:250:600", "clipper_small:250:2090"], help="body:first_frame:last_frame")
ap.add_argument("--every", type=int, default=6); ap.add_argument("--ply", default=None, help="scan mesh with vertex colours for the case fallback")
ap.add_argument("--more_episodes", type=int, nargs="*", default=[], help="further episodes of ROOT pooled in; frames = after the gripper first opens until the first grasp (objects untouched); the case may sit anywhere (whole image, yaw search)")
ap.add_argument("--case_obj", default=str(pathlib.Path(__file__).resolve().parents[1] / "assets/clipper_case/clipper_case_visual.obj"))
ap.add_argument("--topk", type=int, default=5, help="views kept per texel (median)")
ap.add_argument("--top_frame", help="real top-camera frame with the objects at the scene's start pose (robot parked): low-frequency colours of the case's planar atlas and the top-camera colour space")
ap.add_argument("--planar_res", type=int, default=3000, help="texels per metre of the case's planar (top-down) atlas region")
ap.add_argument("--to_top_gain", type=float, nargs=3, default=[0.766, 0.655, 0.558], help="wrist = gain x top (RGB): wrist-baked colours are divided by it so every texture lives in top-camera colours and the camera config applies the gain"); ap.add_argument("--min_px", type=int, default=2500); ap.add_argument("--min_iou", type=float, default=0.6); ap.add_argument("--min_iou_box", type=float, default=0.45); ap.add_argument("--k_case", type=int, default=8); ap.add_argument("--k_box", type=int, default=160)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
H, W = 480, 640; FX, FY, CX, CY = Wr.FX, Wr.FY, Wr.CX, Wr.CY
L2 = np.array([0.6, 0.5, 0.6]); L2 /= np.linalg.norm(L2)   # the second directional light, pointing back at the source

model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); renderer = mujoco.Renderer(model, H, W)
objects = [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in a.objects]


def auto_range(states):
    closed = states[:, 6] > -60; edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
    return int(edges[1]) + 30, int(edges[2]) if len(edges) > 2 else len(states)


episodes = []  # (ep, states, frames to use, {body: (lo, hi)}, any_pose)
for ep in [a.ep] + list(a.more_episodes):
    states, _ = episode_states(a.root, ep); depth_frames = Wr.depth_frames(a.root, ep)
    ranges = {n: (lo, hi) for n, lo, hi in objects} if ep == a.ep else dict.fromkeys([n for n, _, _ in objects], auto_range(states))
    frames = sorted({f for lo, hi in ranges.values() for f in depth_frames[::a.every // 3 or 1] if lo <= f <= hi})
    episodes.append((ep, states, frames, ranges, ep != a.ep)); print(f"episode {ep}: ranges {ranges}, {len(frames)} candidate frames")


def cam_cv(model, data):
    """World->OpenCV camera (x right, y down, z forward) of the sim wrist camera at the current pose."""
    cid = model.camera("wrist").id; R = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1]); t = data.cam_xpos[cid].copy()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def body_T(model, data, name):
    T = np.eye(4); b = model.body(name).id; T[:3, :3] = data.xmat[b].reshape(3, 3); T[:3, 3] = data.xpos[b]; return T


def project(P_cv):
    z = P_cv[:, 2]; return np.stack([FX * P_cv[:, 0] / z + CX, FY * P_cv[:, 1] / z + CY], -1), z


def atlas(V, F, K):
    """Per-face KxK texel blocks. Corner c of face i sits at block texel (0,0), (K-1,0), (0,K-1); texels past the diagonal mirror
    the inside ones. Returns texel 3D points (S,S,3) in the mesh frame, face index per texel (S,S), per-face-corner UVs (3n,2)."""
    n = len(F); side = int(np.ceil(np.sqrt(n))); S = side * K
    b, aa = np.meshgrid(np.arange(K), np.arange(K), indexing="ij"); outside = aa + b > K - 1
    a2 = np.where(outside, K - 1 - b, aa); b2 = np.where(outside, K - 1 - aa, b); l1 = a2 / (K - 1); l2 = b2 / (K - 1); l0 = 1 - l1 - l2
    P = np.zeros((S, S, 3)); face = np.full((S, S), -1, int); uv = np.zeros((3 * n, 2)); corners = np.array([[0, 0], [K - 1, 0], [0, K - 1]])
    for i, f in enumerate(F):
        bx, by = (i % side) * K, (i // side) * K; v0, v1, v2 = V[f]
        P[by:by + K, bx:bx + K] = l0[..., None] * v0 + l1[..., None] * v1 + l2[..., None] * v2; face[by:by + K, bx:bx + K] = i
        for c, (dx, dy) in enumerate(corners): uv[3 * i + c] = [(bx + dx + 0.5) / S, 1 - (by + dy + 0.5) / S]
    return P, face, uv


def atlas_planar(V, F, res, K, top_nz=0.3):
    """Case atlas: faces with nz > top_nz share one top-down planar region (texel = (x, y) of the body footprint, 3D point = first
    mesh hit from above), so neighbouring texels are neighbours on the object and the far top camera's mip filtering stays coherent;
    the wall faces keep per-face KxK blocks below the planar region. Returns P_tex, face_tex (-1 = no hit), uv, planar mask."""
    fn = trimesh.Trimesh(V, F, process=False).face_normals; top = fn[:, 2] > top_nz; walls = np.flatnonzero(~top)
    x0, y0 = V[:, :2].min(0) - 2 / res; x1, y1 = V[:, :2].max(0) + 2 / res; nx, ny = int(np.ceil((x1 - x0) * res)), int(np.ceil((y1 - y0) * res))
    S = max(nx, ny); 
    while (S - ny) // K * (S // K) < len(walls): S += K
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V, o3d.core.float32), o3d.core.Tensor(F, o3d.core.int32)))
    xs = x0 + (np.arange(nx) + 0.5) / res; ys = y1 - (np.arange(ny) + 0.5) / res; X, Y = np.meshgrid(xs, ys)
    origins = np.stack([X, Y, np.full_like(X, V[:, 2].max() + 0.01)], -1).reshape(-1, 3); rays = np.concatenate([origins, np.tile([0, 0, -1.0], (len(origins), 1))], 1)
    hit = scene.cast_rays(o3d.core.Tensor(rays, o3d.core.float32)); t = hit["t_hit"].numpy(); fid = hit["primitive_ids"].numpy().astype(np.int64)
    ok = np.isfinite(t); P = np.zeros((S, S, 3)); face = np.full((S, S), -1, int)
    P[:ny, :nx] = (origins + np.where(ok, t, 0)[:, None] * np.array([0, 0, -1.0])).reshape(ny, nx, 3); face[:ny, :nx] = np.where(ok, fid, -1).reshape(ny, nx)
    uv = np.zeros((3 * len(F), 2)); planar = np.zeros((S, S), bool); planar[:ny, :nx] = True
    for i in np.flatnonzero(top):
        for c in range(3): uv[3 * i + c] = [(V[F[i, c], 0] - x0) * res / S, 1 - (y1 - V[F[i, c], 1]) * res / S]
    b, aa = np.meshgrid(np.arange(K), np.arange(K), indexing="ij"); outside = aa + b > K - 1
    a2 = np.where(outside, K - 1 - b, aa); b2 = np.where(outside, K - 1 - aa, b); l1 = a2 / (K - 1); l2 = b2 / (K - 1); l0 = 1 - l1 - l2
    per_row = S // K; corners = np.array([[0, 0], [K - 1, 0], [0, K - 1]]); row0 = int(np.ceil(ny / K)) * K
    for j, i in enumerate(walls):
        bx, by = (j % per_row) * K, row0 + (j // per_row) * K; v0, v1, v2 = V[F[i]]
        P[by:by + K, bx:bx + K] = l0[..., None] * v0 + l1[..., None] * v1 + l2[..., None] * v2; face[by:by + K, bx:bx + K] = i
        for c, (dx, dy) in enumerate(corners): uv[3 * i + c] = [(bx + dx + 0.5) / S, 1 - (by + dy + 0.5) / S]
    print(f"planar atlas {S}x{S}: footprint {nx}x{ny} texels ({1e3 / res:.2f} mm), {top.sum()} top faces, {len(walls)} wall faces in blocks from row {row0}")
    return P, face, uv, planar


def write_obj(path, V, F, uv):
    verts = V[F.reshape(-1)]; faces = np.arange(3 * len(F)).reshape(-1, 3)
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts); fh.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
        fh.writelines(f"f {a0 + 1}/{a0 + 1} {b0 + 1}/{b0 + 1} {c0 + 1}/{c0 + 1}\n" for a0, b0, c0 in faces)


def scan_colours(ply, V_obj, F_obj):
    """Vertex colours of the scan cluster matching the visual OBJ (case_texture.py)."""
    m = o3d.io.read_triangle_mesh(ply); cl, cnt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl)
    for k in np.argsort(-np.asarray(cnt)):
        sub = o3d.geometry.TriangleMesh(m); sub.remove_triangles_by_mask(cl != k); sub.remove_unreferenced_vertices()
        if len(sub.vertices) == len(V_obj): break
    Vs = np.asarray(sub.vertices); shift = (Vs[:, :2].min(0) + Vs[:, :2].max(0)) / 2; assert np.abs(Vs[:, :2] - shift - V_obj[:, :2]).max() < 1e-4
    return (np.asarray(sub.vertex_colors) * 255)[:, ::-1]  # BGR like the frames


def label(im, t):
    cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); return im


def components(mask, min_px=150):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == i, cent[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


for name, lo, hi in objects:
    body = model.body(name).id; geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    vis = [g for g in geoms if model.geom_group[g] == 2]; is_box = len(vis) == 0  # clippers: one box geom, no visual mesh
    if is_box:
        size = model.geom_size[geoms[0]].copy(); mesh = trimesh.creation.box(extents=2 * size); V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces); K = a.k_box
    else:
        # the visual OBJ sits at the body origin (no geom pos/quat in the scene); the scan colours are indexed by its vertices
        tm = trimesh.load(a.case_obj, process=False); V, F = np.asarray(tm.vertices), np.asarray(tm.faces); K = a.k_case
    fn = trimesh.Trimesh(V, F, process=False).face_normals
    if is_box: P_tex, face_tex, uv = atlas(V, F, K); planar = np.zeros(face_tex.shape, bool)
    else: P_tex, face_tex, uv, planar = atlas_planar(V, F, a.planar_res, K)
    S = P_tex.shape[0]; N_tex = fn[np.clip(face_tex, 0, None)]; has_face = (face_tex >= 0).reshape(-1)
    o3m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); o3m.compute_vertex_normals(); mesh_pts = o3m.sample_points_uniformly(40000)
    jnt = model.body_jntadr[body]; qadr = model.jnt_qposadr[jnt]
    # per texel the TOPK best-weighted views only (nearest, most frontal): a mean over every frame blurs the slots by the 1-2 mm
    # alignment scatter and averages shadowed with lit frames
    topw = np.zeros((S, S, a.topk)); topc = np.zeros((S, S, a.topk, 3)); kept = []; rows = []; top_scale = np.ones(3); ref_cols = []
    for ep, states, ep_frames, ranges, any_pose in episodes:
      lo, hi = ranges[name]; real_rgb = decode_frames(a.root, ep, WRIST, [f for f in ep_frames if lo <= f <= hi])  # BGR
      for f in [f for f in ep_frames if lo <= f <= hi]:
        mujoco.mj_resetData(model, data); data.qpos[:8] = to_qpos(states[f].astype(float)); mujoco.mj_forward(model, data)
        renderer.enable_segmentation_rendering(); renderer.update_scene(data, "wrist"); seg = renderer.render()[:, :, 0]; renderer.disable_segmentation_rendering()
        sim_mask = (seg >= 0) & (model.geom_bodyid[np.clip(seg, 0, None)] == body)
        if any_pose: sim_mask[:] = True   # the object may be anywhere: whole image as the region, the sim pose only seeds the yaw search
        elif sim_mask.sum() < a.min_px: continue
        T_wc = cam_cv(model, data); T_wo = body_T(model, data, name); T_co = np.linalg.inv(T_wc) @ T_wo
        d16, fd = Wr.load_depth(a.root, a.ep, f); dm = d16 * Wr.UNITS_M; valid = (d16 > 0) & (dm < 0.8); pts = Wr.backproject(np.where(valid, dm, 0))
        region = cv2.dilate(sim_mask.astype(np.uint8), np.ones((201, 201), np.uint8)) > 0
        tab = valid & ~region if (valid & ~region).sum() > 3000 else valid
        Pt = pts[tab]; sub = Pt[np.random.default_rng(0).choice(len(Pt), min(6000, len(Pt)), replace=False)]; n_pl, d_pl, _ = Wr.ransac_plane(sub)
        h = np.full((H, W), np.nan); h[valid] = pts[valid] @ n_pl + d_pl  # height above the real table, toward the camera
        above = (h > 0.004) & (h < 0.04) & region
        bgr = real_rgb[f]; hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if is_box:
            above &= (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 70); above = (cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0) & valid
        comps = components(above)
        if not comps: continue
        sim_c = np.array(np.nonzero(sim_mask)).mean(1)[::-1]
        if is_box:
            # footprint on the plane sized like the box: length within 25% of 2*size[0], width below 3*size[1]
            good = []
            for m, c in comps:
                Q = pts[m]; Q = Q - np.outer(Q @ n_pl + d_pl, n_pl); mu = Q.mean(0); _, s, vt = np.linalg.svd(Q - mu, full_matrices=False); ax = vt[0]
                length = np.ptp((Q - mu) @ ax); width = np.ptp((Q - mu) @ vt[1])
                if abs(length - 2 * size[0]) < 0.25 * 2 * size[0] and width < 2 * 2 * size[1]: good.append((np.linalg.norm(c - sim_c), m, mu, ax, length, width))
            if not good: continue
            _, real_mask, mu, ax, length, width = min(good, key=lambda g: g[0])
            if bgr[real_mask].mean() > 200:
                print(f"{name} ep{ep} f{f}: blown highlights (mean {bgr[real_mask].mean():.0f}) -> skipped"); continue
            # box pose on the plane: x = long axis (sign from the sim x axis), z = plane normal, centre at the box half height
            x_sim = (np.linalg.inv(T_wc) @ T_wo)[:3, 0]; ax = ax if ax @ x_sim > 0 else -ax; zax = n_pl; ax -= (ax @ zax) * zax; ax /= np.linalg.norm(ax); yax = np.cross(zax, ax)
            T_co = np.eye(4); T_co[:3, :3] = np.stack([ax, yax, zax], 1); T_co[:3, 3] = mu + zax * size[2]
            top_scale = np.array([np.clip(length / (2 * size[0]), 1, 1.6), np.clip(width / (2 * size[1]), 1, 1.5), 1.0]); iou_mask = real_mask
            frame_iou = None
        else:
            real_mask, c = max(comps, key=lambda mc: mc[0].sum()); real_mask = cv2.morphologyEx(real_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
            Q = pts[real_mask & valid]
            if len(Q) < 2000: continue
            # the component must hold the white foam (the black headphone case is a bigger raised blob the mesh fits inside at IoU 0.7;
            # a size test fails because the real raised footprint, 100 x 150 mm, exceeds the 88 x 119 mm scan mesh)
            foam = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 110))[real_mask].mean()
            if foam < 0.2:
                print(f"{name} ep{ep} f{f}: foam fraction {foam:.2f} in the raised component -> skipped"); continue
            # centroid shift along the plane, then ICP (real points -> mesh), coarse to fine; other episodes: the case may be turned,
            # so the four yaws about the plane normal are tried and the best fitness wins
            c_sim = (T_co @ np.r_[V.mean(0), 1])[:3]; c_real = Q.mean(0); shift = c_real - c_sim; shift -= (shift @ n_pl) * n_pl; T_co[:3, 3] += shift
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q[np.random.default_rng(0).choice(len(Q), min(len(Q), 20000), replace=False)]))
            best = None
            for yaw in ((0.0,) if not any_pose else (0.0, np.pi / 2, np.pi, -np.pi / 2)):
                Rz = o3d.geometry.get_rotation_matrix_from_axis_angle(n_pl * yaw); T0 = T_co.copy(); T0[:3, :3] = Rz @ T0[:3, :3]; T0[:3, 3] = Rz @ (T0[:3, 3] - c_real) + c_real
                T_oc = np.linalg.inv(T0)
                for dist in (0.03, 0.012, 0.005):
                    reg = o3d.pipelines.registration.registration_icp(src, mesh_pts, dist, T_oc, o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)); T_oc = reg.transformation
                if best is None or reg.fitness > best[0]: best = (reg.fitness, reg.inlier_rmse, T_oc)
            T_co = np.linalg.inv(best[2]); iou_mask = real_mask; frame_iou = best[1]
        # projected mesh mask vs the real mask
        Pm = (T_co @ np.c_[np.asarray(mesh_pts.points), np.ones(len(mesh_pts.points))].T).T[:, :3]; uvm, zm = project(Pm); ok = (zm > 0.05) & (uvm[:, 0] >= 0) & (uvm[:, 0] < W) & (uvm[:, 1] >= 0) & (uvm[:, 1] < H)
        proj = np.zeros((H, W), np.uint8); proj[uvm[ok, 1].astype(int), uvm[ok, 0].astype(int)] = 1; proj = cv2.morphologyEx(proj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        iou = (proj & iou_mask).sum() / max((proj | iou_mask).sum(), 1)
        accept = iou >= (a.min_iou_box if is_box else a.min_iou + (0.1 if any_pose else 0.0))  # the box is narrower than the real lever's head: the footprint check carries the box; other episodes: stricter
        # texel sampling
        Ps = P_tex.reshape(-1, 3) * (top_scale if is_box else 1.0); Pc = (T_co[:3, :3] @ Ps.T).T + T_co[:3, 3]; Nc = (T_co[:3, :3] @ N_tex.reshape(-1, 3).T).T
        uvt, zt = project(Pc); cosv = -(Nc * (Pc / np.linalg.norm(Pc, axis=1, keepdims=True))).sum(1)
        inside = (zt > 0.05) & (uvt[:, 0] >= 1) & (uvt[:, 0] < W - 1) & (uvt[:, 1] >= 1) & (uvt[:, 1] < H - 1)
        ui = np.clip(uvt[:, 0], 0, W - 1).astype(int); vi = np.clip(uvt[:, 1], 0, H - 1).astype(int)
        in_mask = cv2.dilate(iou_mask.astype(np.uint8), np.ones((3, 3), np.uint8))[vi, ui] > 0; dist_edge = cv2.distanceTransform(cv2.dilate(iou_mask.astype(np.uint8), np.ones((3, 3), np.uint8)), cv2.DIST_L2, 3)[vi, ui]
        dz = np.abs(zt - dm[vi, ui]); depth_ok = valid[vi, ui] & (dz < (0.015 if is_box else 0.008))
        if is_box: depth_ok = in_mask  # shiny metal: trust the mask, the depth is noisy
        good = inside & in_mask & depth_ok & (cosv > 0.2) & has_face
        if accept and good.sum() > 0:
            col = cv2.remap(bgr, uvt[:, 0].astype(np.float32).reshape(S, S), uvt[:, 1].astype(np.float32).reshape(S, S), cv2.INTER_LINEAR).reshape(-1, 3).astype(float)
            # other episodes were recorded under another white balance: match the frame's object chromaticity to the scene episode's
            fc = bgr[iou_mask].mean(0)
            if any_pose and ref_cols:
                ref = np.mean(ref_cols, 0); col *= (ref / ref.mean()) / (fc / fc.mean())
            elif not any_pose: ref_cols.append(fc)
            # lit frames first: the sim casts no arm shadow, and the nearest views (largest 1/z^2) are the ones under the arm, so the
            # frame's own object brightness enters the weight steeply
            lit = (bgr[iou_mask].mean() / 255.0) ** (1 if is_box else 4)  # metal: the brightest views are saturated reflections
            w = np.where(good, lit * cosv ** 2 / zt ** 2 * np.minimum(dist_edge, 4) / 4, 0.0)
            w = w.reshape(S, S); col = col.reshape(S, S, 3); slot = topw.argmin(-1); cur = np.take_along_axis(topw, slot[..., None], -1)[..., 0]; better = w > cur
            ii, jj = np.nonzero(better); topw[ii, jj, slot[ii, jj]] = w[ii, jj]; topc[ii, jj, slot[ii, jj]] = col[ii, jj]; kept.append(dict(episode=ep, frame=f, depth_frame=fd, iou=float(iou), brightness=float(bgr[iou_mask].mean()), texels=int(good.sum()), icp_rmse_mm=None if frame_iou is None else 1e3 * float(frame_iou)))
        ov = bgr.copy(); ov[iou_mask] = (0.6 * ov[iou_mask] + np.array([0, 0, 100])).astype(np.uint8); ov[proj] = (0.6 * ov[proj] + np.array([100, 0, 0])).astype(np.uint8)
        rows.append(label(ov, f"ep{ep} f{f} iou {iou:.2f} {'ok' if accept else 'rejected'}" + (f" rmse {1e3 * frame_iou:.1f}mm" if frame_iou is not None else f" len {length * 1e3:.0f}mm")))
        print(f"{name} ep{ep} f{f}: sim px {sim_mask.sum()}, real px {iou_mask.sum()}, IoU {iou:.2f}" + (f", ICP rmse {1e3 * frame_iou:.1f} mm" if frame_iou is not None else f", footprint {length * 1e3:.0f} x {width * 1e3:.0f} mm") + (f", texels {good.sum()}" if accept else " -> rejected"))
    n_rows = len(rows)
    if rows:  # at most 60 tiles (JPEG height limit), evenly sampled
        rows = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, min(len(rows), 60)).astype(int))]
        cv2.imwrite(str(out / f"{name}_align.jpg"), np.vstack([np.hstack(rows[i:i + 3] + [np.zeros_like(rows[0])] * (3 - len(rows[i:i + 3]))) for i in range(0, len(rows), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    # assemble the texture: observed texels / lighting factor, fallback for the rest
    # median over the kept views (robust to one misaligned or off-colour frame); unfilled slots are ignored
    wsum = topw.sum(-1); seen = wsum > 1e-9; tex = np.zeros((S, S, 3)); cm = np.where((topw > 0)[..., None], topc, np.nan); tex[seen] = np.nanmedian(cm, axis=2)[seen]
    n_w = N_tex.reshape(-1, 3); Rwo = body_T(model, data, name)[:3, :3]; nz = (n_w @ Rwo.T)
    # the renderer clamps the lit material colour at 1 before the texture multiplies it (grey-texture probe: factor 1.00 on
    # top faces, 0.65-0.86 on walls), so texels facing up keep the real colour and only the walls are brightened
    Lf = np.minimum(0.6 + 0.65 * np.clip(nz[:, 2], 0, None) + 0.25 * np.clip(nz @ L2, 0, None), 1.0).reshape(S, S)
    g = np.array(a.to_top_gain)[::-1]  # BGR
    tex = np.clip(tex / g, 0, 255)  # wrist colours -> top-camera colours (the camera config multiplies the wrist render by the gain)
    if is_box:
        top = (face_tex >= 0) & (fn[np.clip(face_tex, 0, None)][:, :, 2] > 0.9); mean_top = tex[seen & top].mean(0) if (seen & top).any() else np.array([150, 150, 150.])
        tex[~(seen & top)] = mean_top; tex[seen & top] = tex[seen & top]  # side/bottom faces and unseen top texels = top mean
        seen_frac = (seen & top).sum() / max(top.sum(), 1)
    else:
        C = scan_colours(a.ply, V, F) if a.ply else np.full((len(V), 3), 140.0)
        fb = C[F[np.clip(face_tex, 0, None)]].mean(-2)  # per texel: mean scan colour of its face -> (S, S, 3)
        gain = tex[seen].mean(0) / np.maximum(fb[seen].mean(0), 1); fb = np.clip(fb * gain, 0, 255)
        tex[~seen] = fb[~seen]; seen_frac = seen[face_tex >= 0].mean()
        print(f"{name}: fallback scan colours scaled by {gain.round(2)} (BGR) on {(~seen).sum()} texels")
        if a.top_frame:
            # planar region: low frequencies from the real top frame (objects placed from that image, so the alignment is exact; the
            # top camera's own colour space), detail ratio from the wrist bake; unseen texels take the top colour alone
            mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data); T_wo = body_T(model, data, name); cid = model.camera("ext").id
            Rc = data.cam_xmat[cid].reshape(3, 3); pc = data.cam_xpos[cid]; fpx = (H / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
            Pw = (T_wo[:3, :3] @ P_tex.reshape(-1, 3).T).T + T_wo[:3, 3]; c = (Pw - pc) @ Rc; zc = -c[:, 2]
            U = (W / 2 + fpx * c[:, 0] / zc).reshape(S, S).astype(np.float32); Vv = (H / 2 - fpx * c[:, 1] / zc).reshape(S, S).astype(np.float32)
            top_tex = cv2.remap(cv2.imread(a.top_frame), U, Vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(float)
            sig = 1.5e-3 * a.planar_res; m = (seen & planar).astype(np.float32)
            low = cv2.GaussianBlur(tex * m[:, :, None], (0, 0), sig) / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 1e-3)[:, :, None]
            top_low = cv2.GaussianBlur(top_tex, (0, 0), sig); hyb = np.clip(top_low * tex / np.maximum(low, 1), 0, 255)
            tex[seen & planar] = hyb[seen & planar]; tex[planar & ~seen & (face_tex >= 0)] = top_tex[planar & ~seen & (face_tex >= 0)]
            nohit = planar & (face_tex < 0); idx = cv2.distanceTransform((~(planar & (face_tex >= 0))).astype(np.uint8), cv2.DIST_L2, 3, dstType=cv2.CV_32F)
            # texels outside the footprint silhouette: nearest filled texel (mip safety)
            from scipy import ndimage; _, inds = ndimage.distance_transform_edt(~(planar & (face_tex >= 0)), return_indices=True); tex[nohit] = tex[inds[0][nohit], inds[1][nohit]]
            print(f"{name}: planar region hybrid (top frame low-pass x wrist detail) on {(seen & planar).sum()} texels, top colour alone on {(planar & ~seen & (face_tex >= 0)).sum()}")
    tex = np.clip(tex / Lf[:, :, None], 0, 255).astype(np.uint8)
    obj = out / f"{name}_photo.obj"; png = out / f"{name}_photo.png"; write_obj(obj, V, F, uv); cv2.imwrite(str(png), tex)
    json.dump(dict(body=name, frames=kept, texture_px=S, k=K, top_scale=top_scale.tolist(), seen_frac=float(seen_frac), box_size=size.tolist() if is_box else None), open(out / f"{name}_frames.json", "w"), indent=1)
    print(f"{name}: {len(kept)} frames accepted of {n_rows}, texels seen {100 * seen_frac:.0f}%, texture {S}x{S} -> {obj}, {png}")
