"""base.ply + lid.ply + hinge.json -> MuJoCo asset dir: visual OBJs in body frames, CoACD hulls, <name>.xml body include.
Base body origin = base footprint centre at table level; lid body = child at the hinge point, hinge joint about hinge axis.
The TSDF halves are open one-sided shells: holes in the visual mesh are fanned shut and a flat bottom disc closes the underside. A pull tab measured in hinge.json ("lid_tab") is added to the lid as four boxes (slotted).
usage: build_clamshell_asset.py work_dir asset_dir name [--mass_kg 0.3] [--hulls 24] [--threshold 0.05]"""
import argparse, json, pathlib, numpy as np, cv2, trimesh, coacd, open3d as o3d
p = argparse.ArgumentParser(); p.add_argument("work"); p.add_argument("asset"); p.add_argument("name"); p.add_argument("--mass_kg", type=float, default=0.3)
p.add_argument("--hulls", type=int, default=24); p.add_argument("--threshold", type=float, default=0.05); a = p.parse_args()
W = pathlib.Path(a.work); A = pathlib.Path(a.asset); A.mkdir(parents=True, exist_ok=True); h = json.load(open(W / "hinge.json"))
def load(n):
    m = o3d.io.read_triangle_mesh(str(W / n)); return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8), process=False)
base, lid = load("base.ply"), load("lid.ply")
hp = np.array(h["point_m"]); ax = np.array(h["axis"]); ax /= np.linalg.norm(ax); u = np.array(h["lid_side_dir"]); u /= np.linalg.norm(u)
# rim height of a half = top of its wall: polar bins about the wall-ring centre, 95th pct z of the outer 8 mm in each bin, median over bins
# (robust to the strap loop, the missing rim under the tab and the headphones; the hinge side is skipped). A pi flip about height h maps z -> 2h - z,
# so the lid rim lands on the base rim when h is the mean of the two rim heights.
def rim_z(m, toward):  # bins within 40 deg of the hinge direction are skipped (fabric seam, no rim there)
    V = m.vertices; ring = V[(V[:, 2] > 0.015) & (V[:, 2] < 0.035)][:, :2]; c = ring.mean(0); rel = V[:, :2] - c; ang = np.arctan2(rel[:, 1], rel[:, 0]); r = np.linalg.norm(rel, axis=1); zs = []
    for a0 in np.arange(-np.pi, np.pi, np.pi / 18):
        b = (ang >= a0) & (ang < a0 + np.pi / 18)
        if b.sum() < 30 or np.cos(a0 + np.pi / 36) * toward[0] + np.sin(a0 + np.pi / 36) * toward[1] > np.cos(np.radians(40)): continue
        outer = b & (r > np.percentile(r[b], 99) - 0.008); zs.append(np.percentile(V[outer, 2], 95))
    return float(np.median(zs))
zb, zl = rim_z(base, u[:2]), rim_z(lid, -u[:2]); hp[2] = (zb + zl) / 2; print("rim heights base %.3f lid %.3f -> hinge height %.3f" % (zb, zl, hp[2]))
base_origin = np.r_[(base.vertices[:, :2].min(0) + base.vertices[:, :2].max(0)) / 2, 0.0]
base_local = base.copy(); base_local.apply_translation(-base_origin)
lid_local = lid.copy(); lid_local.apply_translation(-hp)
# the lid closes by rotating about the hinge so its centroid ends up above the base: pick the axis sign that does that at +pi
def rot(axis, ang):
    return trimesh.transformations.rotation_matrix(ang, axis)
c_lid = lid_local.centroid
closed = (rot(ax, np.pi) @ np.r_[c_lid, 1])[:3] + hp; half = (rot(ax, np.pi / 2) @ np.r_[c_lid, 1])[:3] + hp
if half[2] < hp[2]: ax = -ax  # the lid must rise through the half-open pose, not dive through the table
print("closed lid centroid %s (base centroid xy %s); half-open centroid z %.3f" % (closed.round(3), base.centroid[:2].round(3), max(half[2], 2 * hp[2] - half[2])))
# solid visual shell: the TSDF half is an open surface (no underside, holes in the floor and cups, open outline at the table).
# Small boundary loops (holes, the slit along the seam cut) are fanned from their centroid, wound like the faces around them;
# the outline loop is left alone (a skirt from it casts spike shadows into the cavity); a flat bottom disc plus a short can wall
# close the underside and the unseen lower outer wall, which is what shows as the lid's top and sides when the case is closed.
def cap_loops(m, z0, z_can):  # z0 = table height in the body frame
    e = m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]; V = m.vertices; C = m.visual.vertex_colors; F = [m.faces]; newV = []; newC = []
    for comp in trimesh.graph.connected_components(e, min_len=3):
        if np.ptp(V[comp, :2], axis=0).max() > 0.08: continue
        ce = e[np.isin(e[:, 0], comp) & np.isin(e[:, 1], comp)]; k = len(V) + len(newV)
        newV.append(V[comp].mean(0)); newC.append(C[comp].mean(0).astype(np.uint8)); F.append(np.c_[ce[:, 1], ce[:, 0], np.full(len(ce), k)])  # (b, a, centre): the hole lies right of the directed edge a->b
    # the phone never saw the lower outer wall (grazing angles): close it with a short tapered can from the bottom disc up to
    # z_can, on the hull of the mid-height wall ring (the widest outline); the real rim band continues above it
    ring = V[(V[:, 2] > z0 + 0.015) & (V[:, 2] < z0 + 0.035), :2]; hull = cv2.convexHull(ring.astype(np.float32))[:, 0, :].astype(np.float64); n = len(hull); cen = hull.mean(0)
    if np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1]) > 0: hull = hull[::-1]  # clockwise from above -> disc normal down, can normal out
    k = len(V) + len(newV); bot = cen + 0.96 * (hull - cen); i = np.arange(n); j = (i + 1) % n
    newV += [np.r_[q, z0 + 0.0005] for q in bot] + [np.r_[q, z0 + z_can] for q in hull] + [np.r_[cen, z0 + 0.0005]]; newC += [C.mean(0).astype(np.uint8)] * (2 * n + 1)
    F += [np.c_[k + i, k + j, np.full(n, k + 2 * n)], np.c_[k + j, k + i, k + n + i], np.c_[k + j, k + n + i, k + n + j]]
    return trimesh.Trimesh(np.vstack([V, newV]), np.vstack(F), vertex_colors=np.vstack([C, newC]), process=False)
base_vis, lid_vis = cap_loops(base_local, 0.0, 0.8 * zb), cap_loops(lid_local, -hp[2], 0.8 * zl)
print("visual shells: base %d faces (%d added), lid %d faces (%d added)" % (len(base_vis.faces), len(base_vis.faces) - len(base_local.faces), len(lid_vis.faces), len(lid_vis.faces) - len(lid_local.faces)))
# visual meshes (OBJ keeps no vertex colours; MuJoCo gets flat rgba; the PLYs keep the colours for later texturing)
base_vis.export(A / f"{a.name}_base_visual.obj"); lid_vis.export(A / f"{a.name}_lid_visual.obj")
hull_files = {}
for part, m in (("base", base_local), ("lid", lid_local)):
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=a.threshold, max_convex_hull=a.hulls, preprocess_mode="auto", seed=0)
    hull_files[part] = []
    for i, (v, f) in enumerate(parts):
        hull = trimesh.Trimesh(v, f, process=False); hull = hull.convex_hull; fn = f"{a.name}_{part}_hull{i:02d}.obj"; hull.export(A / fn); hull_files[part].append((fn, hull.volume))
    print(part, "hulls", len(parts), "total hull volume %.4f m3" % sum(v for _, v in hull_files[part]))
vol = {k: sum(v for _, v in hf) for k, hf in hull_files.items()}; tot = sum(vol.values()); dens = a.mass_kg / tot
# pull tab: four boxes (top bar, two posts, block below the slot that also overlaps the lid wall), lid body frame, long side along the hinge axis
tab_geoms = []; tab = h.get("lid_tab")
if tab:
    th, w, top = tab["thickness_m"], tab["width_m"], tab["top_z_m"]; perp = np.cross([0, 0, 1], u); perp /= np.linalg.norm(perp)
    far_l = float(((lid.vertices[:, :2] - hp[:2]) @ u[:2]).max())
    tc = max(tab["centre_ts_m"][0], far_l + th / 2 + 0.002)  # inner face at least 2 mm outside the lid wall so the tab clears the base wall when closed
    s_c = tab["centre_ts_m"][1]; z_bot = tab["rim_z_m"] - 0.015; slot_w, slot_h, top_bar = min(0.036, 0.6 * w), 0.006, 0.008
    yaw = np.arctan2(u[1], u[0]); quat = f"{np.cos(yaw / 2):.5f} 0 0 {np.sin(yaw / 2):.5f}"; post_w = (w - slot_w) / 2
    boxes = [("top", s_c, top - top_bar, top, w), ("post_a", s_c - slot_w / 2 - post_w / 2, top - top_bar - slot_h, top - top_bar, post_w),
             ("post_b", s_c + slot_w / 2 + post_w / 2, top - top_bar - slot_h, top - top_bar, post_w), ("block", s_c, z_bot, top - top_bar - slot_h, w)]
    tab_vol = sum(th * bw * (z1 - z0) for _, _, z0, z1, bw in boxes); tab_mass = 0.008
    for nm, sc, z0, z1, bw in boxes:
        cxy = hp[:2] + u[:2] * tc + perp[:2] * sc; pos = f"{cxy[0] - hp[0]:.5f} {cxy[1] - hp[1]:.5f} {(z0 + z1) / 2 - hp[2]:.5f}"; size = f"{th / 2:.5f} {bw / 2:.5f} {(z1 - z0) / 2:.5f}"
        tab_geoms.append((nm, pos, quat, size, tab_mass * th * bw * (z1 - z0) / tab_vol))
    print("lid tab: %d x %d x %d mm (width x thickness x height above table %d..%d), slot %d x %d, centre %d mm from the hinge, %d mm across" % (w * 1000, th * 1000, (top - z_bot) * 1000, z_bot * 1000, top * 1000, slot_w * 1000, slot_h * 1000, tc * 1000, s_c * 1000))
L = ["<mujoco>", "  <asset>", f'    <mesh name="{a.name}_base_visual" file="{a.name}_base_visual.obj"/>', f'    <mesh name="{a.name}_lid_visual" file="{a.name}_lid_visual.obj"/>']
for part, hf in hull_files.items():
    L += [f'    <mesh name="{fn[:-4]}" file="{fn}"/>' for fn, _ in hf]
L += ["  </asset>", "  <default>", f'    <default class="{a.name}_col"><geom type="mesh" group="3" condim="4" friction="0.8 0.005 0.0001"/></default>',
      f'    <default class="{a.name}_vis"><geom type="mesh" group="2" contype="0" conaffinity="0" mass="0" rgba="0.12 0.12 0.13 1"/></default>', "  </default>", "  <worldbody>",
      f'    <body name="{a.name}" pos="0 0 0">', '      <freejoint/>', f'      <geom class="{a.name}_vis" mesh="{a.name}_base_visual"/>']
L += [f'      <geom class="{a.name}_col" mesh="{fn[:-4]}" mass="{a.mass_kg * v / tot:.5f}"/>' for fn, v in hull_files["base"]]
rel = hp - base_origin
L += [f'      <body name="{a.name}_lid" pos="{rel[0]:.5f} {rel[1]:.5f} {rel[2]:.5f}">', f'        <joint name="{a.name}_hinge" type="hinge" axis="{ax[0]:.5f} {ax[1]:.5f} {ax[2]:.5f}" range="0 3.1416" damping="0.02" armature="0.0005"/>',
      f'        <geom class="{a.name}_vis" mesh="{a.name}_lid_visual" rgba="0.14 0.14 0.15 1"/>']
L += [f'        <geom class="{a.name}_col" mesh="{fn[:-4]}" mass="{a.mass_kg * v / tot:.5f}"/>' for fn, v in hull_files["lid"]]
for nm, pos, quat, size, mass in tab_geoms:
    L += [f'        <geom name="{a.name}_tab_{nm}_vis" class="{a.name}_vis" type="box" pos="{pos}" quat="{quat}" size="{size}" rgba="0.10 0.10 0.11 1"/>',
          f'        <geom name="{a.name}_tab_{nm}" class="{a.name}_col" type="box" pos="{pos}" quat="{quat}" size="{size}" mass="{mass:.5f}"/>']
L += ["      </body>", "    </body>", "  </worldbody>", "</mujoco>"]
(A / f"{a.name}.xml").write_text("\n".join(L) + "\n")
json.dump({"base_origin_table_m": base_origin.tolist(), "hinge_point_m": hp.tolist(), "hinge_axis": ax.tolist(), "rim_z_m": {"base": zb, "lid": zl}, "hull_volume_m3": vol, "mass_kg": a.mass_kg, "density": dens,
           "lid_tab": tab}, open(A / f"{a.name}_meta.json", "w"), indent=1)
print("wrote", A / f"{a.name}.xml")
