"""Photo version of the nail scene for the wrist camera (the top camera uses the background composite instead):
1. table-top texture = the robot-free background rectified onto the table plane z=0 through the calibrated top camera;
   everything that is not table in the sim segmentation (clutter, robot at park, free objects) is filled along the grain;
2. flat lighting, no cast shadows (the texture is already lit) and the texture's gain calibrated so the rendered table
   matches the real frame where the table is visible; by default the texture is the smooth low-pass shade only (--grain
   keeps the wood grain with its strip fills);
3. clipper case visual mesh with the scan colours baked as a texture (case_texture.py); finer shadows;
4. optional robot include swap (robot_materials.py output, fitted under this lighting);
5. --object_tex: real wrist pixels baked on the case and the clippers (object_photo_texture.py), --gain: the wrist colour gain
   folded into the table texture (the table is only ever seen by the wrist camera; the top view is a composite).
Writes assets/table/table_top.png and SCENE_OUT; SCENE_IN stays untouched.
usage: MUJOCO_GL=egl uv run python scripts/photo_scene.py SCENE_IN BACKGROUND_PNG SCENE_OUT [--robot_xml R.xml]"""
import argparse, pathlib, re, cv2, mujoco, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("background"); ap.add_argument("out"); ap.add_argument("--px_per_m", type=int, default=1000)
ap.add_argument("--robot_xml"); ap.add_argument("--grain", action="store_true", help="keep the top camera's wood grain with the strip fills instead of the smooth shade")
ap.add_argument("--detail", help="table_detail.py patch (.npz): sharp real grain mirror-tiled over the smooth shade")
ap.add_argument("--object_tex", help="object_photo_texture.py + clutter_texture.py output dir: photo textures for the case, the two clipper boxes and the static clutter primitives")
ap.add_argument("--gain", type=float, nargs=3, help="wrist camera colour gain (RGB) folded into the table texture, so the camera config needs none and the object textures (already in wrist colours) stay untouched"); a = ap.parse_args()
src = pathlib.Path(a.scene).read_text(); H, W = 480, 640
m = re.search(r'<geom name="table" type="box" size="([\d.]+) ([\d.]+) ([\d.]+)" pos="([-\d.]+) ([-\d.]+) ([-\d.]+)"', src); sx, sy, _, cx, cy, _ = map(float, m.groups())
model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
cid = model.camera("ext").id; pos = data.cam_xpos[cid]; R = data.cam_xmat[cid].reshape(3, 3); f = (H / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
nx, ny = int(round(2 * sx * a.px_per_m)), int(round(2 * sy * a.px_per_m))
# texel (i, j): column i along +x of the plane, row j from the top along -y (MuJoCo puts the image's first row at the +y
# edge; checked against the real frame with photoreal_check.py).
xs = cx - sx + (np.arange(nx) + 0.5) / a.px_per_m; ys = cy + sy - (np.arange(ny) + 0.5) / a.px_per_m
X, Y = np.meshgrid(xs, ys); P = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
c = (P - pos) @ R; z = -c[:, 2]; U = (W / 2 + f * c[:, 0] / z).reshape(ny, nx).astype(np.float32); V = (H / 2 - f * c[:, 1] / z).reshape(ny, nx).astype(np.float32)
bg = cv2.imread(a.background); tex = cv2.remap(bg, U, V, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
r = mujoco.Renderer(model, H, W); r.enable_segmentation_rendering(); r.update_scene(data, "ext"); seg = r.render()[:, :, 0]; r.disable_segmentation_rendering()
# task objects (the gripper hovers around them, wide margin) vs clutter (free bodies too since 2026-10-17 night, narrow margin)
task = np.isin(np.arange(model.nbody), [model.body(n).id for n in ("clipper_case", "clipper_large", "clipper_small")])
gid = np.clip(seg, 0, None)
robot = np.isin(model.geom_bodyid, [model.body(b).id for b in ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right"]])
objects = cv2.morphologyEx(((seg >= 0) & (task[model.geom_bodyid[gid]] | robot[gid])).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # stray robot-id edge lines
clutter = (seg >= 0) & ~np.isin(seg, [model.geom("table").id, model.geom("floor").id]) & ~objects
occluded = cv2.dilate(clutter.astype(np.uint8), np.ones((15, 15), np.uint8)) | cv2.dilate(objects.astype(np.uint8), np.ones((81, 81), np.uint8))
bad = cv2.remap(occluded, U, V, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) > 0
bad[:30] = bad[-30:] = bad[:, :30] = bad[:, -30:] = True  # the sim box overhangs the real table edge: floor and basket pixels
# Fill on the texture divided by its own low-pass illumination (normalised convolution over the clean texels), so a copied
# strip lands at the local shade; each masked run of a column takes the clean strip of the same length right above it (else
# right below, else the nearest clean texel), so the grain keeps its along-grain variation instead of a constant streak.
valid = (~bad).astype(np.float32); sigma = 60
low = cv2.GaussianBlur(tex.astype(np.float32) * valid[:, :, None], (0, 0), sigma) / np.maximum(cv2.GaussianBlur(valid, (0, 0), sigma), 1e-3)[:, :, None]
tex = tex.astype(np.float32) / np.maximum(low, 1)
for i in range(nx):
    col = bad[:, i]
    if not col.any() or col.all(): continue
    edges = np.flatnonzero(np.diff(np.r_[0, col.astype(int), 0]))
    clean_runs = np.flatnonzero(np.diff(np.r_[0, (~col).astype(int), 0]))  # [start, end) of each clean run
    for j0, j1 in zip(edges[::2], edges[1::2]):
        L = j1 - j0; above = [(s0, e0) for s0, e0 in zip(clean_runs[::2], clean_runs[1::2]) if e0 == j0]; below = [(s0, e0) for s0, e0 in zip(clean_runs[::2], clean_runs[1::2]) if s0 == j1]
        s0, e0 = max(above + below, key=lambda r: r[1] - r[0])  # the longer clean neighbour, tiled back and forth over the run
        strip = tex[s0:e0, i]; cycle = np.concatenate([strip, strip[::-1]]); tex[j0:j1, i] = cycle[np.arange(L) % len(cycle)] if above and (s0, e0) == above[0] else cycle[(len(cycle) - 1 - np.arange(L)[::-1] % len(cycle))][::-1]
tex = np.clip(tex * low if a.grain else low, 0, 255).astype(np.uint8)
if a.detail:
    # smooth shade (top camera) x sharp grain ratio from a rectified wrist frame, mirror-tiled (seamless) at the patch's scale
    d = np.load(a.detail); det = d["detail"]; scale = int(d["px_per_m"]) / a.px_per_m
    low_hi = cv2.resize(low, (int(nx * scale), int(ny * scale)), interpolation=cv2.INTER_LINEAR)
    cycle = np.concatenate([det, det[::-1]], 0); cycle = np.concatenate([cycle, cycle[:, ::-1]], 1)
    reps = (int(np.ceil(low_hi.shape[0] / cycle.shape[0])), int(np.ceil(low_hi.shape[1] / cycle.shape[1])), 1)
    tex = np.clip(low_hi * np.tile(cycle, reps)[: low_hi.shape[0], : low_hi.shape[1]], 0, 255).astype(np.uint8)
tex_path = pathlib.Path(a.out).resolve().parent / "table_top.png"  # per layout, next to the photo scene
case_dir = pathlib.Path(__file__).resolve().parents[1] / "assets" / "clipper_case"

def write_scene(tex):
    cv2.imwrite(str(tex_path), tex); out = src
    out = out.replace('<headlight diffuse="0.45 0.45 0.45" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>', '<headlight diffuse="0.15 0.15 0.15" ambient="0.6 0.6 0.6" specular="0.05 0.05 0.05"/>', 1)
    out = out.replace('<global offwidth="640" offheight="480"/>', '<global offwidth="640" offheight="480"/>\n    <quality shadowsize="8192"/>', 1)
    out = out.replace('castshadow="true"', 'castshadow="false"')  # the hard black arm shadow reads as a dark patch; the real one is soft
    out = out.replace("<asset>", f'<asset>\n    <texture name="table_top_tex" type="2d" file="{tex_path}"/>\n    <material name="table_top" texture="table_top_tex" texrepeat="1 1" texuniform="false" specular="0" shininess="0"/>\n'
                      f'    <texture name="clipper_case_tex" type="2d" file="{case_dir / "clipper_case_visual_colors.png"}"/>\n    <material name="clipper_case_mat" texture="clipper_case_tex" specular="0.2" shininess="0.2"/>\n'
                      f'    <mesh name="clipper_case_textured" file="{case_dir / "clipper_case_visual_textured.obj"}"/>', 1)
    table_line = re.search(r'<geom name="table"[^>]*/>', out).group(0)
    out = out.replace(table_line, table_line + f'\n    <geom name="table_top" type="plane" size="{sx} {sy} 0.001" pos="{cx} {cy} 0.0005" material="table_top" contype="0" conaffinity="0"/>', 1)
    out = re.sub(r'<geom class="clipper_case_vis" mesh="clipper_case_visual"[^>]*/>', '<geom class="clipper_case_vis" mesh="clipper_case_textured" material="clipper_case_mat" rgba="1 1 1 1"/>', out, count=1)  # object_materials.py meshes carry the case z shift themselves
    if a.robot_xml: out = re.sub(r'<include file="[^"]+"/>', f'<include file="{a.robot_xml}"/>', out, count=1)
    if a.gain:  # the camera gain used to dim everything the wrist sees; now it lives in the colours: clutter, floor, sky, haze (table: texture)
        g = np.array(a.gain)
        def scale(m):
            vals = np.array(m.group(2).split(), float); vals[:3] *= g; return m.group(1) + '="' + " ".join(f"{v:.4f}" for v in vals) + '"'
        out = re.sub(r'\b(rgba|rgb1|rgb2|haze)="([^"]+)"', scale, out)
    if a.object_tex:
        tex_dir = pathlib.Path(a.object_tex).resolve(); assets = ""
        for body in sorted(p.stem[:-6] for p in tex_dir.glob("*_photo.obj")):
            spec = ' specular="0.3" shininess="0.4"' if body in ("clipper_case_metal", "clipper_large", "clipper_small") else ' specular="0" shininess="0"'  # brushed steel: rim/walls, clippers (object_materials.py)
            assets += f'    <texture name="{body}_photo_tex" type="2d" file="{tex_dir / f"{body}_photo.png"}"/>\n    <material name="{body}_photo_mat" texture="{body}_photo_tex"{spec}/>\n    <mesh name="{body}_photo" file="{tex_dir / f"{body}_photo.obj"}"/>\n'
            if body == "clipper_case_metal":
                case_line = re.search(r'<geom class="clipper_case_vis" mesh="clipper_case_photo"[^>]*/>', out).group(0)
                out = out.replace(case_line, case_line + '\n      <geom class="clipper_case_vis" mesh="clipper_case_metal_photo" material="clipper_case_metal_photo_mat" rgba="1 1 1 1"/>', 1)
            elif body not in ("clipper_case", "clipper_large", "clipper_small"):  # static clutter primitive (clutter_texture.py): hide it, add the world-frame textured mesh
                prim = re.search(rf'<geom name="{body}"[^>]*/>', out)
                if prim is None: continue
                placement = " ".join(m_.group(0) for m_ in re.finditer(r'\b(pos|quat)="[^"]+"', prim.group(0)))  # the primitive's own frame (free body: none)
                out = out.replace(prim.group(0), prim.group(0).replace("<geom ", '<geom group="3" ', 1) + f'\n      <geom name="{body}_photo" type="mesh" mesh="{body}_photo" material="{body}_photo_mat" group="2" contype="0" conaffinity="0" {placement}/>', 1)
            elif body == "clipper_case":
                out = re.sub(r'<geom class="clipper_case_vis" mesh="clipper_case_textured"[^>]*/>', '<geom class="clipper_case_vis" mesh="clipper_case_photo" material="clipper_case_photo_mat" rgba="1 1 1 1"/>', out, count=1)
            else:  # the box keeps the contacts (hidden, group 3); the textured box mesh is the visual
                box = re.search(rf'<body name="{body}"[^>]*>\s*<freejoint/>\s*(<geom type="box"[^>]*/>)', out).group(1)
                out = out.replace(box, box.replace("<geom ", '<geom group="3" ') + f'\n      <geom type="mesh" mesh="{body}_photo" material="{body}_photo_mat" group="2" contype="0" conaffinity="0" mass="0"/>', 1)
        out = out.replace("<asset>", "<asset>\n" + assets, 1)
    pathlib.Path(a.out).write_text(out)

# gain: rendered table pixels vs the real background where the table is visible (one linear pass).
write_scene(tex); model2 = mujoco.MjModel.from_xml_path(a.out); data2 = mujoco.MjData(model2); mujoco.mj_forward(model2, data2)
r2 = mujoco.Renderer(model2, H, W); r2.update_scene(data2, "ext"); sim = r2.render()[:, :, ::-1]; r2.enable_segmentation_rendering(); r2.update_scene(data2, "ext"); seg2 = r2.render()[:, :, 0]
vis = (seg2 == model2.geom("table_top").id) & (occluded == 0); gain = bg[vis].mean(0) / sim[vis].mean(0)
tex = np.clip(tex.astype(float) * gain * (np.array(a.gain)[::-1] if a.gain else 1.0), 0, 255).astype(np.uint8); write_scene(tex)  # texture is BGR
print(f"texture {tex.shape[1]}x{tex.shape[0]}, {bad.mean() * 100:.0f}% filled along the grain, gain BGR {gain.round(3)} on {vis.sum()} table px -> {tex_path}; scene -> {a.out}")
