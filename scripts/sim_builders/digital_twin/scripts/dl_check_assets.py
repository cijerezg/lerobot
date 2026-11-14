"""Check the assets_dl assets: settle 1 s on a floor plane (asset at z = 0.01), hole / cavity test against the hulls, renders.
usage: python -I dl_check_assets.py [names...]  -> assets_dl/<name>/<name>_sheet.png, assets_dl/check_results.json"""
import json, pathlib, sys, numpy as np, mujoco, trimesh
from PIL import Image, ImageDraw

ROOT = pathlib.Path("/home/user/Documents/Research/RL/LeRobot/assets/sim_sources/digital_twin"); OUT = ROOT / "assets_dl"; THUMBS = ROOT / "downloads/_index/thumbs"
NAMES = sys.argv[1:] or ["pill_bottle", "pump_bottle", "mug", "tape_roll", "power_strip", "glasses_case"]
results = json.load(open(OUT / "check_results.json")) if (OUT / "check_results.json").exists() else {}


def scene_xml(name):
    A = OUT / name; body = (A / f"{name}.xml").read_text().replace(f'<body name="{name}" pos="0 0 0">', f'<body name="{name}" pos="0 0 0.01">')
    inner = body.split("<mujoco>", 1)[1].rsplit("</mujoco>", 1)[0]
    return (f'<mujoco><compiler meshdir="{A}" texturedir="{A}"/><option timestep="0.002"/>'
            '<visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/><global offwidth="640" offheight="480"/></visual>'
            '<asset><texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.7 0.7 0.7" width="256" height="256"/>'
            '<material name="grid" texture="grid" texrepeat="40 40"/></asset>'
            f"{inner}".replace("<worldbody>", '<worldbody><light pos="0.3 -0.3 1.0" dir="-0.3 0.3 -1"/><geom name="floor" type="plane" size="2 2 0.01" material="grid"/>')
            + "</mujoco>")


def inside_hull(h, pt):  # convex hull: behind every face plane
    return bool(np.all(np.einsum("ij,ij->i", h.face_normals, pt - h.triangles[:, 0]) <= 1e-6))


def render(model, data, size, groups, elev, az, dist_mul=2.6):
    r = mujoco.Renderer(model, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, size[2] / 2]; cam.distance = dist_mul * max(size); cam.azimuth = az; cam.elevation = elev
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0
    for g in groups: opt.geomgroup[g] = 1
    r.update_scene(data, cam, opt); im = Image.fromarray(r.render().copy()); r.close(); return im


for name in NAMES:
    A = OUT / name; meta = json.load(open(A / f"{name}_meta.json")); size = np.array(meta["size_m"])
    model = mujoco.MjModel.from_xml_string(scene_xml(name)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    bid = model.body(name).id; p0 = data.xpos[bid].copy()
    for _ in range(500): mujoco.mj_step(model, data)
    p1 = data.xpos[bid].copy(); q = data.xquat[bid]; tilt = np.degrees(2 * np.arccos(min(1.0, abs(q[0]))))
    # hole / cavity test: is the axis point at mid height inside any hull
    hulls = [trimesh.load(str(f), force="mesh") for f in sorted(A.glob(f"{name}_hull*.obj"))]
    axis_pts = np.array([[0, 0, size[2] * f] for f in (0.25, 0.5, 0.75)])
    inside = [any(inside_hull(h, pt) for h in hulls) for pt in axis_pts]
    if name == "tape_roll":  # check out to 80 % of the inner radius
        vis = trimesh.load(str(A / f"{name}_visual.obj"), force="mesh"); r_in = np.linalg.norm(vis.vertices[:, :2], axis=1).min()
        ring = np.array([[0.8 * r_in * np.cos(t), 0.8 * r_in * np.sin(t), size[2] / 2] for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)])
        inside += [any(inside_hull(h, pt) for h in hulls) for pt in ring]; hole_note = f"inner radius {r_in*1000:.1f} mm, hole points inside a hull: {sum(inside)}/{len(inside)}"
    else: hole_note = f"axis points (25/50/75 % height) inside a hull: {inside}"
    # renders from the settled state
    im_obl = render(model, data, size, [0, 2], -30, 135); im_top = render(model, data, size, [0, 2], -89, 90); im_hul = render(model, data, size, [0, 3], -30, 135)
    im_hul_top = render(model, data, size, [3], -89, 90, 1.8)
    tiles = [("visual oblique", im_obl), ("visual top", im_top), ("hulls oblique", im_hul), ("hulls top (no floor)", im_hul_top)]
    th = THUMBS / f"{meta['thumb']}.jpg" if meta.get("thumb") else None
    if th is None and "thumb" not in meta:
        th = THUMBS / (meta["source_file"].split("/")[2] + ".jpg")
    if th is not None and th.exists():
        t = Image.open(th).convert("RGB"); t = t.resize((640, int(640 * t.size[1] / t.size[0]))); bg = Image.new("RGB", (640, 480), (30, 30, 30)); bg.paste(t, (0, (480 - t.size[1]) // 2)); tiles.append(("source thumbnail", bg))
    for n, im in tiles: ImageDraw.Draw(im).text((8, 8), f"{name}: {n}", fill=(255, 40, 40))
    sheet = Image.new("RGB", (640 * len(tiles), 480)); [sheet.paste(im, (640 * i, 0)) for i, (_, im) in enumerate(tiles)]
    sheet.save(A / f"{name}_sheet.png"); im_obl.save(A / f"{name}_render_oblique.png"); im_hul.save(A / f"{name}_render_hulls.png")
    results[name] = {"drift_xy_mm": float(np.linalg.norm((p1 - p0)[:2]) * 1000), "rest_height_mm": float(p1[2] * 1000), "tilt_deg": float(tilt),
                     "hole_or_cavity": hole_note, "sheet": str(A / f"{name}_sheet.png")}
    print(name, results[name])
json.dump(results, open(OUT / "check_results.json", "w"), indent=1)
