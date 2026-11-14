"""Appearance calibration on a COPY of the twin scene (twin/appearance/scene_test.xml, never scene_twin.xml / room.xml).
Each iteration: build scene_test.xml (+ room_test.xml = room.xml with the fitted wall rgba and the plank floor material), render ext +
wrist at frames 0/300/2400 with the robot posed from the recorded states, segment desk / wall / floor pixels, compare their trimmed
colour statistics with fit_real.json, and update the desk texture affine, the wall rgba and the floor-texture tone through the
rendered/texture lighting ratio (rendered = L * texture per channel, so texture_target = real / L). State in params.json.
Run from the LeRobot root: MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/digital_twin/scripts/appear_calibrate.py [--iters N] [--compose]"""
import sys, re, json, pathlib, shutil, numpy as np, cv2, mujoco, pandas as pd
sys.path.insert(0, "lerobot/src"); from lerobot.probes.action_trace_sim import to_qpos
ROOT = pathlib.Path("lerobot/scripts/sim_builders/digital_twin"); SRC = pathlib.Path("assets/sim_sources/digital_twin"); APP = ROOT / "twin/appearance"; TWIN = ROOT / "twin"
SCENES = pathlib.Path("lerobot/src/lerobot/robots/rebot_b601_sim/scenes"); TEX0 = SCENES / "table_top.png"
LAM = SRC / "downloads/textures/laminate_floor_02/laminate_floor_02_diff_1k.jpg"
DS = pathlib.Path("outputs/rebot_teleop_2026-10-29-cut-v1"); FRAMES = [0, 300, 2400]
REAL = json.load(open(APP / "fit_real.json")); PARAMS = APP / "params.json"
states = np.stack(pd.read_parquet(DS / "data/chunk-000/file-000.parquet", columns=["observation.state"])["observation.state"].to_numpy())
iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 1
p = json.load(open(PARAMS)) if PARAMS.exists() else dict(desk_gain=[1, 1, 1], desk_offset=[0, 0, 0], wall_rgb=[0.75, 0.75, 0.75], floor_gain=[1, 1, 1], floor_offset=[0, 0, 0])

def trim(px):
    lum = px @ np.array([0.114, 0.587, 0.299]); lo, hi = np.percentile(lum, [25, 95]); return px[(lum >= lo) & (lum <= hi)]
def write_textures():
    t = cv2.imread(str(TEX0)).astype(float)                        # BGR
    g, o = np.array(p["desk_gain"])[::-1], np.array(p["desk_offset"])[::-1]
    cv2.imwrite(str(APP / "table_top_fit.png"), np.clip(t * g + o, 0, 255).astype(np.uint8))
    lam = cv2.imread(str(LAM)).astype(float); g, o = np.array(p["floor_gain"])[::-1], np.array(p["floor_offset"])[::-1]
    cv2.imwrite(str(APP / "floor_planks.png"), np.clip(lam * g + o, 0, 255).astype(np.uint8))
def build_scene():
    room = (ROOT / "room/room.xml").read_text()
    room = re.sub(r'(<geom name="room_floor"[^>]*?) rgba="[^"]*"', r'\1 material="room_floor_mat"', room)
    wall = " ".join(f"{v:.3f}" for v in p["wall_rgb"]) + " 1"
    room = re.sub(r'(<geom name="room_wall_[a-z]+"[^>]*?) rgba="[^"]*"', lambda m: f'{m.group(1)} rgba="{wall}"', room)
    (APP / "room_test.xml").write_text(room)
    s = (TWIN / "scene_twin.xml").read_text()
    s = s.replace('assetdir="../../../../src', 'assetdir="../../../../../src').replace('<include file="../../../../src', '<include file="../../../../../src')
    s = s.replace('<include file="../room/room.xml" />', '<include file="room_test.xml" />')
    s = re.sub(r"<visual>.*?</visual>", '<include file="lights_visual.xml" />', s, flags=re.S)
    s = s.replace('file="table_top.png"', 'file="../../../../../scripts/sim_builders/digital_twin/twin/appearance/table_top_fit.png"')
    s = s.replace("<asset>", '<include file="materials.xml" />\n  <asset>', 1)
    s = re.sub(r'\n\s*<light [^>]*/>', "", s)                         # drop detail_key / detail_fill / soft_shadow_*
    s = s.replace("</worldbody>", '  <include file="lights.xml" />\n  </worldbody>')
    (APP / "scene_test.xml").write_text(s)
def render_all(tag):
    m = mujoco.MjModel.from_xml_path(str(APP / "scene_test.xml")); d = mujoco.MjData(m); r = mujoco.Renderer(m, 480, 640)
    gid = {k: m.geom(k).id for k in ["room_desk_top", "room_wall_front", "room_floor"]}; out = {}
    for fr in FRAMES:
        d.qpos[:8] = to_qpos(states[fr]); mujoco.mj_forward(m, d)
        for cam in ["ext", "wrist"]:
            r.disable_segmentation_rendering(); r.update_scene(d, cam); img = np.ascontiguousarray(r.render()[:, :, ::-1])
            r.enable_segmentation_rendering(); r.update_scene(d, cam); seg = r.render()[:, :, 0]
            out[(fr, cam)] = (img, {k: seg == v for k, v in gid.items()})
            cv2.imwrite(str(APP / f"{tag}_{'top' if cam == 'ext' else 'wrist'}_f{fr}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out
def measure(out):
    regions = {"desk": out[(0, "ext")][0][out[(0, "ext")][1]["room_desk_top"]], "desk_wrist_ref": out[(0, "wrist")][0][out[(0, "wrist")][1]["room_desk_top"]], "wall": out[(0, "wrist")][0][out[(0, "wrist")][1]["room_wall_front"]],
               "floor": out[(2400, "wrist")][0][out[(2400, "wrist")][1]["room_floor"]]}
    res = {}
    for k, px in regions.items():
        px = trim(px.reshape(-1, 3).astype(float)); res[k] = dict(mean_rgb=px[:, ::-1].mean(0).round(2).tolist(), std_rgb=px[:, ::-1].std(0).round(2).tolist(), n=int(len(px)))
    return res
def update(sim):
    tex = cv2.imread(str(APP / "table_top_fit.png")).astype(float)[:, :, ::-1].reshape(-1, 3); tex0 = cv2.imread(str(TEX0)).astype(float)[:, :, ::-1].reshape(-1, 3)
    L = np.array(sim["desk"]["mean_rgb"]) / tex.mean(0)                                  # lighting factor on the desk
    lum = np.array([0.299, 0.587, 0.114])                                                # ONE gain (luminance grain contrast, within-patch real std), per-channel offsets
    a = np.clip(float(np.array(REAL["desk"]["std_within_rgb"]) @ lum / (np.array(sim["desk"]["std_rgb"]) @ lum)) * p["desk_gain"][0], 0.6, 1.6) * np.ones(3)
    b = np.array(REAL["desk"]["mean_rgb"]) / L - a * tex0.mean(0)
    if "--mult" in sys.argv: a = np.array(REAL["desk"]["mean_rgb"]) / (L * tex0.mean(0)); b = np.zeros(3)   # gain-only variant (keeps grain streaks warmer than the background)
    p["desk_gain"], p["desk_offset"], p["desk_L"] = a.round(4).tolist(), b.round(2).tolist(), L.round(3).tolist()
    Lw = np.array(sim["wall"]["mean_rgb"]) / (255 * np.array(p["wall_rgb"])); w = np.array(REAL["wall"]["mean_rgb"]) / (255 * Lw)
    p["wall_rgb"], p["wall_L"], p["wall_rgb_unclipped"] = np.clip(w, 0, 1).round(4).tolist(), Lw.round(3).tolist(), w.round(4).tolist()
    fl = cv2.imread(str(APP / "floor_planks.png")).astype(float)[:, :, ::-1].reshape(-1, 3); lam = cv2.imread(str(LAM)).astype(float)[:, :, ::-1].reshape(-1, 3)
    Lf = np.array(sim["floor"]["mean_rgb"]) / fl.mean(0)
    a = np.clip(float(np.array(REAL["floor"]["std_rgb"]) @ lum / (np.array(sim["floor"]["std_rgb"]) @ lum)) * p["floor_gain"][0], 0.5, 1.5) * np.ones(3)
    b = np.array(REAL["floor"]["mean_rgb"]) / Lf - a * lam.mean(0)
    p["floor_gain"], p["floor_offset"], p["floor_L"] = a.round(4).tolist(), b.round(2).tolist(), Lf.round(3).tolist()
def err(sim):
    return {k: dict(abs_mean_diff_rgb=np.abs(np.array(REAL[k]["mean_rgb"]) - np.array(sim[k]["mean_rgb"])).round(1).tolist(),
                    mean_abs=float(np.abs(np.array(REAL[k]["mean_rgb"]) - np.array(sim[k]["mean_rgb"])).mean().round(2))) for k in ["desk", "wall", "floor", "desk_wrist_ref"]}
def before():
    """Same measurement on the ORIGINAL twin scene (old lights, old texture, old room colours) = the 'before' numbers."""
    m = mujoco.MjModel.from_xml_path(str(TWIN / "scene_twin.xml")); d = mujoco.MjData(m); r = mujoco.Renderer(m, 480, 640)
    gid = {k: m.geom(k).id for k in ["room_desk_top", "room_wall_front", "room_floor"]}; out = {}
    for fr, cam in [(0, "ext"), (0, "wrist"), (2400, "wrist")]:
        d.qpos[:8] = to_qpos(states[fr]); mujoco.mj_forward(m, d)
        r.disable_segmentation_rendering(); r.update_scene(d, cam); img = np.ascontiguousarray(r.render()[:, :, ::-1])
        r.enable_segmentation_rendering(); r.update_scene(d, cam); seg = r.render()[:, :, 0]; out[(fr, cam)] = (img, {k: seg == v for k, v in gid.items()})
    sim = measure(out); res = dict(sim=sim, err=err(sim)); json.dump(res, open(APP / "fit_before.json", "w"), indent=1); print("before:", json.dumps(res)); return res
if __name__ == "__main__":
    if "--before" in sys.argv: before()
    for it in range(iters):
        write_textures(); build_scene(); out = render_all("sim"); sim = measure(out); e = err(sim)
        print(f"iter {it}: sim {json.dumps({k: v['mean_rgb'] for k, v in sim.items()})} err {json.dumps({k: v['mean_abs'] for k, v in e.items()})}")
        json.dump(dict(sim=sim, err=e, params=dict(p)), open(APP / "fit_result.json", "w"), indent=1)
        if "--no-update" not in sys.argv: update(sim); json.dump(p, open(PARAMS, "w"), indent=1)
    if "--compose" in sys.argv:
        def label(im, t):
            cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); return im
        for fr in FRAMES:
            rows = []
            for view, realp in [("top", TWIN / "renders" / f"real_top_f{fr}.jpg"), ("wrist", ROOT / "room/renders" / f"real_wrist_f{fr}.png")]:
                before = cv2.imread(str(TWIN / "renders" / f"sim_{view}_f{fr}.jpg")); after = cv2.imread(str(APP / f"sim_{view}_f{fr}.jpg")); real = cv2.imread(str(realp))
                rows.append(np.hstack([label(before, f"before {view} f{fr}"), label(after, f"after {view} f{fr}"), label(real, f"real {view} f{fr}")]))
            cv2.imwrite(str(APP / f"compare_f{fr}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("composed")
