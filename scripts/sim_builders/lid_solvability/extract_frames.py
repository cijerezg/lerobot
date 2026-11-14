"""Frames: policy close_lid ep0 (top + wrist) at several times; T071 ep0/2/3 top at frame 0 and ep0 mid-grasp; sim renders."""
import cv2, numpy as np, mujoco
from pathlib import Path
WS = Path(__file__).resolve().parents[4]; OUT = Path(__file__).resolve().parent / "frames"
def grab(path, t_s):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000); ok, f = cap.read(); cap.release(); return f
pol = WS / "outputs/rebot-mix40_20_40-resume1200-20261027/validation/step_00001600/sim/close_lid/videos"
rows = []
for t in [0, 15, 25, 30, 33, 36]:
    top = grab(pol / "ep0_top.mp4", t); wr = grab(pol / "ep0_wrist.mp4", t)
    for im, lab in ((top, f"top t={t}s"), (wr, f"wrist t={t}s")):
        cv2.putText(im, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    rows.append(np.hstack([top, wr]))
cv2.imwrite(str(OUT / "policy_ep0_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
demo = WS / "outputs/rebot_additions_2026-10-04-annotated-v12-tasks1/videos/observation.images.top/chunk-000"
demo_w = WS / "outputs/rebot_additions_2026-10-04-annotated-v12-tasks1/videos/observation.images.wrist/chunk-000"
for ep, fi in ((0, 0), (2, 1), (3, 2)):
    cv2.imwrite(str(OUT / f"t071_ep{ep}_top_f0.png"), grab(demo / f"file-{fi:03d}.mp4", 0.0))
sheet = []
for fr in [0, 300, 450, 510, 600, 700, 813]:
    t = fr / 30
    top = grab(demo / "file-000.mp4", t); wr = grab(demo_w / "file-000.mp4", t)
    cv2.putText(top, f"T071 ep0 f{fr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    sheet.append(np.hstack([top, wr]))
cv2.imwrite(str(OUT / "t071_ep0_sheet.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 80])
# sim: ext camera with visual (group 2) and with collision (group 3) geoms, plus close-ups
m = mujoco.MjModel.from_xml_path(str(WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml")); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
def render(cam, groups):
    opt = mujoco.MjvOption();
    for g in range(6): opt.geomgroup[g] = 1 if g in groups else 0
    r.update_scene(d, cam, opt); return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [0.36, 0.017, 0.02]; cam.distance = 0.45; cam.azimuth = 180; cam.elevation = -40
cam2 = mujoco.MjvCamera(); cam2.type = mujoco.mjtCamera.mjCAMERA_FREE; cam2.lookat[:] = [0.36, 0.017, 0.02]; cam2.distance = 0.45; cam2.azimuth = 90; cam2.elevation = -35
a = np.hstack([render("ext", {2}), render("ext", {3})]); b = np.hstack([render(cam, {2}), render(cam, {3})]); c = np.hstack([render(cam2, {2}), render(cam2, {3})])
cv2.imwrite(str(OUT / "sim_visual_vs_collision.jpg"), np.vstack([a, b, c]), [cv2.IMWRITE_JPEG_QUALITY, 85])
r.close(); print("done")
