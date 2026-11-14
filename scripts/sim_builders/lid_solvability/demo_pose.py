"""Where the case sits in the T071 demos (ep 0 frame 0 primary; ep 2, 3 for spread), in sim table coordinates, through the
calibrated top camera of the sim (camera `ext` of close_lid/scene.xml). Writes demo_pose.json and frames/demo_pose_*.jpg."""
import json
from pathlib import Path
import cv2, mujoco, numpy as np

WS = Path(__file__).resolve().parents[4]; OUT = Path(__file__).resolve().parent
SCENE = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml"
m = mujoco.MjModel.from_xml_path(str(SCENE)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
cam = m.camera("ext").id; W_, H_ = 640, 480
f = 0.5 * H_ / np.tan(np.deg2rad(m.cam_fovy[cam]) / 2); cx, cy = W_ / 2, H_ / 2
Rc, oc = d.cam_xmat[cam].reshape(3, 3), d.cam_xpos[cam]


def pix_to_plane(u, v, z):
    ray = Rc @ np.array([(u - cx) / f, -(v - cy) / f, -1.0]); t = (z - oc[2]) / ray[2]; return oc + t * ray


def world_to_pix(p):
    c = Rc.T @ (np.asarray(p) - oc); return np.array([cx + f * c[0] / -c[2], cy - f * c[1] / -c[2]])


report = {"camera": dict(pos=oc.round(4).tolist(), fovy_deg=float(m.cam_fovy[cam]), f_px=float(f))}
# sanity: the sim case's known centre vs its silhouette centroid in the sim render
r = mujoco.Renderer(m, H_, W_); r.enable_segmentation_rendering(); r.update_scene(d, "ext"); seg = r.render()[:, :, 0]; r.disable_segmentation_rendering(); r.close()
case_px = np.argwhere((seg >= 0) & np.isin(m.geom_bodyid[np.clip(seg, 0, None)], [m.body("headphone_case").id]) & (seg != m.geom("hinged_nail_ep2_21").id))
v, u = case_px.mean(0); base_est = pix_to_plane(u, v, 0.02)
report["sim_check"] = dict(base_centre_xml=[0.41892, 0.0174], base_centre_from_silhouette=base_est[:2].round(4).tolist(), px=[round(u, 1), round(v, 1)])
print("sim check:", report["sim_check"])

# table rectangle (geom `table`) projected, as the search region; the robot's reserved strip near the base excluded
tb = m.geom("table").id; tc, th = d.geom_xpos[tb], m.geom_size[tb]
corners = [world_to_pix([tc[0] + sx * th[0], tc[1] + sy * th[1], 0.0]) for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
poly = np.array(corners, np.int32)


def segment_case(img, name):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); mask = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(mask, [poly], 1)
    # the case fabric is dark and bluish (V 30-95, H ~105); the wood is V ~175, H ~20; the white box V ~230
    dark = ((hsv[:, :, 2] < 120) & (mask > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark)
    order = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    if len(order) > 1 and stats[order[1], cv2.CC_STAT_AREA] > 0.4 * stats[order[0], cv2.CC_STAT_AREA]:
        a, b = (np.argwhere(lab == k)[:, ::-1].astype(float) for k in order[:2]); pts = np.vstack([a, b])
    else:  # one blob through the hinge: 2-means along its principal axis
        pts = np.argwhere(lab == order[0])[:, ::-1].astype(float)
        mean = pts.mean(0); ev, evec = np.linalg.eigh(np.cov((pts - mean).T)); ax = evec[:, 1]
        proj = (pts - mean) @ ax; a, b = pts[proj < 0], pts[proj >= 0]
    for _ in range(10):  # 2-means along the blob
        ca, cb = a.mean(0), b.mean(0); da = np.linalg.norm(pts - ca, axis=1); db = np.linalg.norm(pts - cb, axis=1); a, b = pts[da < db], pts[da >= db]
    halves = []
    for h in (a, b):
        rect = cv2.minAreaRect(h.astype(np.float32)); halves.append(dict(px=h.mean(0).tolist(), area=len(h), rect=rect))
    vis = img.copy(); cv2.polylines(vis, [poly], True, (0, 255, 0), 1)
    for h, col in zip(halves, ((255, 0, 0), (0, 0, 255))):
        cv2.drawContours(vis, [cv2.boxPoints(h["rect"]).astype(np.int32)], 0, col, 2); cv2.circle(vis, tuple(int(x) for x in h["px"]), 4, col, -1)
    return halves, vis


def measure(path, name):
    img = cv2.imread(str(path)); halves, vis = segment_case(img, name)
    # which half is the loaded base: the one with the smaller dark area? no: both are dark. Use the sheet: the lid lies toward the robot
    # (smaller image row = nearer the robot). The lid is the half whose centroid is nearer the robot base pixel (robot at the origin).
    base_px = world_to_pix([0, 0, 0.1])
    halves.sort(key=lambda h: np.linalg.norm(np.array(h["px"]) - base_px))
    lid_h, base_h = halves  # lid nearer the robot
    Z = 0.02  # shells are 4 cm tall; the silhouette centroid sits mid-height
    lid_w = pix_to_plane(*lid_h["px"], Z); base_w = pix_to_plane(*base_h["px"], Z)
    v_bl = lid_w[:2] - base_w[:2]; dist = float(np.linalg.norm(v_bl)); ang = float(np.degrees(np.arctan2(v_bl[1], v_bl[0])))
    # sim convention: base->lid direction is world -x (180 deg) at body_yaw 242.18
    body_yaw = (242.18 + (ang - 180.0)) % 360
    hinge = (base_w[:2] + lid_w[:2]) / 2
    for p, col in ((lid_w, (255, 0, 0)), (base_w, (0, 0, 255)), (np.r_[hinge, 0.03], (0, 255, 255))):
        cv2.circle(vis, tuple(int(x) for x in world_to_pix(np.r_[p[:2], Z])), 6, col, 1)
    cv2.putText(vis, f"{name}: base ({base_w[0]:.3f},{base_w[1]:.3f}) lid ({lid_w[0]:.3f},{lid_w[1]:.3f}) base->lid {ang:.0f}deg", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(OUT / f"frames/demo_pose_{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    ppm = f / (oc[2] - Z)  # px per m at the shells' mid height
    out = dict(base_xy=base_w[:2].round(4).tolist(), lid_xy=lid_w[:2].round(4).tolist(), hinge_xy=hinge.round(4).tolist(), base_to_lid_deg=round(ang, 1),
               lid_shell_size_m=[round(v / ppm, 3) for v in sorted(lid_h["rect"][1], reverse=True)], base_shell_size_m=[round(v / ppm, 3) for v in sorted(base_h["rect"][1], reverse=True)],
               sim_shell_size_m=[0.186, 0.151], sim_centre_distance_m=0.158,
               body_yaw_deg_sim_convention=round(body_yaw, 1), centre_distance_m=round(dist, 4), lid_px=[round(x, 1) for x in lid_h["px"]], base_px=[round(x, 1) for x in base_h["px"]],
               lid_rect_deg=round(lid_h["rect"][2], 1), base_rect_deg=round(base_h["rect"][2], 1), areas_px=[lid_h["area"], base_h["area"]])
    print(name, json.dumps(out)); return out


for ep in (0, 2, 3):
    report[f"t071_ep{ep}_f0"] = measure(OUT / f"frames/t071_ep{ep}_top_f0.png", f"ep{ep}")
# camera consistency 10-04 vs 10-16: table edge overlay (the 10-16 background of the scene vs the 10-04 frame)
bg = cv2.imread(str(SCENE.parent / "background_top.png")); fr = cv2.imread(str(OUT / "frames/t071_ep0_top_f0.png"))
blend = cv2.addWeighted(bg, 0.5, fr, 0.5, 0); cv2.polylines(blend, [poly], True, (0, 255, 0), 1)
e1 = cv2.Canny(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), 60, 150); e2 = cv2.Canny(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 60, 150)
edges = np.dstack([e1, np.zeros_like(e1), e2])
cv2.imwrite(str(OUT / "frames/camera_check_blend.jpg"), np.hstack([blend, edges]), [cv2.IMWRITE_JPEG_QUALITY, 85])
# table-edge shift: the long table edges in both images via Hough on the lower half (table far edge) and sides
def table_edge_rows(gray):
    # the far table edge: strongest horizontal gradient row within the projected table bottom band
    g = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3); prof = np.abs(g[:, 200:440]).mean(1); band = slice(int(poly[:, 1].max()) - 40, int(poly[:, 1].max()) + 40)
    return int(band.start + np.argmax(prof[band]))
def table_edge_cols(gray):
    g = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3); prof = np.abs(g[150:400, :]).mean(0)
    l = slice(max(0, int(poly[:, 0].min()) - 40), int(poly[:, 0].min()) + 40); rr = slice(int(poly[:, 0].max()) - 40, min(639, int(poly[:, 0].max()) + 40))
    return int(l.start + np.argmax(prof[l])), int(rr.start + np.argmax(prof[rr]))
gb, gf = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
report["camera_check"] = dict(projected_table_corners_px=poly.tolist(), far_edge_row_bg_vs_demo=[table_edge_rows(gb), table_edge_rows(gf)], side_edge_cols_bg_vs_demo=[table_edge_cols(gb), table_edge_cols(gf)], px_per_m_on_table=float(f / (oc[2] - 0.0)))
print("camera check:", report["camera_check"])
json.dump(report, open(OUT / "demo_pose.json", "w"), indent=1)
