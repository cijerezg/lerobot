"""Approach angle for every diverse-corpus grasp atom where the state gives the tool orientation.

Franka (droid, droid_success): Panda DH FK on the 7 joints (hand TCP), tool z vs down.
FMB: tcp_pose quaternion (xyzw, hand TCP). MolmoAct: end-effector Euler rx, ry (R[2,2] =
cos rx cos ry for roll-pitch-yaw). UR5 / UR7e: standard UR DH orientation chain, tool z.
ARX5 / YAM: no model here -> NaN, classified visually from sheets.

Outputs: diverse_angles.jsonl, diverse_angles.txt, sheets/calib__<group>.jpg (12 atoms spread
over the angle range, to check each convention by eye), sheets/arx5__<object>_<k>.jpg and
sheets/yam__all.jpg (every atom, for the visual count). Nothing annotated.
"""
import json, sys
from pathlib import Path
import numpy as np, cv2

from lerobot.annotation.precision_contact import sample_grasps as S
from lerobot.annotation.precision_contact.panda_fk import fk_batch, quat_to_rot

OUT = S.OUT; SHEETS = S.SHEETS


def ur_rot(q6):
    """Tool rotation for the UR family (standard DH alphas); link lengths do not affect orientation."""
    alphas = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]; R = np.eye(3)
    for th, al in zip(q6, alphas):
        cz, sz, cx, sx = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        R = R @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return R


def down_angle(R22):
    return float(np.degrees(np.arccos(np.clip(-R22, -1, 1))))


def angle_for(rec):
    d = Path(rec["root"]) / rec["_ep"].get("directory", f"episodes/{rec['episode']}"); c = rec["commit"]; src = rec["source"]; sub = rec.get("sub_source")
    if src == "fmb":
        q = np.load(d / "tcp_pose.npy", mmap_mode="r")[c, 3:7]; return down_angle(quat_to_rot(np.asarray(q, float), "xyzw")[2, 2])
    st = np.load(d / "state.npy", mmap_mode="r")[c].astype(float)
    if sub in ("droid", "droid_success"): return down_angle(fk_batch(st[None, :7])[0, 2, 2])
    if sub == "molmoact": return down_angle(np.cos(st[3]) * np.cos(st[4]))
    if rec["embodiment"] in ("UR5", "UR7e"): return down_angle(ur_rot(st[:6])[2, 2])
    return float("nan")


def label(img, text):
    img = cv2.resize(img, S.TILE); strip = np.zeros((22, S.TILE[0], 3), np.uint8)
    cv2.putText(strip, text[:52], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def sheet(recs, path):
    cells = []
    for r in recs:
        a, b = S.diverse_tiles(r)
        ang = f"{r['approach_deg']:.0f}deg" if np.isfinite(r["approach_deg"]) else "n/a"
        cells.append(np.hstack([label(a, f"{r['embodiment']} {r['episode'][:26]} a{r['atom']}"), label(b, f"{r['object'][:22]} | {ang} | {r['sub_source']}")]))
    w, h = cells[0].shape[1], cells[0].shape[0]
    rows = [np.hstack(cells[i:i + 3] + [np.zeros((h, w, 3), np.uint8)] * (3 - len(cells[i:i + 3]))) for i in range(0, len(cells), 3)]
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    recs = S.diverse("outputs/diverse_robot_dataset_v3/corpus", "diverse") + S.diverse("outputs/diverse_robot_dataset_v3/fmb", "fmb")
    for r in recs: r["approach_deg"] = angle_for(r); r["group"] = r["sub_source"] if r["sub_source"] in ("droid", "droid_success", "molmoact", "fmb") else r["embodiment"]
    with open(OUT / "diverse_angles.jsonl", "w") as f:
        for r in recs: f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")

    # calibration sheets: 12 atoms spread over the angle range per kinematic group
    for g in ("droid", "droid_success", "molmoact", "fmb", "UR5"):
        rs = sorted([r for r in recs if r["group"] == g and np.isfinite(r["approach_deg"])], key=lambda r: r["approach_deg"])
        if not rs: continue
        idx = np.unique(np.linspace(0, len(rs) - 1, 12).round().astype(int))
        sheet([rs[i] for i in idx], SHEETS / f"calib__{g}.jpg"); print("calib", g, len(rs), flush=True)

    # every ARX5 / YAM atom, grouped by object, for the visual count
    for emb in ("ARX5", "YAM"):
        rs = sorted([r for r in recs if r["embodiment"] == emb], key=lambda r: (r["object"], r["episode"], r["atom"]))
        for k in range(0, len(rs), 12):
            chunk = rs[k:k + 12]; sheet(chunk, SHEETS / f"{emb.lower()}__{k // 12:02d}.jpg")
        print(emb, len(rs), "atoms ->", (len(rs) + 11) // 12, "sheets", flush=True)

    # numbers: per group x object, n / <45 / >=45 / median
    with open(OUT / "diverse_angles.txt", "w") as f:
        for g in ("droid", "droid_success", "molmoact", "fmb", "UR5", "UR7e", "ARX5", "YAM"):
            rs = [r for r in recs if r["group"] == g]
            a = np.array([r["approach_deg"] for r in rs]); fin = a[np.isfinite(a)]
            f.write(f"== {g}: {len(rs)} grasp atoms, {len(fin)} with angle; <45: {(fin < 45).sum()}  >=45: {(fin >= 45).sum()}  median {np.median(fin) if len(fin) else float('nan'):.1f}\n")
            objs = {}
            for r in rs: objs.setdefault(r["object"], []).append(r["approach_deg"])
            for o, v in sorted(objs.items(), key=lambda kv: -len(kv[1])):
                v = np.array(v); fin = v[np.isfinite(v)]
                if len(fin): f.write(f"   {o:32s} n={len(v):3d} <45={int((fin < 45).sum()):3d} >=45={int((fin >= 45).sum()):3d} median={np.median(fin):5.1f}\n")
                else: f.write(f"   {o:32s} n={len(v):3d} (visual)\n")
    print(open(OUT / "diverse_angles.txt").read())


if __name__ == "__main__":
    main()
