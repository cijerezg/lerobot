"""Object-choice scenes (2026-11-17): the basket and TWO objects on the twin desk; the task names one of them.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/object_choice/build_choice_envs.py --check
    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/object_choice/build_choice_envs.py

Question: does the arm go to the object the instruction names? Three pairs: pill bottle + orange bottle (both in the training data),
pill bottle + banana, pill bottle + mug (banana and mug are not in the training data; build_assets.py). Six placements per pair in
three swapped couples: one object on each side (p1, p2 = the two objects exchanged), both on the robot's right (p3, p4 = exchanged),
both on the robot's left (p5, p6 = exchanged). So an arm that always goes to the same place, the same side or the nearer object picks
the named object in exactly half of a couple. Every placement is written twice, once per instruction:
scenes/choice_<pair>_ask_<object>_p<k> (same scene.xml, tasks.py tier "choice").

Built with the twin eval tools (../twin_eval_v2, ../digital_twin/envs): same desk, room, camera, checks (2 cm clearances, no start
contact, IK reach of both objects and the basket, 1 s settle). objects.json adds, per object, the side seen from the robot
(robot at the origin facing +x, z up: left = +y) and the side in the top camera image (object centre projected into `ext`).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
WS = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "twin_eval_v2"))
import build_twin_envs_v2 as V2  # noqa: E402  (the reach rule and the tilted pinch retry on top of v1)

V1 = V2.V1
SCENES = V1.SCENES
ASSETS = SCENES / "assets/choice"
RENDERS = HERE / "renders"
NEW = {"banana": 90.0, "mug": 0.0}  # the added objects and their yaw (banana: long axis toward the robot, the pinch goes across it)
SHORT = {"pill_bottle": "pill", "roar_bottle": "roar", "banana": "banana", "mug": "mug"}
PAIRS = {"pill_roar": ("pill_bottle", "roar_bottle"), "pill_banana": ("pill_bottle", "banana"), "pill_mug": ("pill_bottle", "mug")}
# (note, basket, first spot, second spot); spots are (x, y). The odd placement puts object A on the first spot, the even one object B.
COUPLES = [
    ("one on each side", (0.47, 0.03, 0), (0.25, 0.22), (0.25, -0.22)),
    ("both on the robot's right", (0.36, 0.26, 0), (0.21, -0.26), (0.40, -0.17)),
    ("both on the robot's left", (0.36, -0.26, 0), (0.21, 0.26), (0.40, 0.17)),
]


def add_objects(twin):
    """The banana and the mug as twin bodies (asset lines, default classes, body block, a pose off to the side)."""
    for i, name in enumerate(NEW):
        meta = json.loads((ASSETS / name / f"{name}.json").read_text())
        twin.asset_lines += meta["asset_lines"]
        twin.default = twin.default.replace("  </default>\n", meta["default"] + "  </default>\n", 1) if i == 0 else twin.default.replace(
            f'    <default class="{list(NEW)[0]}_col">', meta["default"] + f'    <default class="{list(NEW)[0]}_col">', 1)
        twin.bodies[name] = meta["body"]
        twin.pose[name] = (np.array([0.30, 0.20 - 0.40 * i, 0.0020]), V1.quat_z(0.0))


def sides(folder, bodies):
    """Per body: xy, the side seen from the robot (left = +y) and the pixel / side of its centre in the top camera image."""
    m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml")); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = m.camera("ext").id; R = d.cam_xmat[cam].reshape(3, 3); W, H = 640, 480
    f = 0.5 * H / np.tan(np.radians(m.cam_fovy[cam]) / 2)
    out = {}
    for b in bodies:
        p = d.xipos[m.body(b).id]; c = R.T @ (p - d.cam_xpos[cam])  # camera frame: x right, y up, looking along -z
        u, v = W / 2 + f * c[0] / -c[2], H / 2 - f * c[1] / -c[2]
        out[b] = dict(xy=[round(float(p[0]), 3), round(float(p[1]), 3)], side_robot="left" if p[1] > 0 else "right",
                      image_uv=[round(float(u), 1), round(float(v), 1)], side_image="left" if u < W / 2 else "right")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS))
    ap.add_argument("--check", action="store_true", help="validate every placement, leave no scene folder in the package")
    args = ap.parse_args()
    twin = V1.TwinText(); add_objects(twin)
    rules = dict(margin_m=V1.MARGIN_M, settle_s=V1.SETTLE_S, settle_tol_m=V1.SETTLE_TOL_M, start_contact_tol_m=V1.PEN_TOL_M)
    RENDERS.mkdir(exist_ok=True)
    failures, rng = [], np.random.default_rng(0)  # rng unused: no jitter
    for pair in args.pairs:
        a, b = PAIRS[pair]
        bodies, tiles = ["basket", a, b], []
        for k in range(1, 7):
            note, basket, first, second = COUPLES[(k - 1) // 2]
            on_first, on_second = (a, b) if k % 2 else (b, a)
            base = {"basket": basket, on_first: (*first, NEW.get(on_first, 0.0)), on_second: (*second, NEW.get(on_second, 0.0))}
            name = f"choice_{pair}_p{k}"
            folder = SCENES / f"_check_{name}"; folder.mkdir(exist_ok=True)
            (folder / "scene.xml").write_text(twin.scene_xml(f"scene_{name}", bodies, {}))
            scene = V1.Scene(folder / "scene.xml", bodies)
            spec = dict(bodies=bodies, targets=[a, b], base={n: tuple(float(v) for v in p) for n, p in base.items()}, jitter=[])
            poses, info = V1.place(twin, scene, spec, bodies, rng, jitter=False)
            chk = None
            if poses is not None:
                (folder / "scene.xml").write_text(twin.scene_xml(f"scene_{name}", bodies, poses))
                chk = V1.settle(folder, bodies)
            if poses is None or chk["max_moved_mm"] > V1.SETTLE_TOL_M * 1e3:
                info = info if poses is None else dict(failed="settle", moved_mm=chk["moved_mm"])
                failures.append(name); print(f"  {name}: FAIL {json.dumps(info, default=str)[:700]}")
                shutil.rmtree(folder)
                continue
            side = sides(folder, [a, b])
            info["targets"] = [a, b]
            where = ", ".join(f"{SHORT[n]} robot-{s['side_robot']} image-{s['side_image']} u={s['image_uv'][0]:.0f}" for n, s in side.items())
            print(f"  {name}: ok, settle max {chk['max_moved_mm']:.1f} mm, start contact {info['pen_mm']:.1f} mm; {where}")
            if not args.check:
                imgs = V1.render(folder, cams=("ext", "wrist"))
                for cam, img in imgs.items():
                    Image.fromarray(img).save(RENDERS / f"{name}_{cam}.png")
                tiles += [(f"p{k}: {note}, {SHORT[on_first]} first spot", imgs["ext"]), (f"p{k} wrist", imgs["wrist"])]
                for asked, other in ((a, b), (b, a)):
                    out = SCENES / f"choice_{pair}_ask_{SHORT[asked]}_p{k}"; out.mkdir(exist_ok=True)
                    shutil.copyfile(folder / "scene.xml", out / "scene.xml")
                    meta = dict(config=f"choice_{pair}_ask_{SHORT[asked]}", placement=k, tier="choice", note=note, asked=asked, other=other, sides=side,
                                swapped_with=f"p{k + 1 if k % 2 else k - 1}", generator="lerobot/scripts/sim_builders/object_choice/build_choice_envs.py",
                                twin_snapshot=twin.snapshot, rules=rules, objects=V1.pose_meta(poses, twin, bodies, info), ik=info["ik"],
                                start_contact_mm=info["pen_mm"], settle=chk)
                    (out / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
            shutil.rmtree(folder)
        if tiles:
            V1.sheet(tiles, RENDERS / f"sheet_{pair}.png", cols=4, size=(640, 480))
    print(f"\n{len(failures)} failing placement(s)" + (": " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
