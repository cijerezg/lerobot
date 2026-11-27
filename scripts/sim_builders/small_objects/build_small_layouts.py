"""The small objects layouts on the twin desk (tasks.py tier ``small``): die -> dice cup, plug -> its colour compartment of the
parts box, screwdriver -> its case, nut -> the pin of the holder; four configurations each (user, 2026-11-26).
The start levels of every layout (arm near the object, wrist turned) are build_small_starts.py's, run after this.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/small_objects/build_small_layouts.py

Writes scenes/<layout>_p1/{scene.xml,objects.json} in the robot package and renders/layouts.png here. Per layout: 1 s settle
(< 1 mm), then the scripted pick and place of verify_small_objects.py (close command 0 deg, 10 N finger cap) with the task's
scorer polled every control tick; a layout passes when the scorer reaches reward 1.0.

Frame: x toward the far edge / top camera, y to the robot's left. Poses are (x, y, yaw deg). Objects start at x >= 0.36 (seen by
the top camera) and within 0.47 m of the base; the cup stays where the top pinch reaches 11 cm above the table (x <= 0.40, |y| <= 0.2,
or x <= 0.30). Case yaw -90 = as recorded (label at the far end, lid open to the robot's right); box yaw -90 = plug row toward the robot.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import small_objects as SO  # noqa: E402
import verify_small_objects as V  # noqa: E402

from lerobot.robots.rebot_b601_sim.tasks import task_for  # noqa: E402

V1 = V.V1
DRIVER_AT = (V.DRIVER_AT, (0.024, -0.022))  # first and second screwdriver in the case, beside the two already there
LAYOUTS = {
    "die_cup_left_p1": dict(note="die near left, cup toward the middle, carried inward", die=(0.36, 0.24, 25), dice_cup=(0.38, 0.09, 0)),
    "die_cup_right_p1": dict(note="die far, right of the middle, cup near right, long carry toward the robot", die=(0.44, -0.08, 60), dice_cup=(0.29, -0.27, 0)),
    "die_cup_three_p1": dict(note="cup near the middle, dice far left, far middle and right", die=(0.42, 0.15, 10), die_2=(0.43, -0.05, 45), die_3=(0.36, -0.25, 70), dice_cup=(0.29, 0.03, 0)),
    "die_cup_across_p1": dict(note="die near right, cup left, carried across the desk", die=(0.38, -0.18, 40), dice_cup=(0.30, 0.12, 0)),
    "plug_box_yellow_p1": dict(note="yellow plug far left, box near the middle with the plug row toward the robot", plug_yellow=(0.42, 0.14, 40), plug_box=(0.30, -0.03, -90)),
    "plug_box_blue_p1": dict(note="blue plug right, box left of the middle, turned a quarter", plug_blue=(0.38, -0.22, -30), plug_box=(0.36, 0.08, 0)),
    "plug_box_three_p1": dict(note="box near right, plugs far middle, right of the middle and near left", plug_yellow=(0.44, 0.07, 80), plug_red=(0.40, -0.08, 15), plug_blue=(0.36, 0.25, -45), plug_box=(0.28, -0.22, -70)),
    "plug_box_red_p1": dict(note="red plug near the middle, box near left, turned a quarter the other way", plug_red=(0.37, 0.00, 70), plug_box=(0.31, 0.22, 90)),
    "screwdriver_case_left_p1": dict(note="screwdriver far left, case in the middle", screwdriver=(0.40, 0.20, 110), screwdriver_case=(0.33, 0.03, -90)),
    "screwdriver_case_right_p1": dict(note="screwdriver far middle, case near right, turned", screwdriver=(0.43, -0.03, 20), screwdriver_case=(0.30, -0.24, -70)),
    "screwdriver_case_two_p1": dict(note="case near the middle, one screwdriver far left, one right", screwdriver=(0.43, 0.11, 60), screwdriver_2=(0.37, -0.23, -20), screwdriver_case=(0.30, -0.05, -100)),
    "screwdriver_case_across_p1": dict(note="screwdriver right, case left, carried across the desk", screwdriver=(0.37, -0.22, -45), screwdriver_case=(0.33, 0.20, -90)),
    "nut_peg_left_p1": dict(note="nut left, holder near left", nut=(0.40, 0.16, 0), nut_holder=(0.30, 0.27, 30)),
    "nut_peg_right_p1": dict(note="nut far middle, holder right, turned round", nut=(0.44, -0.05, 0), nut_holder=(0.37, -0.22, -120)),
    "nut_peg_across_p1": dict(note="nut right, holder left of the middle, turned a quarter", nut=(0.37, -0.20, 0), nut_holder=(0.32, 0.12, 90)),
    "nut_peg_two_p1": dict(note="holder near the middle, nuts left and right, the second goes on top of the first", nut=(0.38, 0.22, 0), nut_2=(0.42, -0.16, 0), nut_holder=(0.30, 0.0, 180)),
}
COMPARTMENT = {colour: ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2 - 0.006) for colour, (lo, hi) in SO._box_compartments().items()}  # in front of the plug already there


def body_xml(name):
    kind = SO.kind_of(name)
    if kind == "plug":
        return SO.plug_body(name, name.split("_")[1])
    return dict(die=SO.die_body, screwdriver=SO.screwdriver_body, nut=SO.nut_body, dice_cup=SO.cup_body, plug_box=SO.plug_box_body,
                screwdriver_case=SO.case_body, nut_holder=SO.nut_holder_body)[kind or name](name)


def moves(bodies):
    """One move per object in scene order; fingers close across the holder's hoop axis for the nut (the hoop is 14 mm from the pin)."""
    out = []
    for body in bodies:
        kind = SO.kind_of(body)
        if kind is None:
            continue
        i = len(out)
        at, yaw = dict(die=((0.0, 0.0), None), plug=(COMPARTMENT.get(body.split("_")[-1]), 0.0), screwdriver=(DRIVER_AT[min(i, 1)], 90.0), nut=((0.0, 0.0), 0.0))[kind]
        out.append(dict(body=body, place=SO.PLACE[kind], at=at, yaw=yaw, up=SO.NUT_H * i if kind == "nut" else 0.0))
    return out


def scorer_poll(name):
    made = {}

    def poll(model, data):
        made["scorer"] = task_for(name).scorer(model, data, lambda: 0.0)
        return made["scorer"].poll

    return poll, made


def main():
    names = sys.argv[1:] or list(LAYOUTS)
    twin = V1.TwinText()
    tiles, ok_all = [], True
    out = HERE / "renders"; (out / "solutions").mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = LAYOUTS[name]
        bodies = [b for b in spec if b != "note"]
        twin.bodies.update({b: body_xml(b) for b in bodies})
        poses = {b: (np.array([spec[b][0], spec[b][1], 0.0005 + SO.REST_Z.get(SO.kind_of(b), 0.0)]), V1.quat_z(np.radians(spec[b][2]))) for b in bodies}
        twin.pose.update(poses)
        folder = V1.SCENES / name; folder.mkdir(exist_ok=True)
        (folder / "scene.xml").write_text(twin.scene_xml(f"scene_{name}", bodies, poses))
        settle = V1.settle(folder, bodies)
        poll, made = scorer_poll(name)
        rows = V.run(folder, moves(bodies), poll=poll, video=out / "solutions" / f"{name}.mp4")
        scorer = made["scorer"]
        ok = settle["max_moved_mm"] < 1.0 and scorer.reward == 1.0
        ok_all &= ok
        meta = dict(layout=name, tier="small", note=spec["note"], generator="scripts/sim_builders/small_objects/build_small_layouts.py",
                    objects={b: dict(xy=list(spec[b][:2]), yaw_deg=spec[b][2]) for b in bodies},
                    checks=dict(settle=settle, scripted_moves=rows, scorer=dict(reward=scorer.reward, events=scorer.events, **scorer.extra())))
        (folder / "objects.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
        print(f"{name}: {'ok' if ok else 'FAILED'}  settle {settle['max_moved_mm']} mm  ends mm {[r['rel_mm'] for r in rows]}  places moved mm {[r['place_moved_mm'] for r in rows]}  "
              f"reward {scorer.reward}  last stage {scorer.events[-1]['stage'] if scorer.events else None} at {scorer.events[-1]['t'] if scorer.events else None} s", flush=True)
        tiles += [(f"{name} ({cam})", img) for cam, img in V1.render(folder).items()]
    if not sys.argv[1:]:
        V1.sheet(tiles, out / "layouts.png", cols=4, size=(640, 480))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
