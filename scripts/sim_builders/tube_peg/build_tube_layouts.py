"""The tube layouts on the twin desk (tasks.py tier ``tube``): two with one tube and one peg holder, one with two pairs, one with
three (user, 2026-11-15). Tube i (tube, tube_2, tube_3) goes on holder i; the third holder has the dark peg.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/tube_peg/build_tube_layouts.py

Writes scenes/tube_peg_{left,right,two,three}_p1/{scene.xml,objects.json} in the robot package and renders/layouts.png here. Per placement:
1 s settle (< 1 mm), then the scripted grasp / lift / insert of verify_tube_peg.py (close command 0 deg, 10 N finger cap) with the
task's scorer polled every control tick; a layout passes when every tube ends on its peg and the scorer reaches reward 1.0.
Tubes stay at x >= 0.36 and every body within 0.49 m of the base, tubes within 0.47 m (top-pinch reach; a tube at 0.48 m was knocked over): nearer the base the top camera barely shows them (user, 2026-11-15).

Frame: x toward the far edge / top camera, y to the robot's left (= right in the top image); the subtask strings name the robot's side.
Poses are (x, y, yaw deg). The recordings grasp at x 0.09-0.50, y -0.41..0.42 and carry 5-35 cm to the peg.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tube_peg as TP  # noqa: E402
import verify_tube_peg as V  # noqa: E402

from lerobot.robots.rebot_b601_sim.tasks import task_for  # noqa: E402

V1 = V.V1
CLOSE_DEG, CAP_N = 0.0, 10.0
LAYOUTS = {
    "tube_peg_left_p1": dict(note="far left, peg further left", tube=(0.45, 0.12, 0), tube_holder=(0.44, 0.21, 0)),
    "tube_peg_right_p1": dict(note="middle right, peg near right, long carry toward the robot", tube=(0.40, -0.08, 0), tube_holder=(0.27, -0.26, 90)),
    "tube_peg_two_p1": dict(note="left pair carried outward, right pair carried toward the robot",
                            tube=(0.36, 0.16, 0), tube_holder=(0.45, 0.19, 20), tube_2=(0.44, -0.14, 0), tube_holder_2=(0.34, -0.24, -40)),
    "tube_peg_three_p1": dict(note="left pair far left, right pair in the middle, dark peg far right",
                              tube=(0.44, 0.14, 0), tube_holder=(0.34, 0.22, 0), tube_2=(0.38, -0.10, 0), tube_holder_2=(0.48, -0.06, 90),
                              tube_3=(0.40, -0.26, 0), tube_holder_3=(0.30, -0.33, 30)),
}


def scorer_poll(name):
    made = {}

    def poll(model, data):
        made["scorer"] = task_for(name).scorer(model, data, lambda: 0.0)
        return made["scorer"].poll

    return poll, made


def main():
    twin = V1.TwinText()
    twin.bodies.update(tube=TP.tube_body(), tube_holder=TP.holder_body(), tube_2=TP.tube_body("tube_2"), tube_holder_2=TP.holder_body("tube_holder_2"),
                       tube_3=TP.tube_body("tube_3"), tube_holder_3=TP.holder_body("tube_holder_3", peg_rgba="0.12 0.12 0.14 1"))
    tiles, ok_all = [], True
    for name, spec in LAYOUTS.items():
        bodies = [b for b in spec if b != "note"]
        pairs = [(b, b.replace("tube", "tube_holder")) for b in bodies if "holder" not in b]
        poses = {b: (np.array([spec[b][0], spec[b][1], 0.0005]), V1.quat_z(np.radians(spec[b][2]))) for b in bodies}
        twin.pose.update(poses)
        folder = V1.SCENES / name; folder.mkdir(exist_ok=True)
        (folder / "scene.xml").write_text(twin.scene_xml(f"scene_{name}", bodies, poses))
        settle = V1.settle(folder, bodies)
        poll, made = scorer_poll(name)
        res = V.run(folder, CLOSE_DEG, CAP_N, 0.0, poll=poll, pairs=pairs)
        scorer = made["scorer"]
        ok = settle["max_moved_mm"] < 1.0 and res["on_peg"] and scorer.reward == 1.0
        ok_all &= ok
        meta = dict(layout=name, tier="tube", note=spec["note"], generator="scripts/sim_builders/tube_peg/build_tube_layouts.py",
                    objects={b: dict(xy=list(spec[b][:2]), yaw_deg=spec[b][2]) for b in bodies},
                    checks=dict(settle=settle, scripted_insert=[{k: v for k, v in r.items() if k != "trace"} for r in res.get("pairs", [res])],
                                scorer=dict(reward=scorer.reward, events=scorer.events, **scorer.extra())))
        (folder / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"{name}: {'ok' if ok else 'FAILED'}  settle {settle['max_moved_mm']} mm  tubes on pegs {[r['on_peg'] for r in res.get('pairs', [res])]}  "
              f"holders moved mm {[r['holder_moved_mm'] for r in res.get('pairs', [res])]}  reward {scorer.reward}  "
              f"last stage at {scorer.events[-1]['t']} s", flush=True)
        tiles += [(f"{name} ({cam})", img) for cam, img in V1.render(folder).items()]
    out = HERE / "renders"; out.mkdir(exist_ok=True)
    V1.sheet(tiles, out / "layouts.png", cols=4, size=(640, 480))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
