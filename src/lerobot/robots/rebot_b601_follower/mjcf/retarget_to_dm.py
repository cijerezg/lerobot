#!/usr/bin/env python
"""Retarget the MuJoCo Menagerie reBot model from the RS variant to our B601-DM.

The upstream model (`seeed_rebot_devarm.xml`) is the RobStride variant. Our arm is the
Damiao B601-DM, and the two are not the same geometry: the DM upper arm is 28 mm longer
(`link3` at -0.264 m against -0.236 m), and per-link position error against our URDF
reaches 25 mm at the end effector. Tuning a teleop gain or a workspace bound on the RS
arm and moving it to the DM arm would be silently wrong.

Body *orientations* already match our URDF exactly (quats correspond to the URDF rpy, and
FK rotation agrees to 0.01 deg), so only the `pos` attributes need replacing. This script
copies them from `urdf/reBot-DevArm_fixend.urdf` and writes `*_dm.xml` + `scene_dm.xml`.

Caveat, deliberately not fixed: the visual and collision meshes remain RS-shaped and
RS-sized, as do the inertials. After retargeting they sit fractionally off the true DM
link boundaries. That is fine for reach, IK behaviour, workspace bounds and table
contact; it is not a model of DM finger-object contact at millimetre precision.

    uv run python retarget_to_dm.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
URDF = HERE.parent / "urdf" / "reBot-DevArm_fixend.urdf"

# MJCF body <- URDF joint whose origin defines that body's placement in its parent.
BODY_FROM_JOINT = {
    "link1": "joint1",
    "link2": "joint2",
    "link3": "joint3",
    "link4": "joint4",
    "link5": "joint5",
    "link6": "joint6",
    "gripper_end": "end_joint",
}


def main() -> None:
    urdf = ET.parse(URDF).getroot()
    origins = {
        j.get("name"): j.find("origin").get("xyz")
        for j in urdf.findall("joint")
        if j.find("origin") is not None
    }

    tree = ET.parse(HERE / "seeed_rebot_devarm.xml")
    root = tree.getroot()
    root.set("model", "rebot_b601_dm")

    changed = 0
    for body in root.iter("body"):
        joint = BODY_FROM_JOINT.get(body.get("name"))
        if joint is None:
            continue
        new = " ".join(f"{float(v):.6g}" for v in origins[joint].split())
        print(f"  {body.get('name'):12s} {body.get('pos'):32s} -> {new}")
        body.set("pos", new)
        changed += 1

    assert changed == len(BODY_FROM_JOINT), f"patched {changed}, expected {len(BODY_FROM_JOINT)}"

    # Joint limits too, so the plant and the IK solver agree on what is reachable.
    # Upstream joint4 is [-102.6, 96.8] deg against our URDF's [-107, 90].
    limits = {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in urdf.findall("joint")
        if j.find("limit") is not None
    }
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in limits:
            continue
        new = f"{limits[name][0]:.6g} {limits[name][1]:.6g}"
        if new != joint.get("range"):
            print(f"  {name:12s} range {joint.get('range'):20s} -> {new}")
            joint.set("range", new)
    for act in root.iter("position"):
        name = act.get("joint")
        if name in limits:
            act.set("ctrlrange", f"{limits[name][0]:.6g} {limits[name][1]:.6g}")
    tree.write(HERE / "rebot_b601_dm.xml", encoding="utf-8", xml_declaration=True)

    scene = ET.parse(HERE / "scene.xml")
    scene_root = scene.getroot()
    scene_root.set("model", "rebot_b601_dm scene")
    scene_root.find("include").set("file", "rebot_b601_dm.xml")
    scene.write(HERE / "scene_dm.xml", encoding="utf-8", xml_declaration=True)
    print(f"wrote {HERE / 'rebot_b601_dm.xml'} and scene_dm.xml")


if __name__ == "__main__":
    main()
