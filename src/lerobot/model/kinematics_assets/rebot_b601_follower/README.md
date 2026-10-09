# ReBot B601-DM kinematics asset

Corpus `robot_type`: `rebot_b601_follower`. Chain spec: `chain.json`. FK/IK: `lerobot.model.fk.get_chain("rebot_b601_follower")`.

## Joint order and sign convention (as stored in the corpus)

`observation.state` / `action` are 7 floats in degrees, follower driver order. The mapping to the URDF
joints is the identity: same order, same sign, no offset. The FK takes the corpus vector as is.

| dim | corpus name   | URDF joint                             | axis in joint frame | URDF limit (deg) | follower limit (deg) |
| --- | ------------- | -------------------------------------- | ------------------- | ---------------- | -------------------- |
| 0   | shoulder_pan  | joint1                                 | +z                  | ±160.4           | ±145                 |
| 1   | shoulder_lift | joint2                                 | -z                  | [-180, 0]        | [-170, 1]            |
| 2   | elbow_flex    | joint3                                 | +z                  | [-180, 0]        | [-200, 1]            |
| 3   | wrist_flex    | joint4                                 | +z                  | [-107, 90]       | [-80, 90]            |
| 4   | wrist_yaw     | joint5                                 | +z                  | ±90              | ±90                  |
| 5   | wrist_roll    | joint6                                 | +z                  | ±180             | ±90                  |
| 6   | gripper       | (prismatic fingers, not a chain joint) |                     |                  | [-270, 0]            |

Zero is the calibration pose (arm folded, "sit-down", gripper closed). Follower limits are
`config_rebot_b601_follower.py`; the sign and identity mapping were established in
`robots/rebot_b601_follower/urdf/README.md` by matching the URDF limits against the recorded ranges
(`shoulder_lift` and `elbow_flex` live in [-200, 1], which only the DM/fixend chain admits; the RS chain
has those limits mirrored to [0, 180]). `elbow_flex` goes past the URDF's -180 limit down to -200; FK does
not clamp, and the IK here has no joint limits.

Gripper: 0 = shut, negative = open; the raw value is motor degrees, not a finger position.

## Hand frame (fingertip midpoint, common axes)

The URDF's `end_link` is fixed to `link6` at 0.15539 m along the link6 z axis, with
rpy (0, -pi/2, pi). In `end_link`: +x is the approach (pointing) direction (the one
`annotation/rebot/motion.py` uses for the approach angle), the prismatic finger joints of
`ReBot_Arm_DM.urdf` slide along ±y, so y is the closing axis. The `end_link` origin sits at the distal
end of the gripper body: the fixend mesh hull of `end_link` (`robots/rebot_b601_follower/urdf/link_hulls.npz`)
spans x in [-0.107, 0] m, i.e. the whole gripper hangs behind the origin and the origin is at the tips.

Tool transform in `chain.json`: `xyz = 0`, `rpy = (0, pi/2, 0)` from `end_link`, giving the hand frame
`z = end_link x` (approach), `y = end_link y` (closing), `x = -end_link z`. The chain-level transform the FK
uses is `end_joint @ tool` (joint6 frame to hand frame).

Evidence (2026-10-09): the upstream DM collision meshes (`gripper_base.stl`, `left_finger.stl`,
`right_finger.stl`, Seeed-Projects/reBotArm_control_py `urdf/DM/meshes/collision/`, copies and the
measuring script in `migration/kinematics_roundtrip_2026-10-09/rebot_meshes/`) are expressed in the
`end_link` frame. The gripper base spans x in [-0.107, -0.073] m, the fingers x in [-0.089, 0], and with
both finger joints at 0 (closed) the two tips meet at x = 0, y = 0, z in ±3.5 mm: the fingertip midpoint
is the `end_link` origin to within the 4 mm tip chamfer, centred on the roll axis. The fixend hull says the
same (it tapers to a point at the origin). No caliper measurement was made.

## Gripper map

`aperture_m = clip(stroke * (raw - raw_closed) / (raw_open - raw_closed), 0, stroke)` with
`raw_closed = 0` (calibration zero, gripper shut), `raw_open = -338.2`, `stroke = 0.057 m`.

Evidence (2026-10-09): with torque off the follower read 0.8 deg fully shut and -338.2 deg at the
mechanical open stop (`migration/kinematics_roundtrip_2026-10-09/read_gripper_raw.py`). The DM URDF
finger joints travel 0 to 0.0285 m each, closed at 0, so the tip gap is 0 shut and 57 mm at the stop
(the meshes confirm the tips touch at 0). The fingers run on racks, so the opening is linear in motor
degrees: 0.0843 mm per degree per finger. Commands stay inside the follower's software limit of -270
deg (45.5 mm); the corpus never opens past -238.8 deg (40.2 mm).

Assumption: the URDF finger limit coincides with the mechanical open stop. The one number a caliper would
add is the tip gap at that stop.

## Asset provenance

`ReBot_Arm_DM.urdf` is a byte copy of
`migration/rebot_public_datasets_2026-09-20/source_evidence/reBotArm_control_py/urdf/DM/urdf/ReBot_Arm_DM.urdf`
(Seeed-Projects/reBotArm_control_py, Damiao-motor B601-DM description; sha256
`cd3054edec30b70a3afb2e97eb288ad4943a31c9746f8775703733b64ac0c3cf`). Its mesh files are not vendored;
the loader reads joints only.

Three candidate URDFs were diffed on their joint definitions (origin xyz/rpy, axis, limits):

| candidate                                                                                                                                               | joint1..6 + end joint                                                                                                                  | verdict                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `robots/rebot_b601_follower/urdf/reBot-DevArm_fixend.urdf` (repo, fetched 2026-07-27)                                                                   | identical to DM                                                                                                                        | same chain, no gripper joints                       |
| `source_evidence/reBotArm_control_py/urdf/DM/urdf/ReBot_Arm_DM.urdf`                                                                                    | identical to fixend, plus `finger_left`/`finger_right` prismatic on `end_link` (±0.0285 m along y)                                     | chosen: same arm chain and it documents the gripper |
| `source_evidence/reBotArm_control_py/urdf/RS/urdf/ReBot_Arm_RS.urdf` and `lerobot-robot-seeed-b601/.../00-arm-rs_asm-v3.urdf` (identical to each other) | different link lengths (0.236 / 0.228 vs 0.264 / 0.2426), joint1/3/4/5/6 axes flipped to -z, joint2/3 limits [0, pi], end at 0.16621 m | RS arm, not ours                                    |

The chosen chain reproduces pinocchio on the same URDF to 1e-15 (see `lerobot/scripts/kinematics_roundtrip.py`).

## Roundtrip check (Phase 1d, 2026-10-09)

`uv run python lerobot/scripts/kinematics_roundtrip.py`, 200 recorded frames per source spread over the
episodes, `q -> fk -> ik(seed = q + noise) -> q'`, noise 5 deg (ReBot) or 0.1 rad (Panda). Output in
`migration/kinematics_roundtrip_2026-10-09/` (roundtrip.md, roundtrip.json, render\_<source>.png).

| source | converged | same joints | pose err mm med / max | rot err deg med / max | joint err deg med / max |
| ------ | --------- | ----------- | --------------------- | --------------------- | ----------------------- |
| rebot  | 199 / 200 | 193         | 2.1e-08 / 9.9e-07     | 2.8e-10 / 4.6e-08     | 1.7e-08 / 12.3          |

Non-converged: 1 frame (episode 26, frame 1834), slow convergence near a singularity, pose error 4e-5 mm at the
100-iteration cap. Other branch: 6 frames with elbow_flex in [-174, -162] deg, where the elbow-up and elbow-down
solutions nearly coincide and the 5 deg seed noise hops across (pose identical, joints differ by 3 to 12 deg).
Independent check: torch FK equals pinocchio on the same URDF at `end_link` to 4e-16 m on 500 random joint vectors.
Lowest-fingertip frames show the pads on the cloth at 13 to 60 mm above the base plane (mount plate plus cloth).

Render check: first frame and lowest-fingertip frame of 20 episodes per source, camera view beside the FK
skeleton (no global-camera extrinsics exist, so no overlay). Fingertip positions and hand axes are consistent
with the images. Per-episode lowest fingertip height (mm, p10 / median / p90): rebot 21 / 31 / 62.

**Status: frozen 2026-10-09.** The same files serve training and inference; changes need a new roundtrip.
