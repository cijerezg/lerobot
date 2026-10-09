# Franka Panda with Robotiq 2F-85 (DROID, droid_success) kinematics asset

Corpus `robot_type`: `franka_emika_panda_robotiq_2F-85`; registry keys also `droid`, `droid_success`.
Chain spec: `chain.json`. FK/IK: `lerobot.model.fk.get_chain("droid")`.

## Joint order and sign convention (as stored in the corpus)

`state.npy` / `action.npy` are (T, 8): seven Franka joint positions in radians (`joint_0..joint_6`, Franka
order and sign, same as the DH table) and the gripper position in [0, 1], 0 fully open, 1 fully closed
(DROID absolute command; `prepare_droid.py` keeps it source-native). The FK takes the 8-vector as is.

## Chain

Same Craig modified DH table and 0.107 m flange as `../panda_franka_hand/README.md` (one arm, two tools).

## Hand frame (fingertip midpoint, common axes)

Tool transform from the flange: `xyz = (0, 0, 0.1558)`, `rpy = (0, 0, -pi/2)`.

- Distance: the MuJoCo Menagerie `robotiq_2f85/2f85.xml` places `base_mount` at 0.007 m, `base` at
  a further 0.0038 m, and the `pinch` site (the point between the closed pads) at 0.145 m above `base`:
  0.1558 m from the mounting face. The Robotiq coupling between the Franka flange and the gripper
  (GRP-CPL-062 in the Franka kit) is not in that number.
- Closing axis: in the Menagerie model the pads move along the base's y, and `base` is rotated -90 deg
  about z relative to the mount, so the closing axis is the mount's x. With yaw 0 between flange and mount
  the hand frame is `z = flange z` (approach), `y = flange x` (closing), hence `rpy = (0, 0, -pi/2)`.

Status: both the coupling thickness (about 10 mm, adds to 0.1558) and the mounting yaw of the gripper on
the DROID flange are unverified. The DROID `cartesian_position` shards that would settle the TCP offset
are no longer on disk (`outputs/diverse_robot_dataset/droid/staging` was removed); the yaw can be read off
the DROID wrist-camera frames. The lowest-fingertip check in the roundtrip report is the only evidence
so far, and DROID tables sit at unknown heights relative to the base, so it is weak for this robot.

## Gripper map

`aperture_m = stroke * (1 - g)`, `raw_closed = 1`, `raw_open = 0`, `stroke = 0.085 m` (Robotiq 2F-85,
85 mm opening). The DROID value is a command ratio, so the map is linear in that ratio; the 2F-85's real
opening is not exactly linear in the command.

## Asset provenance

DH table: `annotation/precision_contact/panda_fk.py`. Robotiq geometry: google-deepmind/mujoco_menagerie
`robotiq_2f85/2f85.xml` (fetched 2026-10-09). Stroke: Robotiq 2F-85 datasheet.

## Roundtrip check (Phase 1d, 2026-10-09)

`uv run python lerobot/scripts/kinematics_roundtrip.py`, 200 recorded frames per source spread over the
episodes, `q -> fk -> ik(seed = q + noise) -> q'`, noise 5 deg (ReBot) or 0.1 rad (Panda). Output in
`migration/kinematics_roundtrip_2026-10-09/` (roundtrip.md, roundtrip.json, render\_<source>.png).

| source        | converged | same joints | pose err mm med / max | rot err deg med / max | joint err deg med / max                 |
| ------------- | --------- | ----------- | --------------------- | --------------------- | --------------------------------------- |
| droid         | 199 / 200 | 0           | 4.7e-09 / 9.9e-07     | 1.0e-10 / 5.7e-08     | 2.7e+00 / 174.1 (null space, info only) |
| droid_success | 199 / 200 | 0           | 3.5e-09 / 9.2e-07     | 8.8e-11 / 5.6e-08     | 2.6e+00 / 16.0 (null space, info only)  |

Non-converged: 1 frame each (droid episode 21 frame 1736, droid_success episode 61 frame 151), slow convergence
near a singularity. The arm chain is the one verified against FMB; only the tool differs. Negative lowest-fingertip
heights are expected: DROID arms stand on carts above their tables.

Render check: first frame and lowest-fingertip frame of 20 episodes per source, camera view beside the FK
skeleton (no global-camera extrinsics exist, so no overlay). Fingertip positions and hand axes are consistent
with the images. Per-episode lowest fingertip height (mm, p10 / median / p90): droid -130 / 20 / 134; droid_success -158 / -23 / 122.

**Status: frozen 2026-10-09.** The same files serve training and inference; changes need a new roundtrip.
