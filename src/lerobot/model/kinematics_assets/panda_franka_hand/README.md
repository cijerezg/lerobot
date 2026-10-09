# Franka Panda with Franka Hand (FMB) kinematics asset

Corpus `robot_type`: FMB episodes carry none; registry keys `fmb` and `franka_emika_panda_franka_hand`.
Chain spec: `chain.json`. FK/IK: `lerobot.model.fk.get_chain("fmb")`.

## Joint order and sign convention (as stored in the corpus)

FMB `q.npy` is (T, 7) radians, Panda joints 1..7 in order, Franka's own sign convention (the same the
DH table below uses, so no flips). `gripper_pose.npy` is (T,) int, 0 = open, 1 = closed (binary; the
Franka Hand has no continuous width in this source). The FK takes `[q, gripper]` as one 8-vector.

| dim  | name               | units            |
| ---- | ------------------ | ---------------- |
| 0..6 | joint_1 .. joint_7 | rad              |
| 7    | gripper (binary)   | 0 open, 1 closed |

Panda limits (rad): q1 ±2.8973, q2 ±1.7628, q3 ±2.8973, q4 [-3.0718, -0.0698], q5 ±2.8973, q6 [-0.0175, 3.7525], q7 ±2.8973.

## Chain

Craig modified DH, (a, d, alpha) per joint, `T_i = RotX(alpha) TransX(a) RotZ(theta) TransZ(d)`:

| joint  | a (m)   | d (m) | alpha |
| ------ | ------- | ----- | ----- |
| 1      | 0       | 0.333 | 0     |
| 2      | 0       | 0     | -pi/2 |
| 3      | 0       | 0.316 | pi/2  |
| 4      | 0.0825  | 0     | pi/2  |
| 5      | -0.0825 | 0.384 | -pi/2 |
| 6      | 0       | 0     | pi/2  |
| 7      | 0.088   | 0     | pi/2  |
| flange | 0       | 0.107 | 0     |

Copied from `annotation/precision_contact/panda_fk.py` (Franka's published table).

## Hand frame (fingertip midpoint, common axes)

Tool transform from the flange: `xyz = (0, 0, 0.1034)`, `rpy = (0, 0, -pi/4)`. That is Franka's
`panda_hand_tcp`: z approach, y the finger closing axis, origin between the finger pads. The common axes
are already the hand's own, so no extra rotation.

Evidence: on 2354 FMB frames (10 episodes) this FK matches the recorded `tcp_pose.npy` (position + xyzw
quaternion) with median position error 6e-7 mm and median rotation error 5e-4 deg; the other yaw choices
(0, +45, -135 deg) give 45 or 90 deg. FMB's TCP is this frame.

## Gripper map

`aperture_m = stroke * (1 - g)`, `raw_closed = 1`, `raw_open = 0`, `stroke = 0.08 m` (Franka Hand
datasheet, 80 mm). Binary source, so the aperture is 0 or 80 mm.

## Roundtrip check (Phase 1d, 2026-10-09)

`uv run python lerobot/scripts/kinematics_roundtrip.py`, 200 recorded frames per source spread over the
episodes, `q -> fk -> ik(seed = q + noise) -> q'`, noise 5 deg (ReBot) or 0.1 rad (Panda). Output in
`migration/kinematics_roundtrip_2026-10-09/` (roundtrip.md, roundtrip.json, render\_<source>.png).

| source | converged | same joints | pose err mm med / max | rot err deg med / max | joint err deg med / max                |
| ------ | --------- | ----------- | --------------------- | --------------------- | -------------------------------------- |
| fmb    | 200 / 200 | 0           | 3.4e-09 / 8.3e-07     | 6.3e-11 / 5.3e-08     | 2.8e+00 / 12.9 (null space, info only) |

Independent check: FK equals the recorded FMB `tcp_pose` on 2354 frames, median 6e-7 mm and 5e-4 deg, max 2.8 mm
and 0.48 deg (single frames, source timing).

Render check: first frame and lowest-fingertip frame of 20 episodes per source, camera view beside the FK
skeleton (no global-camera extrinsics exist, so no overlay). Fingertip positions and hand axes are consistent
with the images. Per-episode lowest fingertip height (mm, p10 / median / p90): fmb 31 / 50 / 137.

**Status: frozen 2026-10-09.** The same files serve training and inference; changes need a new roundtrip.
