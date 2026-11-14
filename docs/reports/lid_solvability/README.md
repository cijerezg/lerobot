# close_lid solvability (2026-10-28)

> Saved 2026-11-13 from the scratch folder `lid_solvability_2026-10-28`. Kept here: this page,
> `results.json`, `feasibility.json`, `feasibility_deep.json`, `demo_pose.json`. The scripts, per-grasp
> logs, videos and frame sheets named below were not copied.

Question: can the rebot gripper close the sim lid (scenes/close_lid) the way the T071 demos do (rim pinch + flip), and does the
scorer credit it? Headless MuJoCo only, no policy runs.

## Verdict

- Solvable by a side-rim pinch that turns into a push: `side_xp` closes the lid (hinge 3.14 rad, gripper clear, no joint at a
  soft limit). The parallel pinch itself holds only to 0.4 rad with a fixed tool pitch and to 1.2-1.6 rad when the pitch rides
  with the lid; after that one finger inside the tray keeps dragging the wall over the arc and gravity finishes past 1.83 rad.
- A pinch held all the way to the tipping point is kinematically impossible on this arm: multi-start IK (feasibility.py,
  feasibility_deep.py) finds no pose inside the follower's soft limits past 1.2-1.3 rad (side wall) / 0.5 rad (far rim); the
  binding joint is wrist_flex at +90 deg (its hard limit too). The pads leave a 2.4 cm wall once the tool is > ~60 deg from the
  lid's "down", so the wrist would have to pitch ~105 deg while the tips ride a 10-13 cm arc over the hinge.
- The loop handle is on the FIXED BASE (mesh hinged_nail_ep2_21, world x 0.49-0.53, the end away from the robot), not on the
  lid. The policy drives to it (policy_ep0_sheet.jpg, t = 25-36 s). It now has colliders and a pinch on it holds
  (`loop_leg`: both fingers on loop_0/loop_1, gap 4.5 mm) but it cannot move the lid.
- Real T071 case is a different, larger case: shells ~22 x 20 cm, centre-to-centre 22 cm, vs the sim's 18.6 x 15.1 cm / 15.8 cm.

## Scripted test (solve_lid.py; results.json, <grasp>.mp4 + <grasp>_sheet.jpg, <grasp>_log.json)

IK (damped least squares, tips = gripper_end origin, slide axis hard, tool axis soft, soft limits clipped), physics at
30 x 17 Hz with the sim's gravcomp, fingers kp 5000. Contact friction finger/lid = 1.0 (MuJoCo takes the max of the finger
default 1.0 and the lid's 0.8), not 0.8.

| grasp | pinch | pinch lost at (rad) | hinge at release / final | closed | at soft limit |
|---|---|---|---|---|---|
| side_xp (side wall y=+0.107, x=0.215, tool pitch fixed 35 deg fwd) | yes, but wedged against the far wall (gap 13 mm) | 0.38 | 2.14 / 3.14 | YES | none; peaks pan 55, lift -88, elbow -77, flex 57, yaw 28, roll 49 deg |
| side_xn (mirror, y=-0.072) | yes, flat (gap 4.9 mm) | 0.27 | 0 / 0 | no (slipped off) | none |
| side_xp_track (x=0.235, 18 mm deep, pitch +40 -> -40 deg with the lid) | yes, flat | 1.43 | 0 / 0 | no | lift -170, elbow -200, flex 90 |
| side_xn_track | yes, flat | 1.44 | 0 / 0 | no | same |
| side_xp_mid (x=0.265) | yes, flat | 1.64 | 0 / 0 | no | same |
| side_xp_lidfixed (tool fixed to the lid) | wedged | - | 3.14 / 3.14 | yes, by shoving | elbow, flex |
| far_rim (slide across the far wall, tool fixed to the lid) | yes, flat (4.9 mm) | 0.03 | 0 / 0 | no | lift, elbow, flex, yaw, roll |
| far_rim_tilt / far_rim_sideways (tool along the hinge) | no (IK cannot reach: palm below the table) | - | - | no | - |
| loop_leg (loop_0 midpoint, slide y, tool 30 deg fwd) | yes on the loop (4.5 mm) | lift 5 cm pulls the fingers off | hinge 0 | n/a | none |
| loop_apex, loop_apex_fwd | IK misses / wrist_roll at 90 | - | - | n/a | - |

Kinematic ceiling (feasibility.json, 0.1 rad steps, 10 random starts; feasibility_deep.json 60 starts at 1.2/1.5/1.85):
side wall far end 1.2, mid 1.2, near hinge 1.3, far rim 0.5 rad. Tipping point 1.83 rad.

## Changes

- `scenes/close_lid/scene.xml`: header comment + six capsules `headphone_case_loop_0..5` (r 3 mm, 1 g, friction 0.8, group 3)
  on the headphone_case body along the strap centreline fitted to mesh 21 (loop_collider.py, loop_capsules.json). Original:
  scene.xml.orig. `objects.json`: `loop_handle` note (original objects.json.orig).
- `easy_tasks.py`: close_lid reach = finger midpoint within LID_REACH_M = 0.05 of the nearest lid collision geom
  (`body_surface_distance`, geom-frame AABB clamp, exact for the boxes) instead of 8 cm to the lid COM (a side pinch is
  9-10 cm from the COM, and the finger bodies sit 4.2 cm behind the tips). Stage values unchanged. Docstring updated.
- `tests/robots/test_rebot_b601_sim_tasks.py`: reach assertions on the rim, new `test_close_lid_reach_follows_the_rim`. 9 pass.
- `scenes/README.md`: close_lid row + stage line.

## Demo case pose (demo_pose.py -> demo_pose.json, frames/demo_pose_ep*.jpg)

Through the sim `ext` camera (pos 0.609 -0.014 0.849, fovy 60.73, f 409 px; the sim case projects back to within 4 mm).
The 10-04 top camera matches the 10-16 calibration to ~10 px at the table edges (frames/camera_check_blend.jpg).
Shell silhouettes (V < 120 inside the table) back-projected at z = 0.02; lid = the half nearer the robot.

| ep | base (loaded) xy | lid xy | hinge xy | base->lid dir | body_yaw (sim quat convention, sim = 242.18) |
|---|---|---|---|---|---|
| 0 (primary) | (0.497, 0.211) | (0.317, 0.342) | (0.407, 0.276) | 144 deg | 206 deg |
| 2 | (0.423, 0.257) | (0.226, 0.368) | (0.324, 0.313) | 151 deg | 213 deg |
| 3 | (0.467, 0.048) | (0.403, -0.078) | (0.435, -0.015) | -117 deg | 306 deg |

Sim today: base (0.419, 0.017), lid toward -x (180 deg). The demos put the case front-left (+y = robot's left), lid toward the
robot and turned ~36 deg; and the real case is ~1.4x the sim's, so a placement variant at ep0's pose will still show a smaller
case than the demos.

## Files

scene.xml.orig, objects.json.orig; inspect_lid_meshes.py, gripper_geometry.py, loop_mesh.py, loop_collider.py (scene analysis);
solve_lid.py (physics test), feasibility.py, feasibility_deep.py (kinematics); demo_pose.py; extract_frames.py; frames/
(policy + demo frame sheets, loop close-up, sim visual vs collision, pose overlays).

Run (at the time, from the workspace root; the script is not in this folder):
`MUJOCO_GL=egl uv run python <scratch>/lid_solvability_2026-10-28/solve_lid.py [grasp ...]`.
