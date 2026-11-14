# Twin eval suite (2026-10-30, redesign of the morning build)

Graded sim eval of the policy on the digital-twin desk: 4 levels x 4 hand-designed configurations x 4 placements x 4 episodes =
16 configurations, 64 scene folders `scenes/twin_l<L>_<config>_p<1-4>`, 256 episodes (L3 (d) "Close the headphone case lid" added
in the afternoon with the bit kit closed, coordinator's decision). Package side: `lerobot/src/lerobot/robots/rebot_b601_sim/` (`scenes/twin_*`,
`scenes/assets/twin/`, `twin_tasks.py`, the twin parts of `tasks.py`, `run_sim_eval.sh`, `scenes/README.md`); the morning's 70
`twin_easy_* / twin_medium_* / twin_hard_two / twin_fold_kit` folders are deleted. Build and checks here: `build_twin_envs.py`,
`scorer_check.py`, `checks.json`, `renders/`. Dense scorers unchanged except the two-object scorer, now an ordered list.

Twin source: `twin/scene_twin.xml` as of 2026-10-30 11:05 (same snapshot as the morning). The scenes keep the twin's room, the fitted
desk texture, the appearance pass's lights and the aligned `ext` camera; no background composite. New body `tape_roll_2` = the twin's
tape roll block again (same meshes, body and geom names suffixed `_2`); on the full desk its clutter pose is (0.07, -0.30), left of
the robot next to the first tape roll.

Frame: x toward the far edge / top camera, y to the robot's right (image right in the ext view), yaw in degrees.

## Levels and configurations

Registry names (`tasks.py`), tiers `twin_l1 .. twin_l4`; `p1` = the base configuration below, `p2-p4` jittered.

| name | L | task string | script steps | bodies on the desk | base poses (x, y, yaw) of the placed bodies |
|---|---|---|---|---|---|
| twin_l1_tape_basket_right | 1a | Put the tape roll in the basket (T054) | 3 | basket, tape_roll | basket (0.36, 0.22, 0), tape_roll (0.30, -0.06, 0): tape left of the basket, basket right |
| twin_l1_tape_basket_left | 1b | Put the tape roll in the basket (T054) | 3 | basket, tape_roll | basket (0.42, -0.16, 0) far left, tape_roll (0.26, 0.14, 20) right |
| twin_l1_pill_basket_far | 1c | Put the pill bottle in the basket | 4 | basket, pill_bottle | pill_bottle (0.20, 0.00, 0) in front, basket (0.46, 0.03, 0) behind it |
| twin_l1_roar_basket_near | 1d | Put the orange bottle in the basket | 4 | basket, roar_bottle | basket (0.30, -0.26, 0) near left, roar_bottle (0.42, 0.16, 0) far right |
| twin_l2_tape_bits | 2a | Put the tape roll in the basket (T054) | 3 | basket, tape_roll, pill_bottle, bit_1..bit_4 | basket (0.42, -0.12, 0), tape_roll (0.24, 0.16, 0), pill_bottle (0.36, 0.22, 0), bits (0.20, 0.30, 30) (0.28, 0.33, -45) (0.34, 0.38, 80) (0.45, 0.24, 10) |
| twin_l2_pill_case | 2b | Put the pill bottle in the basket | 4 | basket, pill_bottle, clipper_case + clipper_large + clipper_small seated | basket (0.34, 0.26, 0), pill_bottle (0.30, -0.10, 0), clipper_case (0.42, -0.10, 180); the clippers ride in the case at the nail_task slot centres, z 0.025 in the case frame |
| twin_l2_roar_tapes | 2c | Put the orange bottle in the basket | 4 | basket, roar_bottle, tape_roll, tape_roll_2, clipper_large, clipper_small | basket (0.46, 0.08, 0), roar_bottle (0.22, -0.16, 0), tape_roll (0.22, 0.14, 0), tape_roll_2 (0.36, -0.24, 0), clipper_large (0.30, 0.30, 45), clipper_small (0.22, 0.32, -30) flat |
| twin_l2_tape_bottles | 2d | Put the tape roll in the basket (T054) | 3 | basket, tape_roll, pill_bottle, roar_bottle | basket (0.28, -0.26, 0) near left, tape_roll (0.44, 0.16, 0), pill_bottle (0.22, 0.14, 0), roar_bottle (0.40, 0.34, 0) |
| twin_l3_tape | 3a | Put the tape roll in the basket (T054) | 3 | full desk (24 bodies) | tape_roll (0.19, 0.12, 0); basket at its twin pose (0.50, 0.48, 1) with contents |
| twin_l3_pill | 3b | Put the pill bottle in the basket | 4 | full desk | pill_bottle (0.21, 0.15, 0); basket at the twin pose |
| twin_l3_fold_kit | 3c | Close the bit kit | 6 | full desk | bits_kit at its twin pose (0.32, -0.06, 246), jittered in p2-p4; basket at the twin pose |
| twin_l3_close_lid | 3d | Close the headphone case lid (T071) | 3 | full desk, bit kit CLOSED | headphone_case (0.46, -0.40, 210): base half far left, lid toward the robot (lid walls at r 0.44 / 0.47); kit closed at its twin pose; basket at the twin pose |
| twin_l4_tape_cup | 4a | Put the tape roll and the cup in the basket | 7 | full desk | tape_roll (0.19, 0.12, 0), paper_cup (0.20, 0.25, 0) = its twin pose |
| twin_l4_two_tapes | 4b | Put the two tape rolls in the basket | 7 | full desk | tape_roll (0.19, 0.13, 0), tape_roll_2 (0.37, 0.28, 0) |
| twin_l4_pill_roar | 4c | Put the pill bottle and the orange bottle in the basket | 7 | full desk | pill_bottle (0.20, 0.12, 0), roar_bottle (0.37, 0.28, 0) |
| twin_l4_three | 4d | Put the tape roll, the pill bottle and the cup in the basket | 10 | full desk | tape_roll (0.19, 0.13, 0), pill_bottle (0.33, 0.28, 0), paper_cup (0.21, 0.26, 0) |

Full desk = every twin body (kit open, headphone case, navy / pill / ROAR bottles, cup, clippers, clipper case, bits, bits box, bit
holder, basket with bottle / can / two cards) plus tape_roll_2. On the full desk the open kit's hull covers the middle of the README
region (x 0.12-0.53, y -0.32..0.21) and the twin's basket is 0.69 m from the base, so the targets sit in the two free pockets inside the
region and reach disc (near the robot's right, x 0.17-0.25 / y 0.08-0.30, and right-far of the kit, x 0.31-0.43 / y 0.24-0.30; found
by scanning a 2 cm grid against the clearance rule) and the basket is anchored at its twin pose with its contents riding along
(`anchor_basket`, as the morning's hard layout). L3 (a)/(b) and L4 keep every other twin body at its twin pose in p1.

L3 (d): the only configuration where a twin body is not in its twin state. The bit kit is closed at its twin pose: the settled closed
angles of the fold asset (`assets/bits_kit_fold/test_results.json`, "closed_after_10s": hinge_left 121.6, hinge_left_flap 56.2,
hinge_right 147.8, hinge_right_flap 34.3 deg) are baked into the strip / flap body quaternions, and the joint ranges and dead-band
tendon springs are shifted by the same angles (identical physics, re-parametrised joints: qpos0 = the closed stack, FoldKitScorer
angles would read 0 here and it is not used). The closed kit's hull is one panel (0.30 x 0.14 m) and stays closed (hinges within 0.6
deg after a 2 s settle, 3.2 mm displacement = the 2 mm lift plus the stack settling). The open headphone case goes far left where the
open kit's left panel was, the lid toward the robot; its base half is 0.61 m from the base, so, as close_lid in the 10-28 variants, the
case is exempt from the region / reach rule and the pinch IK on both lid side walls is its reach test. Pose from a 2 cm / 15 deg scan
of the clearance + IK rule (11 feasible grid poses, all with the base half at x 0.46-0.50, y -0.44..-0.32, yaw 195-240; the chosen one
has feasible neighbours on three sides). `objects.json` carries `kit_closed: true` and `kit_closed_angles_deg`.

L4 task strings are in the sheet's style (the sheet has no entry for them); scripts = the single-object steps in order
("grasp the tape roll / move the tape roll to the basket / release the tape roll in the basket", the second tape roll is "the other
tape roll") + "return to home"; precisions 3/1/1 per object, contacts rim-pinch (tape, cup) or side-pinch (bottles) / na / drop.

## Placement rules (`build_twin_envs.py`)

- Seed 20261030, per-configuration sub-seed = index in `CONFIGS`; byte-identical on re-run (checked: `twin_l2_roar_tapes` and
  `twin_l4_three` regenerated, md5 of the scene files unchanged).
- p1 = the base poses. p2-p4: every body in the configuration's base list (targets, distractors, basket, kit) gets xy +-6 cm and yaw
  +-20 deg, sampled body by body (up to 2000 draws each) until it is inside the README region (targets and basket also inside the
  reach disc: 0.50 m, basket 0.48 m; the anchored basket is exempt), its AABB 2 cm inside the desk, its convex footprint (all geoms, the
  visual meshes included) 2 cm clear of every body placed so far and of the robot base; then the whole set must pass: worst start
  contact of the placed bodies > -1 mm with the parked arm and the rest of the desk (contacts inside a group, e.g. the twin's small
  clipper in the case at -1.7 mm, are not counted), IK reach for every target (straight-down pinch at the object top; position-only
  side pinch at 12 cm for the ROAR bottle; both kit flap edges; position-only 5 cm over the basket rim), 1 s settle with every placed
  body moving < 5 mm. A failing set is re-drawn (up to 500 sets); the tables below show the draws needed.
- Full-desk clutter (bodies not in the base list) in p2-p4: each gets up to 20 jitter draws of the same size and keeps the first one
  that holds the desk margin and the 2 cm clearance, else its twin pose ("jittered only where the generator can keep clearance"). In
  practice 14-17 of the 18 clutter bodies move on every full-desk placement (the twin desk is sparse enough); the ones that stay are
  the bits sitting closest together. The basket contents and the twin's small clipper ride with their parent.
- Each folder's `objects.json` lists every body with `role` (target / basket / distractor / rides_with_<parent> / clutter), `placed`,
  `jittered` and `jitter` = {dx_m, dy_m, dyaw_deg} drawn, plus the IK results, start contact, attempts and the settle record.
- `tape_roll_2`, the seated clippers and the loose small clipper (flat at z 0.0065, the twin has it tilted in the case) are the only
  bodies with poses not taken from the twin.

Per placement (poses of the placed task bodies; the basket's full-desk pose is its jittered anchor; `clutter` = number of clutter
bodies that moved; `draws` = sets drawn; IK position error max over the targets and the basket; settle displacement max over the placed
bodies and over every body; worst contact distance during the settle; the full record is `checks.json`):

| placement | poses (x, y) yaw | clutter | draws | IK err (mm) | settle placed (mm) | settle all (mm) | worst contact (mm) |
|---|---|---|---|---|---|---|---|
| twin_l1_tape_basket_right_p1 | basket (0.36, 0.22) 0; tape_roll (0.30, -0.06) 0 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_tape_basket_right_p2 | basket (0.36, 0.23) -12; tape_roll (0.30, -0.00) -10 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_tape_basket_right_p3 | basket (0.41, 0.24) -13; tape_roll (0.28, -0.11) -16 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_tape_basket_right_p4 | basket (0.30, 0.25) -12; tape_roll (0.29, -0.03) -16 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_tape_basket_left_p1 | basket (0.42, -0.16) 0; tape_roll (0.26, 0.14) 20 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_tape_basket_left_p2 | basket (0.43, -0.10) -9; tape_roll (0.23, 0.09) 26 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.4 |
| twin_l1_tape_basket_left_p3 | basket (0.38, -0.19) -15; tape_roll (0.27, 0.14) 35 | 0 | 2 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_tape_basket_left_p4 | basket (0.37, -0.18) 17; tape_roll (0.29, 0.16) 39 | 0 | 1 | 0.0 | 2.1 | 2.1 | -4.6 |
| twin_l1_pill_basket_far_p1 | basket (0.46, 0.03) 0; pill_bottle (0.20, 0.00) 0 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l1_pill_basket_far_p2 | basket (0.45, 0.06) -8; pill_bottle (0.19, -0.00) -14 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l1_pill_basket_far_p3 | basket (0.43, -0.00) 4; pill_bottle (0.22, -0.00) -2 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l1_pill_basket_far_p4 | basket (0.47, 0.05) -18; pill_bottle (0.20, -0.04) -15 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l1_roar_basket_near_p1 | basket (0.30, -0.26) 0; roar_bottle (0.42, 0.16) 0 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_roar_basket_near_p2 | basket (0.31, -0.27) 16; roar_bottle (0.46, 0.19) -16 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_roar_basket_near_p3 | basket (0.30, -0.27) -4; roar_bottle (0.43, 0.11) 13 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l1_roar_basket_near_p4 | basket (0.32, -0.21) -15; roar_bottle (0.37, 0.15) -9 | 0 | 1 | 0.0 | 2.1 | 2.1 | -1.2 |
| twin_l2_tape_bits_p1 | basket (0.42, -0.12) 0; tape_roll (0.24, 0.16) 0; pill_bottle (0.36, 0.22) 0; bits (0.20, 0.30) 30, (0.28, 0.33) -45, (0.34, 0.38) 80, (0.45, 0.24) 10 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l2_tape_bits_p2 | basket (0.47, -0.07) -7; tape_roll (0.24, 0.11) -6; pill_bottle (0.33, 0.18) 14; bits (0.21, 0.25) 24, (0.30, 0.29) -38, (0.35, 0.34) 90, (0.45, 0.21) 17 | 0 | 2 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l2_tape_bits_p3 | basket (0.45, -0.15) 12; tape_roll (0.22, 0.11) 2; pill_bottle (0.30, 0.26) 10; bits (0.24, 0.32) 40, (0.24, 0.38) -51, (0.36, 0.32) 77, (0.50, 0.26) 3 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.5 |
| twin_l2_tape_bits_p4 | basket (0.46, -0.13) -12; tape_roll (0.29, 0.14) 16; pill_bottle (0.38, 0.22) -11; bits (0.26, 0.25) 32, (0.28, 0.30) -53, (0.29, 0.39) 75, (0.50, 0.26) 14 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l2_pill_case_p1 | basket (0.34, 0.26) 0; pill_bottle (0.30, -0.10) 0; clipper_case (0.42, -0.10) 180 | 0 | 1 | 0.0 | 4.0 | 4.0 | -1.9 |
| twin_l2_pill_case_p2 | basket (0.28, 0.24) -6; pill_bottle (0.27, -0.12) -14; clipper_case (0.39, -0.14) 199 | 0 | 1 | 0.0 | 4.0 | 4.0 | -2.4 |
| twin_l2_pill_case_p3 | basket (0.35, 0.27) -16; pill_bottle (0.32, -0.10) 4; clipper_case (0.47, -0.14) 190 | 0 | 1 | 0.0 | 4.0 | 4.0 | -2.4 |
| twin_l2_pill_case_p4 | basket (0.29, 0.30) -6; pill_bottle (0.27, -0.15) -11; clipper_case (0.40, -0.09) 172 | 0 | 1 | 0.0 | 4.0 | 4.0 | -1.9 |
| twin_l2_roar_tapes_p1 | basket (0.46, 0.08) 0; roar_bottle (0.22, -0.16) 0; tape_roll (0.22, 0.14) 0; tape_roll_2 (0.36, -0.24) 0; clipper_large (0.30, 0.30) 45; clipper_small (0.22, 0.32) -30 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.5 |
| twin_l2_roar_tapes_p2 | basket (0.43, 0.11) 10; roar_bottle (0.25, -0.11) -2; tape_roll (0.17, 0.15) 13; tape_roll_2 (0.36, -0.29) -13; clipper_large (0.28, 0.33) 45; clipper_small (0.23, 0.27) -37 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.5 |
| twin_l2_roar_tapes_p3 | basket (0.43, 0.03) -8; roar_bottle (0.21, -0.20) -6; tape_roll (0.20, 0.12) -17; tape_roll_2 (0.38, -0.22) -11; clipper_large (0.26, 0.30) 38; clipper_small (0.21, 0.34) -31 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.9 |
| twin_l2_roar_tapes_p4 | basket (0.46, 0.13) 11; roar_bottle (0.19, -0.18) -3; tape_roll (0.23, 0.09) 15; tape_roll_2 (0.36, -0.19) -16; clipper_large (0.29, 0.36) 38; clipper_small (0.20, 0.38) -13 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.5 |
| twin_l2_tape_bottles_p1 | basket (0.28, -0.26) 0; tape_roll (0.44, 0.16) 0; pill_bottle (0.22, 0.14) 0; roar_bottle (0.40, 0.34) 0 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l2_tape_bottles_p2 | basket (0.29, -0.27) 9; tape_roll (0.41, 0.11) -12; pill_bottle (0.17, 0.12) -18; roar_bottle (0.37, 0.32) -14 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l2_tape_bottles_p3 | basket (0.29, -0.24) 6; tape_roll (0.41, 0.14) 6; pill_bottle (0.28, 0.09) 18; roar_bottle (0.34, 0.32) -14 | 0 | 1 | 0.0 | 2.2 | 2.2 | -1.3 |
| twin_l2_tape_bottles_p4 | basket (0.33, -0.21) -5; tape_roll (0.43, 0.15) -16; pill_bottle (0.25, 0.09) 2; roar_bottle (0.41, 0.40) -10 | 0 | 1 | 0.0 | 2.2 | 2.2 | -2.6 |
| twin_l3_tape_p1 | tape_roll (0.19, 0.12) 0; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_tape_p2 | tape_roll (0.18, 0.13) -5; basket (0.50, 0.52) 17 | 16 | 1 | 0.0 | 3.4 | 3.4 | -8.5 |
| twin_l3_tape_p3 | tape_roll (0.18, 0.12) -12; basket (0.50, 0.46) 2 | 16 | 1 | 0.0 | 3.4 | 3.4 | -11.2 |
| twin_l3_tape_p4 | tape_roll (0.19, 0.14) 13; basket (0.56, 0.50) -10 | 17 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_pill_p1 | pill_bottle (0.21, 0.15) 0; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_pill_p2 | pill_bottle (0.18, 0.11) 2; basket (0.52, 0.52) 10 | 17 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_pill_p3 | pill_bottle (0.18, 0.10) 1; basket (0.51, 0.53) 9 | 16 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_pill_p4 | pill_bottle (0.18, 0.12) 6; basket (0.54, 0.44) -17 | 17 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_fold_kit_p1 | bits_kit (0.32, -0.06) 246; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_fold_kit_p2 | bits_kit (0.32, -0.09) 238; basket (0.45, 0.53) 12 | 16 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_fold_kit_p3 | bits_kit (0.30, -0.07) 228; basket (0.51, 0.49) 18 | 17 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l3_fold_kit_p4 | bits_kit (0.30, -0.06) 250; basket (0.49, 0.54) -13 | 16 | 1 | 0.0 | 3.4 | 3.4 | -8.3 |
| twin_l3_close_lid_p1 (kit closed) | headphone_case (0.46, -0.40) 210; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.2 (kit), 2.1 (case) | 3.4 | -1.8 |
| twin_l3_close_lid_p2 | headphone_case (0.48, -0.41) 218; basket (0.52, 0.45) 9 | 17 | 2 | 3.4 | 3.2 | 3.4 | -1.8 |
| twin_l3_close_lid_p3 | headphone_case (0.48, -0.41) 217; basket (0.45, 0.47) -16 | 17 | 2 | 4.0 | 3.2 | 3.4 | -1.8 |
| twin_l3_close_lid_p4 | headphone_case (0.46, -0.44) 196; basket (0.49, 0.51) -12 | 16 | 2 | 0.0 | 3.2 | 3.4 | -1.8 |
| twin_l4_tape_cup_p1 | tape_roll (0.19, 0.12) 0; paper_cup (0.20, 0.25) 0; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l4_tape_cup_p2 | tape_roll (0.19, 0.18) -14; paper_cup (0.26, 0.30) 2; basket (0.49, 0.52) 16 | 15 | 1 | 0.0 | 3.4 | 3.4 | -4.2 |
| twin_l4_tape_cup_p3 | tape_roll (0.19, 0.13) -3; paper_cup (0.21, 0.31) 4; basket (0.54, 0.50) 19 | 16 | 1 | 0.0 | 3.4 | 3.4 | -14.7 |
| twin_l4_tape_cup_p4 | tape_roll (0.18, 0.09) -10; paper_cup (0.21, 0.24) 15; basket (0.56, 0.46) 14 | 14 | 1 | 0.0 | 3.4 | 3.4 | -8.5 |
| twin_l4_two_tapes_p1 | tape_roll (0.19, 0.13) 0; tape_roll_2 (0.37, 0.28) 0; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l4_two_tapes_p2 | tape_roll (0.17, 0.08) -16; tape_roll_2 (0.32, 0.31) -3; basket (0.47, 0.49) -3 | 15 | 1 | 0.0 | 3.4 | 3.4 | -7.2 |
| twin_l4_two_tapes_p3 | tape_roll (0.20, 0.13) -15; tape_roll_2 (0.36, 0.30) 10; basket (0.50, 0.49) -19 | 16 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l4_two_tapes_p4 | tape_roll (0.19, 0.11) 4; tape_roll_2 (0.32, 0.29) -10; basket (0.46, 0.51) -4 | 16 | 1 | 0.0 | 3.4 | 3.4 | -8.6 |
| twin_l4_pill_roar_p1 | pill_bottle (0.20, 0.12) 0; roar_bottle (0.37, 0.28) 0; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l4_pill_roar_p2 | pill_bottle (0.18, 0.13) 4; roar_bottle (0.41, 0.29) -18; basket (0.47, 0.52) 5 | 16 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l4_pill_roar_p3 | pill_bottle (0.18, 0.08) -3; roar_bottle (0.32, 0.27) 14; basket (0.53, 0.49) 10 | 16 | 1 | 0.0 | 3.4 | 3.4 | -3.1 |
| twin_l4_pill_roar_p4 | pill_bottle (0.22, 0.16) 9; roar_bottle (0.31, 0.29) -12; basket (0.47, 0.51) -6 | 16 | 1 | 0.0 | 3.4 | 3.4 | -4.0 |
| twin_l4_three_p1 | tape_roll (0.19, 0.13) 0; pill_bottle (0.33, 0.28) 0; paper_cup (0.21, 0.26) 0; basket (0.50, 0.48) 1 | 0 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l4_three_p2 | tape_roll (0.22, 0.17) -18; pill_bottle (0.31, 0.30) -7; paper_cup (0.18, 0.29) 3; basket (0.49, 0.46) 1 | 14 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |
| twin_l4_three_p3 | tape_roll (0.20, 0.13) -2; pill_bottle (0.33, 0.29) -16; paper_cup (0.18, 0.27) -2; basket (0.47, 0.52) 12 | 14 | 1 | 0.0 | 3.4 | 3.4 | -3.1 |
| twin_l4_three_p4 | tape_roll (0.18, 0.13) 13; pill_bottle (0.28, 0.30) 11; paper_cup (0.18, 0.32) -4; basket (0.54, 0.51) 18 | 15 | 1 | 0.0 | 3.4 | 3.4 | -3.0 |

Settle note (as in the morning): the 2.1-2.2 mm displacements are the 2 mm start lift of every placed body, 3.4 mm on the full desk
is the bits' drop plus the hex settling flat, 4.0 mm in twin_l2_pill_case is the seated clippers finding their rest on the slot hulls.
"Worst contact" below -3 mm (down to -14.7 mm) is MuJoCo's cylinder-plane report on impact of a cylinder collider (cup, tape roll, ROAR
bottle) during the 2 mm drop; the body ends at rest, nothing is pushed. The start-contact check (before any step) is 0.0 mm on
every placement except twin_l2_pill_case (-0.7 mm, the seated clippers against the slot hulls, excluded as group-internal anyway).

## Reward ladders (`twin_tasks.py`)

Unchanged from the morning build and still the approved dense scorers: pick-and-place ladder (`pick_place_scorer`; approach 0.10,
contact 0.20, lifted 0.40, transport 0.70, over 0.85, inside 1.00, with the within-stage progress, running max, drop record and
freeze) for twin_l1_*, twin_l2_*, twin_l3_tape, twin_l3_pill; `FoldKitScorer` (flap_grasped 0.10, inner_past_vertical 0.30, inner_flat
0.50, outer_past_vertical 0.70, closed 1.00) for twin_l3_fold_kit. Thresholds, footprints and cursors as described in the module
docstring and the morning report's section "Reward ladders" (constants APPROACH_M 0.05, CONTACT_WINDOW 10, LIFT_M 0.03,
TRANSPORT_NEAR_M 0.10, INSIDE_HOLD_S 0.5, DROP_LIFT_M 0.01, FOLD_VERTICAL_DEG 90, FOLD_FLAT_DEG 170).

New: `sequence_scorer(bodies, cursors)` replaces `two_object_scorer`. One pick-and-place ladder per object in task order, the i-th
scaled to [i/n, (i+1)/n] (two objects: 0.05 0.10 0.20 0.35 0.425 0.50 then 0.55 ... 1.00, as before; three objects: 0.033 0.067 0.133
0.233 0.283 0.333, 0.367 ... 0.667, 0.70 ... 1.00); a ladder only runs once the previous object is placed (no carried time, drops or
progress before its turn), so the dense value stays monotone. `objects.<body>` in the record keeps each object's carried time,
drop, drops and placement; `drop` / `drops` / `carried_time_s` aggregate. Cursors: object i's stages point at script steps 3i
(approach, contact), 3i+1 (lifted, transport), 3i+2 (over), 3i+3 (inside = the next grasp step, or "return to home" for the last).

twin_l3_close_lid reuses `easy_tasks.CloseLidScorer` (reach 0.1, quarter 0.4, three_quarters 0.7, closed 1.0; no dense value) and
the close_lid T071 script; the twin's `headphone_case` / `headphone_lid` / `lid_hinge` are the close_lid asset, so no adaptation was
needed (checked below). Unlike the package close_lid scene the twin's case is a free body (0.16 kg), so a push on the lid can slide it.

## Verification without a policy

- All 64 scenes load (`mujoco.MjModel.from_xml_path`; ngeom 142-154 for L1, 170-213 for L2, 437 on the full desk), every IK target
  passes (position error < 5 mm: 0.0 everywhere except the far lid wall of twin_l3_close_lid p2/p3 at 3.4 / 4.0 mm; straight-down within
  5 deg where required, inside the soft limits, no table contact: 2 targets per L1/L2 placement, 2-3 for L3 (both kit flaps + basket for
  the fold, both lid walls + basket for the lid), 3-4 for L4), every placed body settles < 5 mm (max 4.0 mm, the seated clippers; the
  closed kit 3.2 mm), start contact > -1 mm everywhere (the table above; -0.1 mm on the lid placements p2-p4).
- Renders (`renders/`): `<config>_p1_ext.png` + `<config>_p1_wrist.png` at the park pose after the robot's 0.5 s settle for every
  configuration, `sheet_<config>.png` (ext of p1..p4), `contact_sheet_p1.png` (4 rows = levels, 4 columns = configurations a-d, built
  from the p1 renders on disk). Checked by eye: L3 (d) shows the closed blue kit block in the middle of the desk and the open headphone
  case far left with the lid toward the robot; the four L1 baskets sit right / far left / far centre / near left with the object on the other side,
  the L2 desks differ in target, distractor set and basket side, the seated clippers show in the case from the wrist camera, the full
  desk's targets sit in the pocket right of the kit with the basket at the twin pose.
- Scorer checks (`scorer_check.py`, kinematic drive as in the morning; dense curve monotone, asserted):
  - L1 `twin_l1_tape_basket_right_p2`: run A (release 10 cm outside) contact 0.97 s, lifted 1.20, transport 2.10, drop at 2.13 s 0.100 m
    from the footprint edge, dense frozen at 0.700; run B contact, lifted, transport 1.50 (0.702), over 1.73, inside 3.27 s = 1.000.
  - L2 `twin_l2_pill_case_p2` (pill bottle, side pinch): run A approach 0.93 (0.100), contact 0.97, lifted 1.20, transport 2.10 (0.702),
    drop 0.0985 m from the edge, final 0.7022; run B over 1.90, inside 3.27 s = 1.000.
  - L3 `twin_l3_fold_kit_p2` (jittered kit): flap_grasped 0.77 s (0.127), inner_past_vertical 1.33 (0.319), inner_flat 1.67 (0.527),
    outer_past_vertical 2.17 (0.728), closed 2.50 s = 1.000; angles 180 / 180 deg.
  - L4 (d) `twin_l4_three_p2` (tape roll, then pill bottle, then cup): tape contact 0.80 s (0.067), lifted 0.133, transport 0.233 (0.262),
    over 0.283, inside 2.60 s (0.333); pill approach 3.60 (0.367), contact 0.400, transport 0.567 (0.587), over 0.617, inside 5.40 s
    (0.667); cup approach 6.37 (0.700), contact 0.733, lifted 0.800, transport 0.900 (0.904), over 0.950, inside 8.20 s = 1.000; all three
    placed, 0 drops.
  - L3 (d) `--lid twin_l3_close_lid_p2`: IK pinch at the near lid wall ok (0.0 mm), reach 0.1 at the pinch, the hinge driven 0 -> 180
    deg gives quarter 0.4, three_quarters 0.7, closed 1.0 (the lid closes onto the free case body; the closed-stack check of the driver is
    kinematic). The same run on `twin_l3_tape_p1` (open kit, case at the twin pose) misses the wall by 17.6 mm, the reason the kit had to be
    closed for this configuration.
- Registry: `tasks.py layouts` (default) prints the 30 old variants unchanged; `TWIN=1 ... layouts` prints the 64 twin placements
  (16 configurations x p1-p4, L1 -> L4); `overrides twin_l4_three_p3` returns the 10-step script. `layout_name` strips `_p<k>` like
  `_v<k>`, `variants()` lists both kinds.
- Sim robot path: `RebotB601Sim(scene="twin_l4_three_p3", auto_subtasks=True)` connects in 0.9 s, the scorer resolves to
  `Sequence_tape_roll_pill_bottle_paper_cup`, `poll()` returns cursor 0, `episode_extra()` carries reward_dense / dense_timeline /
  carried_time_s / objects / drop / drops.
- `run_sim_eval.sh` with TWIN=1 adds `--env.robot.episodes=4` before `SIM_EVAL_ARGS` (draccus keeps the last value of a repeated flag,
  checked on a toy config), so `SIM_EVAL_ARGS="--env.robot.episodes=2"` still overrides; the config copy without the top camera's
  background fields is unchanged.

## Running the twin eval on a checkpoint

    TWIN=1 PARALLEL=3 lerobot/scripts/run_sim_eval.sh outputs/<run>/checkpoints/<step>/pretrained_model

(same arguments as before: `[FRESH_OUTPUT_DIRECTORY] [CONFIG]`, `LAYOUTS="twin_l1_tape_basket_left_p2 twin_l4_three_p1"` to pick
placements, `SIM_EVAL_ARGS="--env.robot.episodes=2 --env.processor.reset.control_time_s=40"` for a short pass). The set is 64
placements x 4 episodes x 60 s = 256 episodes, about the size of the default set's 30 x 8. `sim_summary.py` rolls the placements up
into `reward.<config>` / `reward_dense.<config>` and the tiers `twin_l1 .. twin_l4` (one mean per configuration, then per level). Not
run here.

Regenerate: `MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/digital_twin/envs/build_twin_envs.py` (all 16
configurations, renders and checks.json; `--configs <names>` for a subset, `--check-base` to test hand-edited base poses only).

## Open points

- L3 (d) is the one configuration that changes a twin body's state (the kit closed) and places a target outside the README region /
  reach disc (the case's base half at 0.61 m, the lid walls at 0.44-0.47 m): with the open kit at its twin pose the full desk has no
  in-reach spot for the open case (twin-pose walls at 0.51 / 0.68 m). The closed kit is a re-parametrisation of the fold asset
  (baked angles), not a new asset; the FoldKitScorer must not be pointed at this scene. The case is a free body, so pushing the lid can
  slide the case (the package close_lid scene fixes its case to the table).
- On the full desk the basket is anchored at the twin pose (0.69 m from the base, outside the reach rule) with +-6 cm / +-20 deg
  jitter, as the morning's hard layout: the open kit leaves no in-reach room for it. The four L3 / L4 configurations therefore differ by
  task and target placement, not by basket side. L1 and L2 have the basket right / far left / far centre / near left as asked.
- The full-desk pockets are small (tape and pill bottle at x 0.17-0.22, the second object at x 0.31-0.41 / y 0.26-0.31), so the L3 / L4
  jittered targets stay within a few cm of their base poses; the clutter jitter moves 14-17 of the other bodies instead.
- `twin_l2_pill_case`: the clippers sit on the slot hulls (z 0.025 in the case frame, 4 mm settle), not on the nominal slot floor (the
  convex hulls leave 5-7 mm of penetration there); visually they are in the case.
- The sequence scorer is strict about the order (as the morning's two-object scorer): a policy that places the second object first
  earns nothing for it until the first one is placed.

## Smoke test 2026-10-30 (resume ckpt 600, twin_l1_tape_basket_right_p1, 1 episode)

Loop runs end to end (policy load, 60 s episode, videos, subtask_events.json with the dense record, summary tables). Fixes made on the way: numpy bools in the dense record (json), drop rule extended to off-desk and at-rest objects (the infinite table collider lets a knocked object roll 2 m away at z = 0), raw gains off_desk / basket_moved_m / displaced_m. Policy behaviour in both smoke episodes: approach + contact at ~26-29 s, tape knocked over, no lift (reward 0.2, dense 0.37); in one run the arm pushed the basket ~25 cm. Output outputs/twin_eval_smoke_2026-10-30.
