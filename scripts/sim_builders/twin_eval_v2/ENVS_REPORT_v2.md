# Twin eval suite v2 (2026-11-08): hand-designed placements

The 10-30 build made p2-p4 by jittering p1 (xy +-6 cm, yaw +-20 deg), so 48 of the 64 scenes were near copies of 16 configurations
(user, 2026-11-08). v2 writes every placement by hand: target side and distance, basket pose and orientation, distractor
arrangement, and on the full desk the bit kit open / mirrored / closed / closed and moved right with the clutter re-arranged.
Same names, tasks, scorers and checks as v1 (`ENVS_REPORT_v1.md` here, builder `build_twin_envs_v2.py`, v1 scenes in `assets/sim_sources/twin_eval_v2/scenes_v1_backup/`, outside the repo).
Checks: targets inside the README region and the 0.50 m reach disc (basket 0.48 m), 2 cm inside the desk, 2 cm clearance, no start
contact < 1 mm, IK reach (top pinch / side pinch / 5 cm over the rim / both flap edges / both lid walls), 1 s settle < 5 mm. All 64 pass.

Findings from the free-space scan that shaped the design (`freemap.py`, `feasible.py` in the session scratchpad, not kept):
- A straight-down pinch on a tall object (pill bottle, cup; top z 0.113) fails at x 0.46 everywhere and at x 0.42 for |y| >= 0.20.
- The parked arm hangs over x 0.16-0.30 on the centre line: a basket there penetrates it (14-37 mm), a 3 cm tape roll does not.
- With the kit open, only two pockets are free (near-right x 0.18-0.28 / y 0.10-0.30 once the cup moves, far-right x 0.30-0.42 / y 0.26-0.28);
  the open kit can shift at most to (0.32, 0.08) or rotate to 275.6 / 305.6 deg. Closed, it frees the near-left strip and the band to its right;
  closed and moved to (0.38, 0.14, 155.6) it frees the whole left half and the centre.

## Bits pass (2026-11-08 evening, BUILT)

The set was tape heavy (8 of 16 configurations started with the tape). Replaced: l1 tape_basket_left -> **l1 bit_single**,
l1 roar_basket_near -> **l1 kit_alone** (fold only, no basket, 90 s budget), l2 tape_bits -> **l2 bit_clutter**, l4 two_tapes ->
**l4 two_bits** (the twin's bit pile cut to the two targets; scored in either order). l4 three is now "the pill bottle, the cup and
the tape roll". The removed scene folders are in `scenes_v2_removed_2026-11-08/`, the builder before this pass in
`build_twin_envs_v2_before_bits_2026-11-08.py` (both in `assets/sim_sources/twin_eval_v2/`).

Reach rule changed for the new placements: the IK test is the rule (the 0.50 m disc, the 0.48 m basket disc and the data-coverage
region are dropped), and an object pinch may lean outward up to 30 deg. Min-tilt scan at bit height: straight down reaches ~0.50 m,
15 deg x 0.54, 30 deg x 0.62 at the centre and y +-0.50 beside the robot; 5 cm over the basket rim reaches the whole desk. The
unchanged configurations keep their earlier placements (they pass under the old, stricter rules).

| scene | kit | note | placed bodies (x, y) yaw |
|---|---|---|---|
| twin_l1_bit_single_p1 | - | bit far-right at 30, basket out on the left side | basket (0.26, -0.42) 90; bit_1 (0.54, 0.26) 30 |
| twin_l1_bit_single_p2 | - | bit far-left across the gripper (90), basket far-right corner | basket (0.54, 0.36) 0; bit_1 (0.58, -0.20) 90 |
| twin_l1_bit_single_p3 | - | bit at the far end straight ahead, basket out on the right side | basket (0.26, 0.40) 90; bit_1 (0.62, 0.02) 0 |
| twin_l1_bit_single_p4 | - | bit wide right beside the robot at -45, basket far-left corner | basket (0.54, -0.36) 0; bit_1 (0.26, 0.48) -45 |
| twin_l1_kit_alone_p1 | - | kit alone at its twin pose | bits_kit (0.32, -0.06) 246 |
| twin_l1_kit_alone_p2 | - | kit mirrored (bits flap on the right), centre-right | bits_kit (0.32, 0.08) 66 |
| twin_l1_kit_alone_p3 | - | kit turned -30 on the left side | bits_kit (0.32, -0.12) 216 |
| twin_l1_kit_alone_p4 | - | kit turned +60 on the right side | bits_kit (0.34, 0.14) 306 |
| twin_l2_bit_clutter_p1 | - | basket far-left corner, bit far-right, case at the far end, tape centre, pill near-left | clipper_case (0.62, 0.08) 90; basket (0.54, -0.36) 0; bit_1 (0.54, 0.26) 30; tape_roll (0.38, 0.02) 0; pill_bottle (0.22, -0.34) 0 |
| twin_l2_bit_clutter_p2 | - | basket out on the right side, bit far-left, tape at the far end, pill in the carry path, case near-left | basket (0.26, 0.40) 90; bit_1 (0.58, -0.20) 90; tape_roll (0.62, 0.04) 0; pill_bottle (0.42, 0.18) 0; clipper_case (0.24, -0.38) 0 |
| twin_l2_bit_clutter_p3 | - | basket out on the left side, bit at the far end in the clutter: tape and pill beside it, case in front of it | clipper_case (0.40, 0.04) 90; basket (0.30, -0.42) 90; bit_1 (0.62, 0.02) 0; tape_roll (0.58, -0.18) 0; pill_bottle (0.62, 0.18) 0 |
| twin_l2_bit_clutter_p4 | - | basket far-right corner, bit wide right beside the robot with the pill next to it, tape far-left, case centre | basket (0.54, 0.36) 0; bit_1 (0.26, 0.48) -45; pill_bottle (0.22, 0.36) 0; tape_roll (0.58, -0.24) 0; clipper_case (0.40, -0.08) 0 |
| twin_l4_two_bits_p1 | open | A twin desk, one bit between kit and basket, one at the far end among the clippers | bit_1 (0.50, 0.24) 30; bit_2 (0.64, 0.02) -45; basket (0.50, 0.48) 1 |
| twin_l4_two_bits_p2 | closed | B kit closed, bits far-left and far-right | bit_1 (0.44, -0.24) 60; bit_2 (0.50, 0.26) 0; basket (0.50, 0.48) 1 |
| twin_l4_two_bits_p3 | open | C kit mirrored, cup far-right pocket, one bit at the far end, one wide right beside the robot | bits_kit (0.32, -0.06) 66; paper_cup (0.33, 0.28) 0; bit_1 (0.64, 0.02) 0; bit_2 (0.30, 0.42) 90; basket (0.50, 0.48) 1 |
| twin_l4_two_bits_p4 | closed | D kit closed and moved right, bits far-left and wide left beside the robot, cup centre-left | bits_kit (0.38, 0.14) 156; bit_1 (0.44, -0.22) 45; bit_2 (0.24, -0.44) -30; paper_cup (0.30, -0.16) 0; basket (0.50, 0.48) 1 |

Bit grasp fixed (2026-11-08): the sim gripper could not hold a bit (fingers closed fully, the bit squeezed out or sank into the
desk). Causes: the finger collision hulls (64 vertices) end ~3 mm above the visual tip and taper to 4 mm, and the bit's hex-mesh hull
does not give stable contacts. Fix: (1) a box pad on each fingertip in `scenes/rebot_b601_dm_photo.xml` (10 x 13.5 mm face from
the visual tip up, 1 mm past the hull's inner face, condim 4, friction 1.0, solref 0.004; finger-finger contact was already excluded,
so an empty gripper still closes to 0), backup `gripper_pads_2026-11-08/rebot_b601_dm_photo_before_pads.xml`; (2) every bit's
collision is a box of the mesh bounds (mesh kept as the visual, colours kept), built by `build_twin_envs_v2.py`. Pads and boxes are
collision-only (group 3): the camera images do not change. Scripted vertical pinch, ramped close, fingers across the bit:

| object | before | with pads |
|---|---|---|
| bit at (0.30, 0.00) | 0 cm | 8.3 cm lift, slides stop at 4.1 mm |
| bit at (0.36, 0.12) | 0 cm | 8.1 cm |
| tape roll (rim pinch, scene pose) | 2.2 cm | 2.2 cm |
| pill bottle (top pinch, scene pose) | 0 cm | 4.2 cm |

The pads change the gripper for every sim scene (the old set too): sim numbers before and after 2026-11-08 are not comparable.

## v2 placements (all 16 original configurations; the four replaced ones above are superseded)

Frame: x toward the far edge / top camera, y to the robot's right, yaw in degrees. Renders: `renders/sheet_<config>.png` (the current 16 sheets only).
| scene | kit | note | placed bodies (x, y) yaw |
|---|---|---|---|
| twin_l1_tape_basket_right_p1 | - | v1 base: tape mid-left of the basket, basket right | basket (0.36, 0.22) 0; tape_roll (0.30, -0.06) 0 |
| twin_l1_tape_basket_right_p2 | - | tape near-right, basket far-left turned 90 | basket (0.44, -0.14) 90; tape_roll (0.20, 0.28) 0 |
| twin_l1_tape_basket_right_p3 | - | tape far-left, basket near-right | basket (0.28, 0.30) 0; tape_roll (0.46, -0.08) 0 |
| twin_l1_tape_basket_right_p4 | - | tape right in front of the robot, basket far behind it | basket (0.47, 0.03) 0; tape_roll (0.19, 0.00) 0 |
| twin_l1_tape_basket_left_p1 | - | v1 base: basket far-left, tape right | basket (0.42, -0.16) 0; tape_roll (0.26, 0.14) 20 |
| twin_l1_tape_basket_left_p2 | - | basket near-left turned 90, tape far-right | basket (0.30, -0.25) 90; tape_roll (0.42, 0.24) 0 |
| twin_l1_tape_basket_left_p3 | - | basket right at 45, tape near-left close to the robot | basket (0.36, 0.28) 45; tape_roll (0.17, -0.20) 0 |
| twin_l1_tape_basket_left_p4 | - | basket far-right, tape close to the robot on its left | basket (0.42, 0.22) 0; tape_roll (0.18, -0.10) 0 |
| twin_l1_pill_basket_far_p1 | - | pill near-right, basket far-left | basket (0.42, -0.18) 0; pill_bottle (0.22, 0.28) 0 |
| twin_l1_pill_basket_far_p2 | - | basket near-left, pill adjacent to its right wall | basket (0.30, -0.26) 0; pill_bottle (0.24, -0.06) 0 |
| twin_l1_pill_basket_far_p3 | - | basket far-right turned 90, pill far-left | basket (0.36, 0.30) 90; pill_bottle (0.42, -0.12) 0 |
| twin_l1_pill_basket_far_p4 | - | basket far-right turned 30, pill near-left | basket (0.40, 0.24) 30; pill_bottle (0.22, -0.24) 0 |
| twin_l1_roar_basket_near_p1 | - | v1 base: basket near-left, bottle far-right | basket (0.30, -0.26) 0; roar_bottle (0.42, 0.16) 0 |
| twin_l1_roar_basket_near_p2 | - | basket far centre-right, bottle near-left | basket (0.44, 0.10) 0; roar_bottle (0.22, -0.18) 0 |
| twin_l1_roar_basket_near_p3 | - | basket right turned 90, bottle far centre-left | basket (0.30, 0.32) 90; roar_bottle (0.46, -0.04) 0 |
| twin_l1_roar_basket_near_p4 | - | basket left at 45, bottle near centre-right | basket (0.40, -0.20) 45; roar_bottle (0.20, 0.12) 0 |
| twin_l2_tape_bits_p1 | - | basket far-left, tape near-right, pill far-right, bits spread over the desk | basket (0.42, -0.14) 0; tape_roll (0.22, 0.28) 0; pill_bottle (0.38, 0.24) 0; bit_1 (0.20, -0.20) 30; bit_2 (0.30, 0.04) -45; bit_3 (0.46, 0.08) 80; bit_4 (0.26, 0.40) 10 |
| twin_l2_tape_bits_p2 | - | basket near-right, tape far-left, pill near-left, bits spread along the carry path | basket (0.28, 0.30) 0; tape_roll (0.44, -0.12) 0; pill_bottle (0.20, -0.22) 0; bit_1 (0.22, 0.00) 10; bit_2 (0.36, 0.08) -60; bit_3 (0.46, 0.12) 45; bit_4 (0.30, -0.26) 100 |
| twin_l2_tape_bits_p3 | - | basket far-right turned 90, tape near centre-left with the bits tight around it, pill far-left | basket (0.36, 0.30) 90; tape_roll (0.20, -0.06) 0; pill_bottle (0.38, -0.24) 0; bit_1 (0.20, 0.04) 30; bit_2 (0.28, -0.02) -45; bit_3 (0.21, -0.16) 80; bit_4 (0.27, -0.12) 10 |
| twin_l2_tape_bits_p4 | - | basket near-left turned 90, tape far-right with the pill next to it, bits in the four corners | basket (0.30, -0.25) 90; tape_roll (0.38, 0.30) 0; pill_bottle (0.42, 0.18) 0; bit_1 (0.19, 0.02) 0; bit_2 (0.22, 0.36) 90; bit_3 (0.46, 0.00) 45; bit_4 (0.48, -0.20) -30 |
| twin_l2_pill_case_p1 | - | basket right, pill near-left, case far-left | clipper_case (0.44, -0.14) 180; basket (0.36, 0.26) 0; pill_bottle (0.22, -0.22) 0 |
| twin_l2_pill_case_p2 | - | basket far-left, case near-right turned 90, pill near centre | clipper_case (0.22, 0.30) 90; basket (0.44, -0.14) 0; pill_bottle (0.22, 0.08) 0 |
| twin_l2_pill_case_p3 | - | basket far centre, pill far-right, case left turned 90: pill and case on opposite sides | clipper_case (0.30, -0.26) 90; basket (0.46, 0.02) 0; pill_bottle (0.30, 0.32) 0 |
| twin_l2_pill_case_p4 | - | basket right turned 90, pill near-left, case between pill and basket | basket (0.36, 0.30) 90; pill_bottle (0.20, -0.22) 0; clipper_case (0.30, -0.02) 0 |
| twin_l2_roar_tapes_p1 | - | basket right, bottle near-left, tapes far-left and far centre, clippers near centre | basket (0.34, 0.30) 0; roar_bottle (0.20, -0.22) 0; tape_roll (0.44, -0.22) 0; tape_roll_2 (0.40, 0.06) 0; clipper_large (0.22, 0.04) 45; clipper_small (0.30, -0.04) -30 |
| twin_l2_roar_tapes_p2 | - | basket near-left, bottle far-right, tapes near-right and far centre-left, clippers centre | basket (0.28, -0.26) 0; roar_bottle (0.40, 0.28) 0; tape_roll (0.18, 0.30) 0; tape_roll_2 (0.46, -0.02) 0; clipper_large (0.30, 0.08) 90; clipper_small (0.36, 0.16) 0 |
| twin_l2_roar_tapes_p3 | - | basket far-left, bottle near-right, tapes near-left and far-right, clippers centre and right | basket (0.44, -0.16) 0; roar_bottle (0.22, 0.26) 0; tape_roll (0.20, -0.10) 0; tape_roll_2 (0.44, 0.16) 0; clipper_large (0.32, 0.08) 45; clipper_small (0.30, 0.34) -30 |
| twin_l2_roar_tapes_p4 | - | basket right turned 90, bottle far-left with a tape beside it, other tape near centre, clippers near-left and far centre | basket (0.33, 0.32) 90; roar_bottle (0.46, -0.14) 0; tape_roll (0.42, -0.26) 0; tape_roll_2 (0.20, -0.02) 0; clipper_large (0.20, -0.24) 45; clipper_small (0.46, 0.06) -30 |
| twin_l2_tape_bottles_p1 | - | v1 base: basket near-left, tape far-right, bottles right | basket (0.28, -0.26) 0; tape_roll (0.44, 0.16) 0; pill_bottle (0.22, 0.14) 0; roar_bottle (0.40, 0.34) 0 |
| twin_l2_tape_bottles_p2 | - | basket far-left, tape near-left with the pill adjacent, orange bottle near-right | basket (0.44, -0.16) 0; tape_roll (0.20, -0.20) 0; pill_bottle (0.20, -0.08) 0; roar_bottle (0.22, 0.26) 0 |
| twin_l2_tape_bottles_p3 | - | basket near-right, tape far-left between the two bottles | basket (0.26, 0.28) 0; tape_roll (0.46, -0.12) 0; pill_bottle (0.48, 0.08) 0; roar_bottle (0.36, -0.26) 0 |
| twin_l2_tape_bottles_p4 | - | basket left turned 90, tape near, orange bottle between tape and basket, pill far-right | basket (0.40, -0.22) 90; tape_roll (0.20, 0.08) 0; roar_bottle (0.32, 0.04) 0; pill_bottle (0.42, 0.26) 0 |
| twin_l3_tape_p1 | open | A twin desk, tape in the near-right pocket | tape_roll (0.19, 0.12) 0; basket (0.50, 0.48) 1 |
| twin_l3_tape_p2 | closed | B kit closed, tape near-left, cup moved right of the kit | tape_roll (0.22, -0.26) 0; paper_cup (0.42, 0.14) 0; basket (0.50, 0.48) 1 |
| twin_l3_tape_p3 | open | C kit mirrored, cup to the far-right pocket, tape near-right | bits_kit (0.32, -0.06) 66; paper_cup (0.33, 0.28) 0; tape_roll (0.20, 0.16) 0; basket (0.50, 0.48) 1 |
| twin_l3_tape_p4 | closed | D kit closed and moved right, tape far-left, cup centre | bits_kit (0.38, 0.14) 156; tape_roll (0.44, -0.10) 0; paper_cup (0.30, -0.08) 0; basket (0.50, 0.48) 1 |
| twin_l3_pill_p1 | open | A twin desk, pill in the near-right pocket | pill_bottle (0.21, 0.15) 0; basket (0.50, 0.48) 1 |
| twin_l3_pill_p2 | closed | B kit closed, pill left, cup moved right of the kit | pill_bottle (0.30, -0.26) 0; paper_cup (0.36, 0.16) 0; basket (0.50, 0.48) 1 |
| twin_l3_pill_p3 | open | C kit mirrored, pill in the far-right pocket, cup stays | bits_kit (0.32, -0.06) 66; pill_bottle (0.33, 0.28) 0; basket (0.50, 0.48) 1 |
| twin_l3_pill_p4 | closed | D kit closed and moved right, pill far-left, cup near centre | bits_kit (0.38, 0.14) 156; pill_bottle (0.42, -0.12) 0; paper_cup (0.22, -0.04) 0; basket (0.50, 0.48) 1 |
| twin_l3_fold_kit_p1 | open | A twin desk, kit at its twin pose | bits_kit (0.32, -0.06) 246; basket (0.50, 0.48) 1 |
| twin_l3_fold_kit_p2 | open | kit mirrored: bits flap on the other side | bits_kit (0.32, -0.06) 66; basket (0.50, 0.48) 1 |
| twin_l3_fold_kit_p3 | open | kit shifted right, cup moved to the near-left strip | bits_kit (0.32, 0.08) 246; paper_cup (0.22, -0.26) 0; basket (0.50, 0.48) 1 |
| twin_l3_fold_kit_p4 | open | kit turned 60 deg, cup moved to the near-left strip | bits_kit (0.30, 0.00) 306; paper_cup (0.22, -0.26) 0; basket (0.50, 0.48) 1 |
| twin_l3_close_lid_p1 | closed | v1 base: kit closed at its twin pose, case far-left, lid toward the robot | headphone_case (0.46, -0.40) 210; basket (0.50, 0.48) 1 |
| twin_l3_close_lid_p2 | closed | kit closed and moved right, case near the centre-left | bits_kit (0.38, 0.14) 156; headphone_case (0.34, -0.20) 210; basket (0.50, 0.48) 1 |
| twin_l3_close_lid_p3 | closed | kit closed and moved right, case mid-left turned so the lid opens to the robot's left | bits_kit (0.38, 0.14) 156; headphone_case (0.42, -0.16) 255; basket (0.50, 0.48) 1 |
| twin_l3_close_lid_p4 | closed | kit closed at its twin pose, case at the far end of the feasible band | headphone_case (0.50, -0.34) 225; basket (0.50, 0.48) 1 |
| twin_l4_tape_cup_p1 | open | A twin desk, tape beside the cup in the near-right pocket | tape_roll (0.19, 0.12) 0; paper_cup (0.20, 0.25) 0; basket (0.50, 0.48) 1 |
| twin_l4_tape_cup_p2 | closed | B kit closed, tape near-left, cup right of the kit | tape_roll (0.22, -0.26) 0; paper_cup (0.42, 0.14) 0; basket (0.50, 0.48) 1 |
| twin_l4_tape_cup_p3 | open | C kit mirrored, cup in the far-right pocket, tape near-right | bits_kit (0.32, -0.06) 66; paper_cup (0.33, 0.28) 0; tape_roll (0.20, 0.14) 0; basket (0.50, 0.48) 1 |
| twin_l4_tape_cup_p4 | closed | D kit closed and moved right, tape far-left, cup near centre | bits_kit (0.38, 0.14) 156; tape_roll (0.44, -0.14) 0; paper_cup (0.22, -0.02) 0; basket (0.50, 0.48) 1 |
| twin_l4_two_tapes_p1 | open | A twin desk, one tape per pocket | tape_roll (0.19, 0.13) 0; tape_roll_2 (0.37, 0.28) 0; basket (0.50, 0.48) 1 |
| twin_l4_two_tapes_p2 | closed | B kit closed, one tape near-left, the other far right of the kit | tape_roll (0.22, -0.26) 0; tape_roll_2 (0.44, 0.12) 0; basket (0.50, 0.48) 1 |
| twin_l4_two_tapes_p3 | open | C kit mirrored, both tapes in the near-right pocket, cup far-right | bits_kit (0.32, -0.06) 66; paper_cup (0.33, 0.28) 0; tape_roll (0.20, 0.14) 0; tape_roll_2 (0.22, 0.26) 0; basket (0.50, 0.48) 1 |
| twin_l4_two_tapes_p4 | closed | D kit closed and moved right, tapes far-left and near centre, cup left | bits_kit (0.38, 0.14) 156; tape_roll (0.44, -0.16) 0; tape_roll_2 (0.22, 0.00) 0; paper_cup (0.30, -0.24) 0; basket (0.50, 0.48) 1 |
| twin_l4_pill_roar_p1 | open | A twin desk, pill near-right, orange bottle far-right pocket | pill_bottle (0.20, 0.12) 0; roar_bottle (0.37, 0.28) 0; basket (0.50, 0.48) 1 |
| twin_l4_pill_roar_p2 | closed | B kit closed, orange bottle near-left, pill right of the kit | roar_bottle (0.22, -0.26) 0; pill_bottle (0.42, 0.12) 0; basket (0.50, 0.48) 1 |
| twin_l4_pill_roar_p3 | open | C kit mirrored, both bottles in the near-right pocket, cup far-right | bits_kit (0.32, -0.06) 66; paper_cup (0.33, 0.28) 0; roar_bottle (0.20, 0.14) 0; pill_bottle (0.22, 0.26) 0; basket (0.50, 0.48) 1 |
| twin_l4_pill_roar_p4 | closed | D kit closed and moved right, pill far-left, orange bottle near centre, cup left | bits_kit (0.38, 0.14) 156; pill_bottle (0.42, -0.14) 0; roar_bottle (0.24, -0.06) 0; paper_cup (0.30, -0.24) 0; basket (0.50, 0.48) 1 |
| twin_l4_three_p1 | open | A twin desk, tape and cup near-right, pill far-right pocket | tape_roll (0.19, 0.13) 0; pill_bottle (0.33, 0.28) 0; paper_cup (0.21, 0.26) 0; basket (0.50, 0.48) 1 |
| twin_l4_three_p2 | closed | B kit closed, tape and pill on the near-left strip, cup right of the kit | tape_roll (0.20, -0.26) 0; pill_bottle (0.30, -0.26) 0; paper_cup (0.42, 0.14) 0; basket (0.50, 0.48) 1 |
| twin_l4_three_p3 | open | C kit mirrored, cup far-right pocket, tape and pill near-right | bits_kit (0.32, -0.06) 66; paper_cup (0.33, 0.28) 0; tape_roll (0.20, 0.14) 0; pill_bottle (0.22, 0.26) 0; basket (0.50, 0.48) 1 |
| twin_l4_three_p4 | closed | D kit closed and moved right, tape far-left, pill near centre, cup left | bits_kit (0.38, 0.14) 156; tape_roll (0.44, -0.16) 0; pill_bottle (0.22, -0.02) 0; paper_cup (0.32, -0.24) 0; basket (0.50, 0.48) 1 |
