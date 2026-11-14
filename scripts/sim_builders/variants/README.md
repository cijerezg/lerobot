# Sim placement variants (2026-10-28)

Six placement variants per eval layout (`scenes/<layout>_v1..v6` for bottle_basket, clippers_only, close_lid, ep0, ep3):
the base scene with only the task objects' xy + yaw re-sampled. `generate_variants.py` writes them (seed 20261028,
per-layout sub-seed = index in bottle_basket, clippers_only, ep0, ep3, close_lid; re-running is byte-identical, verified
by md5 over all 91 files), `checks.json` holds every sample's tries, IK solutions and settle numbers, and
`contact_<layout>.png` shows the deployed top-camera composite of the base (region outline in yellow, robot base in cyan,
close_lid demo centres in magenta) and the six variants.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/variants/generate_variants.py [--layouts ep0 ep3]

## Region

The object-coverage grid (`lerobot/adhocs/object_coverage`, 6 x 4 cells over the table in the top image) unprojected through
the scene's `ext` camera onto z = 0. Cells rows 2-3 x columns B-E, where the real data put the clippers, case and kit, are
the quadrilateral (0.151, -0.283), (0.183, 0.480), (0.565, 0.422), (0.538, -0.302) in table coordinates (x toward the
camera, y to the robot's right, base at the origin), i.e. x 0.15-0.57 and y -0.30..0.48, intersected with the 0.50 m reach
disc (basket centre <= 0.48 m; bottle <= 0.45 m from the basket). Grid rows 1 and 4 are the strips x < 0.15 and x > 0.57,
column A is y < -0.30, column F y > 0.48 (the real table is wider on the right than the sim box). The five base layouts'
task objects all lie inside the block.

close_lid is centred on the three T071 demo poses measured by the lid study instead (`lerobot/scripts/sim_builders/lid_solvability/
demo_pose_summary.json`: ep0 (0.497, 0.211) yaw 206, ep2 (0.423, 0.257) yaw 213, ep3 (0.467, 0.048) yaw 306; variants 1-3
and 4-6 follow them in that order) with +-4 cm xy and +-20 deg yaw jitter, lid toward the robot side.

## Checks per candidate

- free bodies: yaw uniform; basket: base yaw +-25 deg; z and the non-yaw rotation stay as in the base
- the body's AABB 2 cm inside the sim table box; its convex footprint (hull of the geoms' oriented boxes, lid and strap-loop
  capsules included) 2 cm clear of every other scene body's footprint and of the robot base link
- no contact penetration > 1 mm against the parked arm, the table or a neighbour (rejects e.g. a basket under the park pose)
- reach: damped-least-squares IK over joint1-6 clipped to the follower's soft limits (config_rebot_b601_follower.joint_limits
  intersected with the MJCF ranges), finger tips (= the gripper_end origin) at the target, approach axis straight down,
  5 starts x 200 iterations, accepted at < 5 mm / < 5 deg with no arm-table penetration. Targets: object top for the
  clippers, case and bottle; position-only 5 cm above the basket rim over its centre; both side walls of the open lid
  (the pinch that closes it, lid agent's README) for close_lid
- after writing: 1 s of physics at 30 x 17 Hz with the arm held at park, task objects (and the lid) move < 5 mm and show no
  penetration deeper than 2 mm. Observed: <= 0.2 mm moved, -0.4 mm worst, on every variant

Known, pre-existing, not touched: base ep0 places `bottle_white_lying` through `bottle_yellow_cap_standing` (AABBs overlap
~5 cm); during the settle the lying bottle is ejected 38 cm (checks.json base_settle). The sim's reset does the same 0.5 s
settle, so the eval has always started ep0 that way.

## Files

- variant folders: `scene.xml` (only the task-object body lines and the model name differ from the base), `objects.json`
  (`variant_of`, seed, poses, radius from the base; close_lid also the lid centre, hinge point and axis), `background_top.png`
  copied; materials/meshes stay referenced through the base folder (`<base>/materials/...` relative to assetdir `..`, as the
  base XMLs already do)
- registry: `tasks.py` `layout_name` strips `_v<k>`, `variants()`, `eval_layouts()`, `layouts` prints variants in place of
  their base; `sim_summary.py` rolls `reward.<layout>_v<k>` up into `reward.<layout>`; tests in
  `tests/robots/test_rebot_b601_sim_tasks.py`; section in `scenes/README.md`
