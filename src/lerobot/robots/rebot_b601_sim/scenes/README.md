# Sim scenes

MuJoCo layouts for `rebot_b601_sim` (`config.scene = <name>` resolves to `scenes/<name>/scene.xml`).
Built 2026-10-17 from the 2026-10-16 nail-clipper recordings (`migration/real2sim_2026-10-17`, HANDOFF.md there has
the calibration numbers and the rebuild order; the build scripts stay in that folder).

| layout | source | task objects | notes |
|---|---|---|---|
| clippers_only | ep0 task-object poses | the original case + two clippers | uncluttered default; `config_rl_sim.yaml` selects it |
| ep0 | 10-16 ep 5 frame 0 | case at (0.16, 0.34), clippers near it | the original cluttered nail-clipper episode |
| ep1 | 10-16 ep 1 frame 0 | clippers 0.25-0.33 m from the case | tidy-up recording |
| ep2 | 10-16 ep 2 frame 0 | case + clippers mirrored to +y (robot's right) | the only right-side layout |
| ep3 | 10-16 ep 3 frame 0 | clippers 0.25-0.33 m from the case | tidy-up recording |
| lift_bottle | ep0's standing bottle at clippers_only's case position | one bottle (`bottle`) | 2026-10-22, easy tier |
| bottle_basket | ep0's standing bottle + measured brown basket (`assets/photoreal/basket`, woven mesh from 10-21 wrist frames, hollow collider) | `bottle`, `basket` | 2026-10-22, easy tier |
| close_lid | ep2's hinged headphone case, base fixed, lid on a real hinge with a tray collider (floor + four 7 mm rim walls); the loop handle on the base's far end has capsule colliders (2026-10-28) | `headphone_case`, `headphone_lid`, hinge `lid_hinge` (0 = open, closed = the nonzero end of its `range`) | 2026-10-22, easy tier; solvability study in `migration/lid_solvability_2026-10-28` |

## Eval set, tiers and staged reward (2026-10-22)

`tasks.py` is the registry: scene -> tier, task string, `eval_subtasks` script, scorer, stages. `run_sim_eval.sh` runs
`LAYOUTS` (easy: clippers_only, bottle_basket, close_lid; hard: ep0, ep3; lift_bottle, ep1 and ep2 stay on disk) with
`episodes: 8` per layout (2026-10-28, was 4) and `control_time_s: 60.0` (`terminate_on_success` ends an episode at the
last stage; the real nail episodes run 78-102 s, so 60 s truncates some). Since 2026-10-28 the eval runs on a sim clock
(`env.robot.inference_delay_steps`, rl/rtc_actor_runtime): the env steps in lockstep with the inference worker instead
of at wall-clock fps, the RTC delay is fixed at 6 control steps (the local wall-clock eval's p95 cycle latency, 0.18 s;
the real robot here runs at 0.30 s = 9), and the env executes exactly 6 queued actions per chunk, so
`PARALLEL=N run_sim_eval.sh` can run N layouts on one GPU without
changing the policy's regime. The episode reward is the cumulative value of the furthest stage
reached (sequential walk, success reachable directly, never decreases); `sim/index.json` carries `reward.<scene>`,
`reward.easy`, `reward.hard` (tier mean over scenes of the per-scene episode mean) and `reward.mean` (mean of the tiers).

| scene | tier | task (sheet id) |
|---|---|---|
| ep0, ep3 (ep1, ep2 on disk) | hard | Put the nail clippers in the clipper case (T085) |
| clippers_only | easy | Put the nail clippers in the clipper case (T085) |
| bottle_basket | easy | Put the bottle in the brown basket (no sheet entry; script after T072 steps 5-8 / T048 wording) |
| close_lid | easy | Close the headphone case lid (T071) |
| lift_bottle | easy | Lift the bottle (no sheet entry; script after T082/T089 + a lift step) |

Stages (predicate = value); thresholds in `nail_task.py` / `easy_tasks.py`:

| task | stages |
|---|---|
| nail (T085) | reach large 0.1, large held 0.2, large over the case 0.3, large seated 0.5, small held 0.6, small over the case 0.7, both seated 1.0 |
| bottle_basket | midpoint within 6 cm 0.1, pinched 0.3, pinched and 3 cm up 0.5, 3 cm up over the basket footprint 0.8, inside the footprint below the rim and released 1.0 |
| lift_bottle | midpoint within 6 cm 0.2, pinched 0.5, pinched and 5 cm up 0.8, pinched and 5 cm up for 2 s 1.0 |
| close_lid | finger midpoint within 5 cm of the nearest lid collision geom (rim walls or tray floor; a rim pinch sits 9-10 cm from the lid COM) 0.1, hinge past 25 % of closed 0.4, past 75 % 0.7, within 8 % of closed with the gripper clear of case and lid 1.0 |

Per layout: `scene.xml` (free bodies for the case and two clippers; recorded layouts also include clutter primitives, all within 0.50 m
of the base; table collision plane; cameras `ext` = top webcam, `wrist` = D405 on gripper_end), `materials/`
(one flat measured colour per material as textured meshes, wrist-camera colour space), `background_top.png`
(real top frame with the robot, its shadow and every object footprint filled with table grain: only the
surroundings survive the composite), `objects.json` (the placed bodies).

Shared: `rebot_b601_dm_photo.xml` (the follower's `rebot_b601_dm.xml` with photo-fitted materials and the calibrated
wrist camera; meshes from `../rebot_b601_follower/mjcf/assets`), `table_top.png` (ep 5's photo texture; the other
layouts' own textures were tinted), `assets/clipper_case/` (scan mesh, 32 convex hulls, skirt box).

Paths inside the XMLs are relative to `assetdir=".."` (this directory). Known and accepted: the sim forward
kinematics of recorded joints misses the real object positions by 9-37 mm (the policy runs closed-loop); the
visual meshes remain more detailed than the simplified collision geometry.


Layout names are the folder names above (simulation layout IDs); the source recording episode numbers in the
table remain unchanged.

## Approved appearance update (2026-10-18)

All `scene.xml` files now use the approved clipper, surrounding-object and
connected headphone-case visuals. `run_sim_eval.sh` and `config_rl_sim.yaml` select
these scenes normally; no experiment config or additional flag is needed.

The clippers have shaped metal plates, pivots, jaws and real photographed lettering.
Bottles use real label crops, shaped shoulders and caps; the other clutter includes
fabric cases/pouches, a compartment organizer, a tapered cup and hollow tape roll.
Open headphone cases have two connected halves, a fabric hinge, headphones in one
half and a molded liner in the other. The table texture and calibrated cameras are
unchanged. Lighting and materials use the approved native MuJoCo appearance pass.

`assets/photoreal/{clippers,objects,headphones}/` contains the promoted meshes and
textures. XML references are relative and all required runtime assets are packaged
with LeRobot. The scenes do not depend on `migration/` at runtime.

Visual additions have zero mass and disabled contacts. Some legacy visual geoms
remain hidden in group 5 because they contribute inferred body mass/inertia.
Original collision shapes, joints and actuator settings are preserved. The extra
headphone half and fabric hinge are rigid visual additions to the original body;
there is no simulated closing joint or new collision surface. Cup/tape openings
and the organizer lid are also visual details over simplified original colliders.

Source/reference and builder history remains in the workspace under
`migration/photoreal_clippers_2026-10-17/`,
`migration/photoreal_objects_2026-10-18/`, and
`migration/photoreal_headphone_hinge_2026-10-18/`.
The integration backup, asset manifest and validation reports are under
`migration/photoreal_integration_2026-10-18/`. Earlier experiment hash checks describe
the pre-integration state; their original-scene hashes intentionally differ now.

## Placement variants (2026-10-28)

`<layout>_v1` .. `<layout>_v6` for bottle_basket, clippers_only, close_lid, ep0 and ep3: the same scene with only the task
objects' xy + yaw re-sampled (bottle + basket; case + two clippers, the recorded clutter stays put; the fixed headphone
case with its lid). `tasks.py` resolves a variant name to the base layout's Task (`layout_name` strips `_v<k>`), `variants(layout)`
lists them on disk and `eval_layouts()` / `tasks.py layouts` replace a base layout by its variants, so `run_sim_eval.sh` runs
the 30 variants and not the five bases; `sim_summary.py` keeps `reward.<layout>_v<k>` and rolls them up into `reward.<layout>`
(mean over all the variants' episodes); the tiers average over base layouts.

Region: the object-coverage grid cells rows 2-3 x columns B-E (`lerobot/adhocs/object_coverage`, where the real data put the
clippers, case and kit), unprojected through the ext camera onto the table: the quadrilateral (0.151, -0.283), (0.183, 0.480),
(0.565, 0.422), (0.538, -0.302) in table coordinates (x toward the camera, y to the robot's right), intersected with the 0.50 m
reach disc around the base (basket centre <= 0.48 m, bottle <= 0.45 m from the basket). Yaw is uniform for the free objects, base
+-25 deg for the basket. close_lid instead centres on the three T071 demo poses measured by the lid study
(`migration/lid_solvability_2026-10-28/demo_pose_summary.json`: ep0, ep2, ep3; variants 1-3 and 4-6 follow them in that order)
with +-4 cm xy and +-20 deg yaw jitter around the demo body yaw (lid toward the robot side as in the demos). Every pose keeps the
body's AABB 2 cm inside the table and its convex footprint (lid open and the strap-loop capsules included) 2 cm clear of every
other scene body and of the robot base, and passes an IK reach test inside the follower's soft joint limits (straight-down pinch at
the object top; position only 5 cm over the basket rim; for close_lid a pinch across each side wall of the open lid, the grasp that
closes it); each variant settles for 1 s with the task objects moving < 5 mm.

Regenerate (seed 20261028, per-layout sub-seed = index in bottle_basket, clippers_only, ep0, ep3, close_lid; byte-identical
output): `MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python migration/sim_variants_2026-10-28/generate_variants.py [--layouts ...]`.
Each variant folder holds `scene.xml` (only the task-object body lines differ from the base), `objects.json` (`variant_of`, seed,
sampled poses, the lid's hinge point) and a copy of `background_top.png`; meshes and `materials/` stay referenced through the base
folder. Contact sheets: `migration/sim_variants_2026-10-28/contact_<layout>.png`.
