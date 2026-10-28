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
| close_lid | ep2's hinged headphone case, base fixed, lid on a real hinge with a tray collider | `headphone_case`, `headphone_lid`, hinge `lid_hinge` (0 = open, closed = the nonzero end of its `range`) | 2026-10-22, easy tier |

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
| close_lid | midpoint within 8 cm of the lid COM 0.1, hinge past 25 % of closed 0.4, past 75 % 0.7, within 8 % of closed with the gripper clear of case and lid 1.0 |

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
