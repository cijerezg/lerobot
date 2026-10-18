# Sim scenes

MuJoCo layouts for `rebot_b601_sim` (`config.scene = <name>` resolves to `scenes/<name>/scene.xml`).
Built 2026-10-17 from the 2026-10-16 nail-clipper recordings (`migration/real2sim_2026-10-17`, HANDOFF.md there has
the calibration numbers and the rebuild order; the build scripts stay in that folder).

| layout | source | task objects | notes |
|---|---|---|---|
| ep0 | 10-16 ep 5 frame 0 | case at (0.16, 0.34), clippers near it | the nail-clipper episode; `config_rl_sim.yaml` default |
| ep1 | 10-16 ep 1 frame 0 | clippers 0.25-0.33 m from the case | tidy-up recording |
| ep2 | 10-16 ep 2 frame 0 | case + clippers mirrored to +y (robot's right) | the only right-side layout |
| ep3 | 10-16 ep 3 frame 0 | clippers 0.25-0.33 m from the case | tidy-up recording |

Per layout: `scene.xml` (free bodies for the case, the two clippers and every clutter primitive, all within 0.50 m
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


Layout names are `ep0`, `ep1`, `ep2`, and `ep3`. These are simulation layout IDs;
the source recording episode numbers in the table above remain unchanged.

## Approved appearance update (2026-10-18)

All four `scene.xml` files now use the approved clipper, surrounding-object and
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
