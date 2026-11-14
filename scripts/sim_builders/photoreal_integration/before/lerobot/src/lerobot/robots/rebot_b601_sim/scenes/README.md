# Sim scenes

MuJoCo layouts for `rebot_b601_sim` (`config.scene = <name>` resolves to `scenes/<name>/scene.xml`).
Built 2026-10-17 from the 2026-10-16 nail-clipper recordings (`migration/real2sim_2026-10-17`, HANDOFF.md there has
the calibration numbers and the rebuild order; the build scripts stay in that folder).

| layout | source | task objects | notes |
|---|---|---|---|
| nail_ep5 | 10-16 ep 5 frame 0 | case at (0.16, 0.34), clippers near it | the nail-clipper episode; `config_rl_sim.yaml` default |
| nail_ep1 | 10-16 ep 1 frame 0 | clippers 0.25-0.33 m from the case | tidy-up recording |
| nail_ep2 | 10-16 ep 2 frame 0 | case + clippers mirrored to +y (robot's right) | the only right-side layout |
| nail_ep3 | 10-16 ep 3 frame 0 | clippers 0.25-0.33 m from the case | tidy-up recording |

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
clutter is flat single-colour primitives; no cast shadows.
