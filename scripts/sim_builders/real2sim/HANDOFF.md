# real2sim handoff (2026-10-17 evening)

Goal: sim evals that rank close checkpoints, starting with the nail-clipper task. Real robot time is the bottleneck.
User decisions today: open-source models, rigid objects first; nail clippers = priority; headphone asset "good enough";
masses nominal until kitchen-scale values arrive; closed-state case scans still to be recorded.

## What exists (all under lerobot/scripts/sim_builders/real2sim/, moved from migration/real2sim_2026-10-17 on 2026-11-13; `work/` was deleted before the move, the ported layouts live in the package scenes)

| piece | where | state |
|---|---|---|
| object pipeline (video -> VGGT -> table frame -> Letter-sheet scale -> TSDF -> CoACD -> MJCF) | scripts/, README.md | works; scale = known-aspect rectangle search |
| headphone case asset | assets/headphone_case/headphone_case.xml | free base + hinge lid + pull tab (4 boxes), solid shells; closes in test; user accepted |
| clipper case + 2 clippers | assets/clipper_case/clipper_case.xml | rigid tray (32 hulls) + two boxes (75x12x10, 53x12x9 mm; lengths from the wrist depth footprints 2026-10-17 night, were 80 and 60 from the scan cloud) |
| recreated nail scene | work/scene_nail/scene_nail.xml (+ local robot copy rebot_b601_dm_scene_nail.xml) | robot at origin, table 0.74 x 1.01 m top at z=0, clutter primitives, calibrated cameras `ext` (top webcam) and `wrist` (D405 on gripper_end) |
| renderer | scripts/scene_render.py ROOT EP SCENE OUT --frames ... | top RGB, wrist RGB, wrist depth (uint16, 0.1 mm/level, D405 range) + real-vs-sim composites |
| reports | work/scene_nail/REPORT.md, work/nail/REPORT.md, README.md "second pass" | numbers, residuals, limits |

Key numbers: top camera from robot-silhouette fit over 14 poses, mean IoU 0.68, same pose for 10-15 and 10-16 episodes,
472 px/m on the table, height/fovy degenerate (0.85-0.98 m / 61-50 deg). Wrist camera: fx=fy=394.98 (config
`rebot_depth_intrinsics`), fovy 62.57, pose on gripper_end (-0.055, -0.003, 0.051); table-plane residual 1.8 deg;
depth median abs error 1-18 mm on table pixels. Finger silhouettes and table depth disagree by 4 cm along the view
(H1 vs H2 in wrist_camera.json; H2 used) -> MJCF finger geometry / slide mapping vs real gripper is an open question.

Dataset facts: camera keys in the 10-15/10-16 roots are `observation.images.top` and `observation.images.wrist`;
depth sidecar depth/wrist.depth/episode-XXXXXX/frame-XXXXXX.png every 3rd frame; state = 6 joints deg + gripper deg,
to_qpos() in lerobot/src/lerobot/probes/action_trace_sim.py maps it to the 8-slot MJCF qpos.

## Inference in the sim: DONE 2026-10-17 night (robot + teleop + config; policy run not yet started)

| piece | where |
|---|---|
| sim robot `rebot_b601_sim` | lerobot/src/lerobot/robots/rebot_b601_sim/ (config + class), registered in robots/utils.py, imported by rl/gym_manipulator.py |
| camera config `mujoco` | same package: `camera` = MJCF camera name, `use_depth` adds `{cam}.depth` (uint16, 0.1 mm/level, zero outside 0.07-2.5 m) |
| leader stand-in `sim_autostart` | lerobot/src/lerobot/teleoperators/sim_autostart/; START_EPISODE true until the robot's episode cap, no intervention, no feedback |
| embodiment alias | datasets/embodiment.py `rebotb601sim` -> "Rebot B601" (prompt clause unchanged) |
| config | config_rl_sim.yaml (workspace root) = config_rl.yaml with env.robot/teleop swapped, task T085 + its 9 subtasks, control_time_s 120, episodes 10 |
| smoke test | scripts/sim_robot_check.py -> work/scene_nail/sim_robot_check/ (renders at 4 steps) |

How it works: `RobotEnv` sees the same seam as the follower (`get_observation`, `send_action`); one `send_action` = 17 physics
steps of the position actuators (1/30 s, the action_trace_sim executor, gravcomp on the arm via MjSpec before compile);
`park()` (the runtime's episode end) ramps to the park pose at 20 deg/s, then `mj_resetData` puts the objects back and the arm
at qpos0 = park pose; after `episodes` parks the robot raises SIGINT in its own process (the operator's Ctrl-C) and the teleop
stops starting episodes, so the recorder finalizes cleanly. Renderers are per thread (GL contexts are thread-bound; RobotEnv
renders once at setup on the main thread). Episodes start from the park pose like the real rig (10-16 ep 5 frame 0 = zeros).

Smoke test numbers (sim_robot_check.py): observation keys/dtypes identical to the follower's (top/wrist 480x640x3 uint8,
wrist.depth 480x640 uint16 -> wrist_0.depth after the rename); env step 3.8 ms mean (physics 1.3 + 3 renders); scripted
165-tick move: max |command - state| 0.6/1.9/3.4/4.2/1.4/0/0.2 deg; objects at rest 30 s: no drift after a one-time settle
(case drops 5 mm in the first 0.5 s: its hulls sit above the table at the placed z=0.001; clippers 0.3 mm); park reset
exact (0.0 mm); SIGINT path works.

Run (display needed for pynput: subtask console n/b/r and rerun):

    MUJOCO_GL=egl uv run python -m lerobot.rl.inference_async --config_path=config_rl_sim.yaml

Output: outputs/train/<date>/<time>_rebot-sim-nail-20261017/inference_dataset (recorder, every episode, depth PNG every 3rd
frame) + logs. `use_rerun: false` and `episodes` for unattended runs. Not yet done: the first policy run itself.

Open after the first run: the subtask script is operator-driven (console keys) exactly as on the real rig, so an unattended
sim run stays on step 1 unless advanced -> scoring + automatic subtask advance from body poses is the next piece; start-pose
sampling from real episodes; `pos_vel_velocity: [40 x6, 180]` slews the target like the follower's POS_VEL mode (off by
default, un-identified); case start z (settles 5 mm); finger geometry; photoreal background; remaining objects; closed-state
scans.

## Photorealism pass: DONE 2026-10-17 night (config_rl_sim.yaml now uses it)

| piece | where |
|---|---|
| top camera | SIMPLER-style composite: real robot-free background + sim pixels of every non-world body (robot, free objects), 2 px feathered mask. `mujoco` camera config `background:`; built by scripts/scene_background.py -> work/scene_nail/photoreal/background_top.png |
| background | per-pixel median over the 234 cached frames of 10-16 ep 5 in which the sim robot mask (posed from the recorded state, dilated 45 px for the 4 cm finger gap + shadow) does not cover the pixel; pixels never free (16%, base + around the case) take the first frame (a narrow-mask median there kept a gripper ghost, the "black hook" the user saw); free-body start footprints inpainted |
| wrist camera | textured scene work/scene_nail/scene_nail_photo.xml from scripts/photo_scene.py: table = smooth shade of the background rectified onto z=0 through the calibrated top camera, times a sharp real grain patch mirror-tiled (scripts/table_detail.py: wrist frame 2220 of 10-16 ep 5 rectified through the sim wrist camera, lit table only, 75 x 52 mm at 2 px/mm; dark pixels = fingers excluded; the bigger shadowed patch (--no_lit_only) tiles the soft shadow edge into lines, the small one repeats as a mild ripple), flat lighting, no cast shadows (user 2026-10-17: the hard arm shadow read as a black patch), texture gain calibrated on the top view. `--grain` = the old top-camera grain with strip fills, no `--detail` = smooth shade only |
| clipper case | scan vertex colours baked into a per-face 2x2-texel texture with per-face UVs (scripts/case_texture.py -> assets/clipper_case/clipper_case_visual_textured.obj + _colors.png) |
| robot materials | alum/lime/black rgba scaled by real/sim mean colour inside the eroded robot mask over the 8 watched frames, under the photo lighting (scripts/robot_materials.py -> work/scene_nail/rebot_b601_dm_scene_nail_photo.xml) |
| wrist colour | per-channel gain real/sim on lit table pixels (scripts/wrist_color_fit.py), `color_gain` on the wrist camera config (affine fit degenerate: lit sim table is uniform) |
| check | scripts/photoreal_check.py CONFIG_YAML FRAMES OUT: real / sim observation / blend at the 8 watched frames, built from the yaml's env.robot exactly as the policy sees it -> photoreal/top_final.jpg, wrist_final.jpg |

Numbers (10-16 ep 5, 8 frames, mean |real - sim| over the whole image, 0-255): top 8-12 (was 9-14 raw robot, background
composite), wrist 23-48 (was ~60+ flat colours). Env step with the composite 13 ms. Rebuild order: scene_background.py ->
photo_scene.py (no --robot_xml) -> robot_materials.py on the photo scene -> photo_scene.py --robot_xml -> wrist_color_fit.py
-> photoreal_check.py; table_detail.py once for the grain patch (photo_scene.py --detail). Superseded by the object-texture section below for the wrist camera's colour gain.

Left: no shadows at all now vs the real soft arm shadow, the grain repeats every 7.5 x 5 cm (mirror tiles; a seamless synthesis or more lit frames would fix it), sim fingers (black wedges) vs real
rounded fingers, clutter still flat primitives, residual strip seams in filled texture regions, case foam slightly brighter
than real. The non-photo scene (scene_nail.xml) stays as the plain variant.

## Object photo textures (wrist view): DONE 2026-10-17 late night, revised after the user's check (config_rl_sim.yaml uses them)

The top view mixes the real image in (background composite); the wrist view had flat colours on the case, the clippers and the
clutter. Now every object carries real pixels. First version (per-face texel blocks, wrist colour space) was WORSE in the top view:
the far top camera mip-filters the atlas into speckle, and wrist-space colours came out dark once the camera gain was back. Fixed.

| piece | where |
|---|---|
| case + clippers baker | scripts/object_photo_texture.py ROOT EP SCENE OUT --objects body:lo:hi ... --more_episodes ... --top_frame TOP0.jpg -> work/scene_nail/photoreal/objects/<body>_photo.obj/.png, _frames.json, _align.jpg, objects_bake.log |
| per frame | real table plane from the depth sidecar (RANSAC outside the object's sim mask); real object = raised 4-40 mm component (clippers: also low-saturation metal, box-sized footprint; case: must hold white foam, else the black headphone case passes); alignment in the camera frame, so the 4 cm wrist-calibration bias and the start-pose error do not matter: case = centroid shift + point-to-plane ICP on the scan mesh (rmse 1.6-2.4 mm, IoU >= 0.6; other episodes: whole image, four-yaw search, IoU >= 0.7), clipper = PCA of the depth points on the plane, box top at the box height, long-axis sign from the sim |
| case atlas | PLANAR: faces with nz > 0.3 share one top-down region (texel = body x, y at 0.33 mm, 3D point = first mesh hit from above; open3d RaycastingScene), wall faces in 8x8 blocks below. Planar = coherent under mip filtering. Colours: low frequencies (sigma 1.5 mm) from the real top frame 0 projected through the calibrated top camera (the objects were placed from that image, so the alignment is exact) x detail ratio of the wrist bake; wrist-unseen texels = top colour alone; walls = wrist bake, unseen = scan colours |
| clipper atlas | per-face blocks (K=160) on a trimesh box, top face from the wrist bake (median of the 5 best views, lit weight, frames with mean > 200 skipped), sides = top mean |
| colour space | everything in TOP-camera colours: wrist-baked colours / to_top_gain (0.766 0.655 0.558 RGB), top-frame colours as they are; config_rl_sim.yaml keeps color_gain on the wrist camera as before (the --gain folding of the first version is reverted; the option stays in photo_scene.py, unused) |
| lighting | texels / min(0.6 + 0.65 nz + 0.25 n.l2, 1): the renderer clamps the lit white material at 1 before the texture multiplies it (grey-texture probe: 1.00 on top faces, 0.65-0.86 on walls); object materials specular 0 |
| clutter | scripts/clutter_texture.py SCENE TOP0.jpg OUT: each static primitive (3 bottles, pouch, headphone case, bits box, mic knob) becomes a mesh in world coordinates with a planar atlas textured from the real top frame 0 (1 mm texels, 2 mm source pixels; the wrist sees them only from afar); vertical sides repeat the silhouette edge (no data) |
| scene | photo_scene.py --object_tex DIR: case visual -> photo mesh; clipper bodies get a textured box mesh (collision box -> group 3); every <name>_photo.obj of a static geom hides the primitive (group 3) and adds the world-frame mesh |
| checks | photoreal_check.py (top_objects.jpg, wrist_objects.jpg), scripts/object_color_check.py SCENE FRAMES ROOT EP --gain (colour statistics inside the real vs sim object masks), review sheets photoreal/wrist_objects_before_after.jpg (real / old / new), top_case_crops_before_after.jpg |

Numbers (10-16 ep 5): case frames accepted 80 (ep5 28, ep4 27, ep0 10, ep1 7, ep2 7), wrist-seen texels 41% (the rest: top colour in the
planar region, scan colours on the far walls); large clipper 21 frames, small 42. Top view: the case matches the real one (white foam, dark
end, two slots) where version 1 was speckle; the clippers are grey streaks (real: bright metal). Whole-image errors unchanged (top 6-12,
wrist 27-49: set by the scene misalignment). Object colour histogram overlap real/sim (wrist, with the gain): case 0.49 -> 0.48, large
clipper 0.14 -> 0.40, small 0.15 -> 0.47; the case's sim spread is 34-45 vs real 55-61 (the real includes the arm's shadow).

Also found and fixed: the clipper boxes were too long. Depth footprints over 11 and 14 frames: large 73-75 x 12-13 mm (box was 80 x 11),
small 48-53 x 11-13 (box was 60 x 10). Boxes now 75 x 12 x 10 and 53 x 12 x 9 in work/scene_nail/scene_nail.xml and
assets/clipper_case/clipper_case.xml. The real raised footprint of the case (100 x 150 mm) is larger than the 88 x 119 mm scan mesh:
the scan may have lost the rim; the ICP IoU of 0.7 (not 0.9) agrees. Previous scene and table texture: work/scene_nail/backup_pre_objects/.

Rebuild order now: scene_background.py -> clutter_texture.py + object_photo_texture.py (both on scene_nail.xml, same OUT dir) ->
photo_scene.py --robot_xml --detail --object_tex -> photoreal_check.py + object_color_check.py --gain (table_detail.py, robot_materials.py,
wrist_color_fit.py as before).

Left: brown streaks on the case rim's top edge (wood pulled in at the silhouette by the 2 mm top pixels and 1-2 mm alignment scatter;
a normalised low-pass inside the footprint would remove most of it), far walls and slot floors partly scan colours, clipper sides and
clutter sides are edge colours, no arm shadow.

## Next: inference in the sim (plan as written before the implementation above)

Seam: lerobot/src/lerobot/rl/gym_manipulator.py `make_robot_env` -> `make_robot_from_config(cfg.robot)`; `RobotEnv`
only calls `robot.get_observation()` / `robot.send_action()`. Plan = a `rebot_b601_sim` robot type that owns the MuJoCo
scene, steps position actuators like action_trace_sim.execute, renders `ext` as "top" and `wrist` as "wrist" (+ depth
in the same key/dtype the RealSense path emits), and returns the follower's observation dict unchanged. Config: a copy
of config_rl.yaml at the workspace root with env.robot swapped, task "Put the nail clippers in the clipper case",
eval_subtasks from docs/annotation_notes/rebot_tasks.md T085, no leader teleop. Policy call path must stay the
deployed one (rtc_actor_runtime). Entry: `uv run python -m lerobot.rl.inference_async --config_path=<yaml>`.

Open items after that: scoring (clipper in slot from body poses), start-pose sampling from real episodes, controller
gain identification (sim lag is half the real lag), finger geometry check, photoreal background (SIMPLER-style
composite or splats), remaining objects (bits box, bottle, basket), closed-state scans for the hinges.

## Object appearance redone from first principles: DONE 2026-10-17 night (config_rl_sim.yaml uses it; supersedes "Object photo textures")

The projection bake (object_photo_texture.py, clutter_texture.py) is replaced by one flat colour per material with boundaries from
the geometry. Nothing is projected or blended across a silhouette any more. Review: photoreal/wrist_materials.jpg,
top_materials.jpg, top_case_crops_materials.jpg (real | composite, 2.5x), materials/case_atlas_review.png (class map | texture).

| piece | where |
|---|---|
| colour measurements | scripts/measure_materials.py -> photoreal/materials/measured.json + samples.jpg (hand-picked windows well inside each region: top frame 0, wrist f334 / f1002 / f668 / f2005, BGR medians, clutter = real top pixels inside eroded sim silhouettes) |
| builder | scripts/object_materials.py SCENE OUT -> OUT/<body>_photo.obj/.png (same contract as before, photo_scene.py --object_tex), materials.json |
| case | planar atlas 0.5 mm in the body frame; class per texel: rim = within 3.5 mm of the footprint edge, slot = mesh height < 13 mm (1 mm open/close), black end = rows whose scan brightness is dark (modes 94/227, boundary row 82/240, +y end), foam = rest; walls (nz < 0.3) and the skirt map to a constant rim patch. Two visual meshes: clipper_case_photo (foam/slot/black, matte) + clipper_case_metal_photo (rim/walls/skirt, specular 0.3) |
| case geometry | mesh shifted +2.5 mm (real foam 17-19 mm and rim 20-21 mm above the table in the depth sidecar at f334/f1002/f1671; mesh foam 15-16.5, rim 17-18.5), closed below by a skirt (footprint outline extruded 0..7 mm) in the visual; collision: 32 hulls get pos 0 0 0.0025 + a skirt BOX 87 x 118 x 7 mm (a flat convex skirt mesh vs the table box settled 6 mm; the box rests at -0.2 mm). Body z 0.001 -> 0.0. Applied to work/scene_nail/scene_nail.xml and assets/clipper_case/clipper_case.xml; backup work/scene_nail/backup_pre_materials/ |
| clippers | textured box: body colour, dark head over the last 25 % of the length at the end the real top frame shows dark (large +u, small +v; the body x axis sign is read through the ext camera: both +x), sides body colour, specular 0.3 |
| clutter | primitive meshes with a uniform 4x4 texture (real top colour through the inverse top affine) |
| colour space | textures in WRIST (D405) colours. Top camera: `color_gain [1.222, 1.278, 1.165]`, `color_offset [8.0, 8.5, 28.1]` (RGB; per-channel affine through foam and black end measured in both cameras), applied to the render (robot + free objects) before the composite. Wrist camera: no correction; the wood gain is folded into the table texture (photo_scene.py --gain 0.758 0.603 0.452, refitted on fully lit wood). Robot materials refitted through the top affine (robot_materials.py --affine) |
| top background | scene_background.py --seam_sigma 30 (default): the frame-0 fill of the never-free pixels (parked arm's shadow toward the case) is lifted by the median / frame-0 ratio of the seam band, extrapolated inward (median x1.05, max x1.36) -> photoreal/background_top_v2.png. Tried and dropped: per-pixel median over the K frames with the farthest sim silhouette (gripper ghost over the case: the sim silhouette lags the real gripper; arm-root ghost at the base), polynomial lit-model flattening (lifted 20 % of the image) |

Why one linear colour map could not serve both cameras (measured, 10-16 ep 5): the top webcam neutralises and compresses the wood
(wood 209/210/223, foam 253/250/250 BGR) while the D405 keeps it brown and spread (wood 60/97/150, foam 193/189/198), so the
foam/wood ratio is 1.2 in one camera and 1.3-3.2 per channel in the other. The wrist is the close-up camera, so it gets the exact
colours; the top gets the affine, exact for the foam and the black end, 40 levels too bright for the clipper bodies (a brushed
steel bar reflects the ceiling in the wrist camera, 138-157, and reads 120 from the top camera: view-dependent, compromise at
120/126/135 wrist).

Numbers (photoreal_check.py, 8 frames): whole-image errors unchanged (top 8-12, wrist 25-49: scene misalignment, not a texture
measure). Per material, median of the rendered observation inside the sim segmentation: wrist foam 193/189/198 (real 193/189/198),
black end 54/63/63 (real 54/63/63), rim 163/168/188 (real 169/169/194), table 81/109/155 before the gain refit (real lit 60-79 /
90-112 / 143-168); top foam 252/250/249 (real 253/250/250), black end 91/89/84 (real 91/89/85), clippers 157-170 (real 120).
Object colour histogram overlap (object_color_check, wrist, no gain): case 0.33, large clipper 0.14, small 0.22 (the real pools
include the arm's shadow and the metal highlights).

Rebuild order now: scene_background.py (-> background_top_v2.png) -> measure_materials.py (only if windows change) ->
object_materials.py (plain scene_nail.xml) -> photo_scene.py --detail --object_tex materials --gain 0.758 0.603 0.452 (no
--robot_xml) -> robot_materials.py --affine 1.222 1.278 1.165 8.0 8.5 28.1 -> photo_scene.py ... --robot_xml -> photoreal_check.py
--tag _materials. photoreal/objects/ (old bake) is unused.

Left: the parked arm's shadow is reduced, not gone (a soft patch below the parked gripper in the composite whenever the arm is
away); no arm shadow in either view; clipper brightness in the top view (above); slot outlines follow the scan (ragged notch);
the case footprint stays the 88 x 119 mm scan (top frame says 89 x 125 mm); fingers unchanged (black wedges, 4 cm short).
Viewer note: the chat image viewer normalises contrast locally, so colours were judged from numbers only (the D405 frames are
much darker than they look).

## Sim run outputs, auto subtasks, and the kinematic miss (2026-10-17 late night)

| piece | where |
|---|---|
| saving | `rerun_save: true` (config field, train.py): rr.init(spawn=False) + rr.save({output_dir}/rerun.rrd), works headless, replay with `rerun <file>`; `record_inference_dataset: false` skips the OnlineEpisodeRecorder (it was unconditional: every run wrote inference_dataset videos/parquet/depth). Both set in config_rl_sim.yaml; config_rl.yaml keeps the old behaviour (viewer, recorder) |
| auto subtasks | lerobot/src/lerobot/robots/rebot_b601_sim/nail_task.py `NailTaskScorer`: one predicate per T085 step from body poses (held = closed gripper, clipper within 35 mm of the finger midpoint and lifted > 15 mm; near = held and within 60 mm of its slot centre in the case frame below 80 mm; released = open, inside the slot footprint below the rim; seated = inside the footprint at floor + half height + 3 mm), forward only, drop rule (not held, not in the slot during move/release -> back to the grasp step), success = both seated -> `nail_task.SUCCESS` event read by sim_autostart as TeleopEvents.SUCCESS (terminate_on_success ends the episode). Slot footprints from the scan height map (centres (-15.6, -3.9) / (16.0, -4.9) mm, halves 9.5 x 37 / 8.5 x 25 mm, floors 10.1 / 10.8 mm after the shift) |
| wiring | config `env.robot.auto_subtasks: true` (sim robot builds the scorer at connect, resets it with the scene); rtc_actor_runtime polls it after every step and syncs `shared_state.advance_subtask`; events (t, cursor, why) per episode -> {output_dir}/subtask_events.json with the success flag |
| checks | predicate walk with the arm posed from real frames (609 grasp, 839 push, 2097 small grasp) and clippers teleported: all 8 steps, the drop rule and success fire in order, reset clears; replaying ep 5's joint states through the sim never advances (below) |

Physics is real: every gripper, link and object geom has contype/conaffinity 1, the clippers and the case are free bodies with
masses, the finger tips reach the table (z -0.3 mm at the real grasp frame). The gripper moves nothing because the sim forward
kinematics of the recorded joints lands the fingers away from where the real fingers were (objects placed from the top frame):

| real event (10-16 ep 5) | sim finger midpoint vs object, xy |
|---|---|
| grasp large clipper f609 | 37 mm |
| close near the slots f839 | 9 mm |
| open over the slots f1069 | 30 mm |
| grasp small clipper f2097 | 30 mm |

Not one rigid offset (mean 12/12 mm, spread 17/16 mm): the arm model or the top-camera placement is off per pose. Open: identify
joint zero offsets + base pose from the grasp events of several episodes (the fingers were on the object at each real grasp), or
accept per-object shifts. Until then the policy cannot grasp in the sim even when it reproduces the real motion.

## Movable clutter, lazy composite, 4 layouts (2026-10-17 late night)

| piece | where |
|---|---|
| clutter = free bodies | scene_from_episode.py: every clutter primitive is a `<body><freejoint/>` with a nominal mass (CLUTTER_MASS_KG, kitchen-scale values pending) and friction; per-object kind/diameter_m/height_m/mass_kg/rgba may come from the pixels json, so new objects need no code. Table collision moved to a plane (`table_contact`, group 3; the box keeps the look): lying cylinders on the box sank 7-9 mm, on the plane every free body settles <= 0.4 mm in 2 s |
| composite | camera config `render_always: [clipper_case, clipper_large, clipper_small]`: those and the robot are always rendered, every other free body keeps the real background pixels until it moves (> 2 mm or 1 deg from its start pose; `RebotB601Sim._composite`). scene_background.py `--inpaint` lists the bodies whose start footprint is inpainted (the same three) |
| segmentation artefact | silhouette edges of the free bodies come back labelled with robot geom ids (thin stray lines, 8000 px per frame): a 3x3 opening on the masks removes them (composite, scene_background.py, photo_scene.py). Without it the never-free region was 34 % and the seam lift blotched the background |
| layouts | one per real episode of 10-16: ep 5 = work/scene_nail (config_rl_sim.yaml), ep 1/2/3 = work/layouts/ep{N} (config_rl_sim_ep{N}.yaml); pixels read by eye on a 20 px grid: work/layouts/pixels_r1016_ep{N}_f0.json (table corners from ep 5, fixed camera). ep 0 (too many unknown objects) and ep 6 (robot mid-task at frame 0) skipped. New objects in ep 1-3: paper_cup, tape_roll, headphone_half_a/b (open case), bits_kit_open, more bottles, all as primitives |
| build | scripts/build_layout.sh EP OUT [PIXELS] [FRAMES]: frames -> scene -> cached frames (scene_masks.py) -> background (seam fix, --inpaint) -> materials (object_materials.py --top_frame: clutter colours from the eroded sim silhouettes) -> photo scene (table texture next to the scene, robot photo include copied from ep 5) |
| run | scripts/run_layouts.sh [yamls]: the four yamls one after the other (3 episodes each) |

Open: the cup task (a cup-centred script needs its own eval_subtasks and scorer predicates; the cup is a primitive in ep 1 and 3 for now);
the 1-4 cm kinematic miss stands (see above); object colours for the new clutter are flat top-frame medians.

Additions (same night, after the first layout builds):
- scene_background.py `--static_until auto` (default): the median uses only the cached frames before the first frame whose table
  pixels differ from frame 0 by > 10 levels (ep 1/2/3 are tidy-up recordings: a whole-episode median showed the headphone halves
  where the real robot left them; ep 1 static until cached frame 28/702, ep 2 68/810, ep 3 78/1081, ep 5 219/234).
- `_reset_scene` steps 0.5 s before taking the start poses the lazy composite compares against (neighbours placed from the
  image overlap by 1-2 mm and push apart; threshold 3 mm / 1 deg).
- Case orientation convention: body +y = black end (object_materials.py class map); scene_from_episode.py puts body +y on the
  pixel axis, which is read from the foam end to the black end (pixels json `axis`, note field). Checked on ep 5: the rendered
  black end lands on the real one (top band 76 vs real 83, bottom band 185 vs 246).

## Rerun replay fixed (2026-10-17, 20:45)

Saved sim recordings were unreplayable: images were logged `static=True` (live-viewer convention, a saved file keeps only the
last frame) and the only timeline was the integer `step` sequence (Rerun plays those at its own rate, ~8x). Now
`log_rerun_data(..., static_images=False)` from the RTC rerun worker, timeline `time` in seconds = step / env.fps (plays at 1x;
speed field in the time panel for 2x/4x), JPEG for colour images, float -> uint8 before compression. A 30 s episode is ~215 MB
(depth still raw). Old recordings before 20:43 are frozen-image files; the 20:31-20:35 light-run folders were deleted.
Quick run: `MUJOCO_GL=egl uv run python -m lerobot.rl.inference_async --config_path=config_rl_sim.yaml`, then
`uv run rerun "$(ls -dt outputs/train/2026-10-17/*rebot-sim-nail* | head -1)/rerun.rrd"`.

## Top background v3: parked-arm shadow and gripper ghost removed (2026-10-17 night)

User 2026-10-17: an arm-and-gripper shaped shadow sat on the table in every sim frame. Cause: not a rendered shadow (the
photo scene casts none) but the background image. The 45 px robot mask never frees a strip around the base (link1/link2 are
always within 45 px of it), so those pixels took frame 0: the real parked arm, its gripper and its soft shadow, which the
composite only hides while the sim arm is parked. The v2 seam-band lift reduced the shadow but kept the shape.

scripts/scene_background.py rewritten (same CLI, writes background_top_v3.png; configs, build_layout.sh updated; v2 kept on disk):

| step | what |
|---|---|
| median | unchanged: static-layout frames, 45 px mask |
| refill | table pixels free in >= 50 % of those frames under a per-body mask (base + link1..3 20 px, wrist + gripper 45 px; frames whose 15 px window is mostly non-wood rejected): the frame at the 70th percentile of the 15 px box-blurred brightness, pixel copied from it. Lit frames beat the arm's soft shadow, the cable and the fingers (dark minorities); whole-frame picks give no speckle |
| grain fill | the rest (strip around the base, parked-arm silhouette, the cone the gripper hovers over, ~20 k px on ep 5): each row takes real grain mirrored from the nearest clean wood run of that row, scaled to a smooth lit shade (normalised Gaussian, sigma 50 px, over wood-coloured lit pixels; smaller sigmas leave zero-weight holes) |
| kept | sim base silhouette + 25 px beside it down to the base's bottom row (the real base, lime bar and mount are wider than the sim mesh; the sim base is always rendered), clutter footprints + 15 px (a brightness quantile fringes dark objects) |
| frame-0 pose | silhouettes posed from states[0] (qpos 0 is not the park pose); the base silhouette rendered with the arm hidden (geomgroup), a robot-hidden table render z-fights so the table under the arm comes from the convex hull |

Tried and dropped: polynomial lit model (highlight misfit), per-pixel brightness quantile (speckle), a per-frame wood-colour
gate on all pixels (rejects lit low-saturation wood at the lamp highlight, keeps shadowed wood: biased dark), donor episodes
(ep 1-3 cover 6 % of the strip and differ in exposure by x1.0-1.4 per channel), TELEA for the holes (smears the base colours).

Check: photoreal_check.py --tag _v3 -> photoreal/top_v3.jpg per layout (ep 5 top error 8.0-10.2, unchanged). Left: a faint
rectangular boundary of the grain-filled region around the case (mostly under the rendered case and clippers); the lamp
highlight inside that region is the shade's extrapolation.

## Every free body rendered, clutter-free background (2026-10-17 night, after the first v3 run)

User 2026-10-17 (first v3 episode): the policy knocked the pill bottle over, the sim rendered it lying down while the real
upright bottle stayed in the background, "I need shit to physically make sense". So the lazy composite is off: `render_always`
removed from the four yamls (the robot config renders every free body when the list is empty), and scene_background.py builds
the background with every free-body start footprint (+30 px) grain-filled (no `--inpaint`, build_layout.sh updated). Flat
single-colour clutter in the top view now (top error 13-16 levels vs 8-10 with the real pixels, expected). Grain fill = the
largest clean rectangle of lit wood as the ratio to its 8 px low-pass, mirror-tiled over the image on the smooth shade (the
row-mirror fill streaked copied shadows, a grain-only row fill went flat once most of the table was a hole). Left: a real
object wider than its primitive + 30 px leaves a sliver (the headphone case corner in ep 5); inpainting object pixels past the
table hull smeared the table edges and was dropped. The tiled grain repeats visibly in ep 2 (37 x 43 px patch).

## Sim table rendered in the top view (2026-10-17 night)

User: the patched table and the headphone sliver. The table is a sim geom with the photo texture already, so the top camera
renders it too: camera config `render_static: [table, table_top]` (new field; `RebotB601Sim._composite` masks dynamic bodies
plus these geoms) and `render_static_gain: [1.054, 1.231, 1.403]` (RGB, applied to those pixels after the top affine: the
texture is in wrist colours, measured as real / rendered table median on wood pixels of 10-16 ep 5, 4 frames). Only the
surroundings (floor, basket, base, speakers, leader arm) come from the real background now; the table part of
background_top_v3.png is irrelevant. Top error 16-19 levels (flat clutter + one texture). scene_background.py additionally
inpaints dark (V < 60) object pixels past the table hull (the black headphone case beyond its primitive, grey smudge now).

## Objects pulled within reach, shared table texture (2026-10-17 night)

User: far objects should be within reach, with leeway for the bent top-pinch pose, estimates fine. The real nail episode's
fingers never went past 0.41 m from the base (tidy-up episodes 0.60-0.66 m). scripts/pull_objects_in.py LAYOUT... pulls
every free body beyond --limit (0.50 m, xy from the base: the arm reached 0.66 m, 0.15 m margin per user) radially in to it, slides crowded ones along their arc away from
neighbours (bounding circles from geom_rbound + 15 mm), keeps y in [-0.44, 0.49] and x >= 0.10, and edits the body pos in
scene_nail.xml + scene_nail_photo.xml (backups *.pre_pull.xml) and objects.json. Applied to the 4 layouts (24 bodies moved;
build_layout.sh regenerates the original positions, rerun the script after it). The table texture of ep 1-3 was tinted and
uneven (derived from their sparse backgrounds), so they use ep 5's table_top.png now (their own kept as table_top.own.png;
photo_scene.py would overwrite it). Review: work/layouts/layouts_adjusted.png (4 top composites at the park pose).
Removed per user (scripts/remove_bodies.py LAYOUT body...; backups *.pre_remove.xml): ep 1 headphone_half_b, blue_pouch;
ep 2 bits_kit_open, bottle_yellow_lying, headphone_half_b; ep 3 headphone_half_a, blue_pouch (corner/edge or stacked after
the pull-in). The *.pre_pull.xml backups still contain them.
Further user edits (scripts/set_body_pos.py LAYOUT body x y ...): ep 5 headphone case in to (0.16, 0.34); ep 3 bits_box
removed (was under the headphone half); ep 2 case + clippers mirrored to +y (the robot's right, the only such layout);
ep 1 and ep 3 clippers moved 0.25-0.33 m away from the case (ep 1: small (0.17, -0.05), large (0.20, -0.40); ep 3: small
(0.30, 0.0), large (0.49, -0.08)). The case quaternions are unchanged.

## Ported to lerobot (2026-10-17 night, the runtime no longer reads this folder)

The four layouts live in lerobot/src/lerobot/robots/rebot_b601_sim/scenes/ (nail_ep5 = work/scene_nail, nail_ep1..3 =
work/layouts/ep1..3; 19 MB: scene.xml with paths relative to assetdir="..", materials/, background_top.png (= v3),
objects.json; shared rebot_b601_dm_photo.xml, table_top.png, assets/clipper_case/). `config.scene` takes a layout name;
a camera `background` resolves against the scene dir. config_rl_sim.yaml is regenerated from config_rl.yaml by
scripts/make_sim_config.py (sim env + T085 + run flags; episodes 3, control_time_s 120, episodes 2 per user 10-17); config_rl_sim_ep1..3.yaml are
gone, the layout is `--env.robot.scene=...`. lerobot/scripts/run_sim_eval.sh CKPT [OUT] [CONFIG] runs every layout
into validation/step_<step>/sim_nail/<layout>/ + summary.md (scripts/probes/sim_summary.py);
remote_validate.sh --sim launches it on the Spark, chase_validate.sh <run> -- --sim chases with it. This folder keeps
the build pipeline (scripts/, work/, assets/) for rebuilding a layout; copy the result into the package afterwards
(scripts/port_scenes.py copies all four and relativises the paths).
