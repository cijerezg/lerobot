Nail clipper appearance experiment — 2026-10-18

Watch first
-----------
comparison.mp4       21 seconds, original left / updated right. Camera orbits of
                     the large clipper, small clipper, and case. Native MuJoCo RGB.
camera_replay.mp4    15 seconds, original left / updated right; top camera above
                     wrist camera. Both simulators receive the same real recorded
                     joint targets, at 30 Hz, from 10-16 episode 5, frames 260–709.
comparison.rrd       The same top/wrist replay, with four synchronized image views.
comparison_sheet.jpg  Still previews of the three objects.

Open the Rerun recording from the workspace root:
  .venv/bin/rerun lerobot/scripts/sim_builders/photoreal_clippers/comparison.rrd

What changed
------------
- Both clipper visual boxes are replaced by reference-guided metal plates,
  shaped jaws, pivot shaft and cross pin, a forked lever, and raised grip ribs.
  Their lettering is extracted from the actual sim/nail.mp4 video, with broad
  illumination removed before mapping it onto the lever. There is no generated
  lettering, generic product-photo substitution, or AI image-generation pass.
- The case visual has a clean rounded tin shell, rolled rim, foam insert, and
  recessed slots. Its footprint and slot outlines come from the existing scan
  and material masks. The shell and foam heights are regularized approximations,
  not a new metrologically validated scan. Foam microtexture is procedural.
- Lighting has stronger metal highlights and four low-intensity shadow lights
  for less abrupt shadows. This is the installed MuJoCo 3.3.7 native renderer,
  not a ray-traced rendering layer or a postprocessed movie effect.
- Four independent scene/config variants are provided: nail_ep5, nail_ep1,
  nail_ep2, and nail_ep3. Camera placement, color transforms, and compositing
  follow the existing setup. Existing table/robot/clutter assets are referenced
  read-only from the original scene directory.

Real sources
------------
sim/nail.mp4: 4K orbit video, frames sampled at 13, 18.6, 27, 41, 62, 83 seconds.
The top view supplies the lettering; oblique and low side views guide the plate,
pivot and lever geometry. references/ contains the exact extracted frames/crops.
assets/provenance.json records the texture-source quadrilaterals and homographies.
assets/case_provenance.json records the case geometry assumptions.
The clipper dimensions remain the existing 75 x 12 x 10 mm and 53 x 12 x 9 mm
collision-box dimensions, previously fitted from the real wrist/depth data.
Motion comes from outputs/rebot_2026-10-16-annotated-v1, episode 5 observation.state.

What the comparison proves, and what it does not
----------------------------------------------
This is an appearance comparison and real-joint-target replay, not a policy
rollout or a task-success benchmark. Identical targets are applied through the
existing simulator's send_action path. Images are obtained through its existing
get_observation path, including top-camera color correction and compositing.
The closeup video orbits the viewing camera around stationary settled objects.
The new visual geometry is more detailed than the retained collision geometry;
contact can still occur at the original simplified box/hull surfaces. No claim
is made that this improves policy transfer or fixes existing camera/kinematic
alignment errors. Native MuJoCo highlights still approximate reflective metal;
clutter remains the existing simple geometry/materials.
Slot-floor shading is explicit material tuning so the slots stay legible in
both cameras; it is not recovered physical reflectance. Lighting and camera color
correction remain approximations to the two real cameras' responses.

Validation
----------
validation.json: all four layouts compile; original and updated masses, inertias,
joints, actuator parameters and collision geometry/settings match exactly;
RGB observations are uint8 480x640x3 and wrist depth is uint16 480x640.
original_sha256.json: hashes of 151 original simulator/config/script files,
checked by validate.py to ensure they remain unchanged.
replay_validation.json: maximum original/updated qpos difference across the
replayed steps, and native RGB observation timing (excluding depth and policy).
video_validation.json: output dimensions, duration and frame counts. Both MP4s
are H.264/yuv420p with fast-start metadata and were checked by full decoding.
The Rerun recording was opened with the installed dataframe reader.

Rebuild from workspace root
---------------------------
  .venv/bin/python -B lerobot/scripts/sim_builders/photoreal_clippers/prepare_references.py
  .venv/bin/python -B lerobot/scripts/sim_builders/photoreal_clippers/build.py
  MUJOCO_GL=egl .venv/bin/python -B lerobot/scripts/sim_builders/photoreal_clippers/validate.py
  MUJOCO_GL=egl .venv/bin/python -B lerobot/scripts/sim_builders/photoreal_clippers/render.py

render.py --quick produces a short preview and overwrites the same output files.
All builders write only under this experiment. Original scene assets and real
recordings are dependencies; this folder is not a standalone portable package.
Absolute scene paths are regenerated by build.py if the workspace is relocated.
previews/ contains intermediate diagnostic renders, not the final comparison.

Optional policy evaluation (not executed for this appearance experiment)
----------------------------------------------------------------------
  bash lerobot/scripts/sim_builders/photoreal_clippers/run_eval.sh \
    outputs/<run>/checkpoints/<step>/pretrained_model nail_ep5

The launcher runs one episode and writes under this experiment's policy_runs/.
Pass nail_ep1, nail_ep2 or nail_ep3 to select another layout. It uses the same
inference entry point as the original evaluation script. The launcher was syntax
checked, but a checkpoint policy evaluation has not been run here. Do not pass
these configs to the original run_sim_eval.sh without accounting for that
script's --env.robot.scene override, which selects the original packaged scenes.
