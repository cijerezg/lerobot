Approved visuals integrated into LeRobot — 2026-10-18

The four packaged nail scenes now use the accepted clippers, textured clutter,
and connected headphone cases. The normal command needs no new flags:

  lerobot/scripts/run_sim_eval.sh outputs/<run>/checkpoints/<step>/pretrained_model

Changes: four scene.xml files, 287 copied meshes/textures under
scenes/assets/photoreal/, package-data patterns in lerobot/pyproject.toml,
scene documentation, and comment-only updates in config_rl_sim.yaml.
All XML runtime paths are relative; there is no dependency on migration/.
The table texture, other existing source assets and robot code were not changed.

Validation (validation.json)
- All four normal named scenes load using config_rl_sim.yaml.
- Exact original mass, inertia, joints, actuators and collision geometry preserved.
- Compiled visual arrays equal the accepted experiment (mesh, texture, materials).
- RGB observations 480x640x3 uint8; depth 480x640 uint16.
- Maximum RGB difference across separately rendered matching views: 3/255;
  depth matches exactly. See per-camera differences in the report.
- 30 matching control steps per layout produce identical joint positions.
- All 531 required runtime files are covered by setuptools package-data.
- All four scenes compile from a relocated copy containing only those package
  assets, with no migration/ files or original-workspace absolute paths.
- run_sim_eval.sh passes bash syntax checking. No checkpoint policy rollout run.

before/ contains the original four scene XMLs, README, pyproject.toml and config.
promotion_manifest.json maps each promoted asset to its experiment source/hash.
The old collision/mass-bearing visual geometry is retained hidden where needed;
extra case halves, hinge, and other detailed features remain visual-only.
The earlier experiments remain on disk. Their stored original-file hash checks
intentionally predate this authorized replacement of the packaged scenes.

Re-run integration validation:
  MUJOCO_GL=egl .venv/bin/python -B lerobot/scripts/sim_builders/photoreal_integration/validate.py
