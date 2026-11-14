# Motor torque capture (2026-11-12)

> **Status:** archived 2026-11-13. Superseded: velocity and torque are dataset columns (`observation.velocity`, `observation.effort`), see `design/data_annotation.md`; the sidecar columns below are no longer written. Original status: experimental capture, built 2026-11-12, torque recorded on hardware the same day (run 14-50-14_rebot-corr10-ckpt2000-bitkit-20261112), velocity not yet. Inference only; teleop recording (`lerobot_record`) does not capture torque. Nothing in training, caches or annotation reads it. If the signal proves useful, the integration is a dataset feature `observation.effort` (7 values) and this note moves to `design/`; if not, delete the code below and archive this note.

Question: does the follower's motor torque show contact (grasp, insertion, collision) clearly enough to be worth
giving to the model?

| Piece | Where | What |
|---|---|---|
| Driver | `robots/rebot_b601_follower/rebot_b601_follower.py` | `get_observation()` also returns `{joint}.torque` in N*m and `{joint}.vel` in deg/s (added after the first run, to model friction and inertia), from the same Damiao feedback frame as the position. Not in `observation_features`, so `observation.state` stays 7 joint positions. |
| Env | `rl/gym_manipulator.py` `RobotEnv._get_observation` | forwards the `.vel` and `.torque` keys in the raw observation, like `{cam}.depth` |
| Recorder | `rl/online_recorder.py` | one `{joint}.vel` and one `{joint}.torque` column per motor in `meta/online_labels.parquet` (one row per recorded frame, park frames included) |
| Live view | `rl/rtc_actor_runtime.py` | `observation.{joint}.torque` scalars in Rerun |
| Viewer | `adhocs/torque_viewer/` | torque vs state, action and the videos for one episode |

The value is the motor's own estimate from phase current, before the gearbox: it includes friction, gravity and
acceleration, not only contact. The gripper runs in FORCE_POS with `gripper_torque_ratio` and carries no gravity
load, so it is the cleanest joint.

Limits of the sidecar: `annotation/rebot/build_root.py` writes a fixed set of label columns, so torque stays in the
raw `inference_dataset` root and does not follow frames into annotated roots.
