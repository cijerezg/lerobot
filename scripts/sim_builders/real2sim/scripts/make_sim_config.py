"""config_rl_sim.yaml = config_rl.yaml with the sim env, the nail task and the sim run flags (text patch, comments kept)."""
from pathlib import Path

WS = Path("/home/user/Documents/Research/RL/LeRobot")
src = (WS / "config_rl.yaml").read_text()


def swap(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)


HEADER = """# Sim eval copy of config_rl.yaml (2026-10-17): env.robot = MuJoCo rebot_b601_sim on a recreated nail-clipper layout
# (lerobot/src/lerobot/robots/rebot_b601_sim/scenes), no leader (sim_autostart), task T085. Everything else is
# config_rl.yaml; keep the two in sync (regenerate: lerobot/scripts/sim_builders/real2sim/scripts/make_sim_config.py).
# 2026-10-22: run_sim_eval.sh overrides policy.task / eval_subtasks / precisions / contacts per layout from the
# registry robots/rebot_b601_sim/tasks.py (tiers easy/hard, staged reward); the values below stay the nail task.
# One layout by hand:  MUJOCO_GL=egl uv run python -m lerobot.rl.inference_async --config_path=config_rl_sim.yaml [--env.robot.scene=ep1]
# Every layout on a checkpoint (what the chase runs on the Spark with remote_validate.sh --sim):
#   lerobot/scripts/run_sim_eval.sh outputs/<run>/checkpoints/<step>/pretrained_model
"""

SUBTASKS_OLD_START = "  # 2026-10-17 sequence under the \"Tidy up the table\" umbrella;"
SUBTASKS_OLD_END = "  eval_subtask_contacts: [top-pinch, na, insert, top-pinch, na, insert, rim-pinch, rim-pinch, top-pinch, na, drop, rim-pinch, na, drop, side-pinch, na, drop, top-pinch, na, drop]\n"
SUBTASKS_NEW = """  # T085 (internal_oct07 ep 0): precision/contact per step from the task sheet. The sim scorer
  # (robots/rebot_b601_sim/nail_task.py) walks these steps from the body poses; other layouts get their
  # script from tasks.py through run_sim_eval.sh.
  eval_subtasks:
  - grasp the large nail clipper
  - move the large nail clipper to the clipper case
  - release the large nail clipper in the large slot
  - push the large nail clipper into the large slot
  - grasp the nail clipper
  - move the nail clipper to the clipper case
  - release the nail clipper in the small slot
  - push the nail clipper into the small slot
  - return to home
  eval_subtask_precisions: [4, 4, 5, 3, 4, 4, 5, 3, 1]
  eval_subtask_contacts: [top-pinch, na, insert, push, top-pinch, na, insert, push, na]
"""

ENV_NEW = """env:
  type: gym_manipulator
  name: real_robot
  fps: 30
  # MuJoCo stand-in (lerobot/robots/rebot_b601_sim): a recreated 2026-10-16 nail-clipper layout, top and wrist cameras
  # calibrated on 10-16 ep 5 (lerobot/scripts/sim_builders/real2sim/HANDOFF.md). pos_vel_velocity null = the raw position
  # actuators action_trace_sim was measured with; [40.0 x6, 180.0] would slew like the follower's POS_VEL.
  robot:
    type: rebot_b601_sim
    id: rebot_sim_nail
    # layout: clippers_only (case + two clippers only; default), ep0 / ep3 (hard tier, recorded cluttered layouts), ep1 / ep2
    # (on disk, out of the default set) and the easy-tier scenes of tasks.py; override per run with --env.robot.scene=...
    # (run_sim_eval.sh loops over the registry's LAYOUTS). Each is scenes/<name>/scene.xml: photo table texture,
    # detailed clippers, and the original collision physics; recorded layouts additionally retain their clutter.
    scene: clippers_only
    fps: 30
    substeps: 17
    gravcomp: true
    pos_vel_velocity: null
    episodes: 4  # per layout; the robot raises SIGINT after the last park, the run ends on its own
    auto_subtasks: true  # subtask cursor + staged reward from the MuJoCo state (robots/rebot_b601_sim/tasks.py scorer); last stage = success
    cameras:
      wrist:
        type: mujoco
        camera: wrist
        use_depth: true
        # materials are tuned for the wrist camera; keep the existing uncorrected wrist output
        # (the table texture already carries the wood gain)
        height: 480
        width: 640
        fps: 30
      top:
        type: mujoco
        camera: ext
        # real robot-free, clutter-free background of the layout (scene_background.py v3), relative to the scene dir
        background: background_top.png
        # the render (robot + free objects, wrist-space colours) mapped to the top camera's colours before the composite:
        # per-channel affine through the case foam and black end measured in both cameras (object_materials.py)
        color_gain: [1.222, 1.278, 1.165]
        color_offset: [8.0, 8.5, 28.1]
        # every free body is rendered (a knocked-over bottle showed twice when the clutter kept its real pixels);
        # the sim table (photo texture, one piece) is rendered too: only the surroundings come from the real background
        render_static: [table, table_top]
        render_static_gain: [1.054, 1.231, 1.403]  # RGB, real / rendered table median on wood pixels of 10-16 ep 5 (4 frames)
        height: 480
        width: 640
        fps: 30
  teleop:
    type: sim_autostart
    id: sim_autostart
  processor:
    control_mode: leader
    observation:
      display_cameras: false
      add_joint_velocity_to_observation: false
      add_current_to_observation: false
    image_preprocessing:
      resize_size: null
    gripper:
      use_gripper: true
      gripper_penalty: 0.0
    reset:
      fixed_reset_joint_positions: null
      reset_time_s: 10
      control_time_s: 90.0  # 2026-10-22 eval budget; the real nail episodes run 78-102 s, so 90 s truncates some
      terminate_on_success: true
"""

out = HEADER + src
out = swap(out, "  task: Tidy up the table\n", "  task: Put the nail clippers in the clipper case\n")
i, j = out.index(SUBTASKS_OLD_START), out.index(SUBTASKS_OLD_END) + len(SUBTASKS_OLD_END)
out = out[:i] + SUBTASKS_NEW + out[j:]
out = swap(out, "job_name: rebot-taskmix-20261010\n", "job_name: rebot-sim-nail\n")
out = swap(out, "use_rerun: true\n", "use_rerun: true\nrerun_save: true  # sim runs keep the rerun session as {output_dir}/rerun.rrd (headless; replay with `rerun <file>`)\nrecord_inference_dataset: false  # no OnlineEpisodeRecorder (videos/parquet/depth); the .rrd and subtask_events.json are the record\n")
i = out.index("\nenv:\n") + 1
tail = out[i:]
assert tail.startswith("env:\n  type: gym_manipulator") and "  features:\n" in tail
features = tail[tail.index("  features:\n"):]
out = out[:i] + ENV_NEW + features
(WS / "config_rl_sim.yaml").write_text(out)
print("wrote config_rl_sim.yaml", len(out.splitlines()), "lines")
