from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from lerobot.rl.flux3 import windows
from lerobot.rl.flux3.rl_flux3 import render_caption
from lerobot.rl.flux3.rl_flux3_trainer import COMMAND_WINDOW, MISTAKE, SPEED, STATE_WINDOW, Flux3Trainer
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.gripper_torque import smooth_gripper_torque
from lerobot.utils.motor_velocity import smooth_velocity

CAMERA = "observation.images.top"


def frames():
    """Two episodes, 5 and 4 frames."""
    rng = np.random.default_rng(0)
    n = 9
    return pd.DataFrame(
        {
            "index": range(n),
            "episode_index": [0] * 5 + [1] * 4,
            ACTION: list(rng.normal(size=(n, 7)).astype(np.float32)),
            OBS_STATE: list(rng.normal(size=(n, 7)).astype(np.float32)),
            windows.OBS_VELOCITY: list(rng.normal(size=(n, 7)).astype(np.float32)),
            windows.OBS_EFFORT: list(rng.normal(size=(n, 7)).astype(np.float32)),
        }
    )


def test_windows_hold_at_the_ends_of_the_episode():
    index = windows.windows(np.arange(9)[:, None], frames(), [-2, 0, 3])[..., 0]
    assert index[0].tolist() == [0, 0, 3]
    assert index[3].tolist() == [1, 3, 4]
    assert index[5].tolist() == [5, 5, 8]
    assert index[8].tolist() == [6, 8, 8]


def test_command_targets_are_deltas_within_the_episode_with_the_gripper_absolute():
    table = frames()
    commands = np.stack(table[ACTION])
    targets = windows.command_targets(commands, table, [-1])
    assert np.allclose(targets[[0, 5], :6], 0)
    assert np.allclose(targets[6, :6], commands[6, :6] - commands[5, :6])
    assert np.allclose(targets[:, 6], commands[:, 6])


def test_state_is_position_velocity_torque_smoothed_per_episode():
    table = frames()
    state = windows.states(table)
    assert state.shape == (9, 21) and state.dtype == np.float32
    assert np.allclose(state[:, :7], np.stack(table[OBS_STATE]))
    assert np.allclose(state[5:, 7:14], smooth_velocity(np.stack(table[windows.OBS_VELOCITY])[5:]), atol=1e-6)
    effort = np.stack(table[windows.OBS_EFFORT]).astype(np.float64)
    assert np.allclose(state[5:, 16], smooth_gripper_torque(effort[5:, 2]), atol=1e-6)


def test_caption():
    assert render_caption("tidy up the table", "approach nut", False, "slow") == (
        "The task is to tidy up the table. The current step is approach nut. No mistakes are made. The speed is slow."
    )
    assert "step is grasp nut. Mistakes are made. The speed is fast." in render_caption("t", "grasp nut", True, "fast")


def test_training_batch_orders_the_frames_and_marks_windows_past_the_episode_end():
    cfg = SimpleNamespace(policy=SimpleNamespace(task="fallback", camera_order=[CAMERA]))
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            tasks=pd.DataFrame({"task_index": [0]}, index=["tidy up the table"]),
            subtasks=pd.DataFrame({"subtask_index": [0, 1]}, index=["approach nut", "grasp nut"]),
        )
    )
    image = lambda *values: torch.tensor(values, dtype=torch.uint8)[None, :, None, None, None].expand(2, -1, 3, 2, 2)  # noqa: E731
    raw = {
        "complementary_info": {
            "task_index": torch.tensor([0, 0]),
            "subtask_index": torch.tensor([0, 1]),
            MISTAKE: torch.tensor([False, True]),
            SPEED: torch.tensor([0, 2]),
            STATE_WINDOW: torch.zeros(2, 3, 21),
            COMMAND_WINDOW: torch.zeros(2, 45, 7),
            f"future.{CAMERA}": image(51, 102),
            "future_visual_valid": torch.tensor([True, False]),
        }
    }
    observations = {f"history.{CAMERA}": image(0, 255), CAMERA: image(0)[:, 0]}
    batch = Flux3Trainer().build_training_batch(raw, observations, None, None, lambda b: b, dataset, cfg)
    assert batch[CAMERA].shape == (2, 5, 3, 2, 2)
    assert torch.allclose(batch[CAMERA][0, :, 0, 0, 0], torch.tensor([0, 1, 0, 0.2, 0.4]))
    assert batch["task"][1] == "The task is to tidy up the table. The current step is grasp nut. Mistakes are made. The speed is fast."
    assert batch["action_is_pad"].tolist() == [False, True]
    assert batch[OBS_STATE].shape == (2, 3, 21) and batch[ACTION].shape == (2, 45, 7)


# -- run time ---------------------------------------------------------------------------------------------


def runtime_config(**overrides):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.rl.flux3.rl_flux3 import Flux3RLConfig

    cameras = [CAMERA, "observation.images.wrist"]
    return Flux3RLConfig(
        n_obs_steps=3,
        history_snapshots=3,
        obs_stride=30,
        camera_keys=cameras,
        canvas_hw=(448, 896),
        chunk_size=42,
        n_action_steps=42,
        delta_absolute_dims=[-1],
        input_features={
            **{camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 448, 448)) for camera in cameras},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(21,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        **overrides,
    )


def test_runtime_state_window_is_the_training_state_window():
    from lerobot.rl.flux3.inference import state_window

    rng = np.random.default_rng(1)
    n = 200
    table = pd.DataFrame(
        {
            "index": range(n),
            "episode_index": [0] * n,
            OBS_STATE: list(rng.normal(size=(n, 7)).astype(np.float32)),
            windows.OBS_VELOCITY: list(rng.normal(size=(n, 7)).astype(np.float32)),
            windows.OBS_EFFORT: list(rng.normal(size=(n, 7)).astype(np.float32)),
        }
    )
    offsets = [-60, -30, 0]
    trained = windows.windows(windows.states(table), table, offsets)
    velocity, effort = np.stack(table[windows.OBS_VELOCITY]), np.stack(table[windows.OBS_EFFORT])
    for now in (0, 3, 40, 199):
        # the runtime keeps the last 60 + 5 readings (RTCSharedState.configure_history)
        kept = slice(max(0, now + 1 - 65), now + 1)
        state = state_window(trained[now, :, :7], velocity[kept].tolist(), effort[kept].tolist(), offsets)
        assert np.allclose(state, trained[now], atol=1e-6)


def runtime_observation(second: float):
    image = lambda *values: torch.tensor(values)[None, :, None, None, None].expand(1, -1, 3, 480, 640)  # noqa: E731
    return {
        f"history.{CAMERA}": image(0.0, 0.5),
        CAMERA: image(1.0)[:, 0],
        "history.observation.images.wrist": image(0.0, 0.5),
        "observation.images.wrist": image(1.0)[:, 0],
        f"history.{OBS_STATE}": torch.tensor([[[0.0] * 7, [1.0] * 7]]),
        OBS_STATE: torch.full((1, 7), second),
    }


def test_inference_batch_has_the_training_inputs():
    from lerobot.rl.flux3.inference import COMMAND_HISTORY

    cfg = SimpleNamespace(policy=runtime_config(eval_speed="slow"))
    context = {"preprocessor": lambda b: b, "subtask": "approach nut"}
    readings = [[float(i)] * 7 for i in range(10)]
    batch = Flux3Trainer().build_inference_batch(
        runtime_observation(2.0), "tidy up the table", cfg, velocities=readings, torques=readings,
        last_action=torch.full((1, 7), 5.0), **context,
    )
    assert batch["task"] == [
        "The task is to tidy up the table. The current step is approach nut. No mistakes are made. The speed is slow."
    ]
    assert batch[CAMERA].shape == (1, 3, 3, 448, 448)
    assert batch[CAMERA][0, :, 0, 0, 0].tolist() == [0.0, 0.5, 1.0]
    assert batch[OBS_STATE].shape == (1, 3, 21)
    assert batch[OBS_STATE][0, :, 0].tolist() == [0.0, 1.0, 2.0]
    # readings 5..9 under the weights 0.15 0.15 0.2 0.25 0.25; -1 s and -2 s are held at the first reading
    assert torch.allclose(batch[OBS_STATE][0, :, 7], torch.tensor([0.0, 0.0, 7.3]))
    assert batch[COMMAND_HISTORY].shape == (1, 3, 7) and (batch[COMMAND_HISTORY] == 5.0).all()
    # warm-up and the start of an episode: no reading, no command yet
    batch = Flux3Trainer().build_inference_batch(
        runtime_observation(2.0), "tidy up the table", cfg, velocities=[], torques=[], last_action=None, **context
    )
    assert (batch[OBS_STATE][0, :, 7:] == 0).all() and (batch[COMMAND_HISTORY] == 2.0).all()


def test_runtime_chunk_is_decoded_from_the_last_command(tmp_path):
    import json

    stats = {
        "state": {"q01": [-1.0] * 21, "q99": [1.0] * 21},
        "action": {"q01": [-2.0] * 7, "q99": [2.0] * 7},
    }
    (tmp_path / "config.json").write_text(json.dumps({"normalization_stats": stats}))
    cfg = SimpleNamespace(policy=runtime_config(pretrained_path=tmp_path))
    trainer = Flux3Trainer()
    preprocessor, postprocessor = trainer.make_inference_processors(cfg)
    readings = [[0.5] * 7]
    batch = trainer.build_inference_batch(
        runtime_observation(2.0), "tidy up the table", cfg, preprocessor=preprocessor, subtask="approach nut",
        velocities=readings, torques=readings, last_action=torch.full((1, 7), 5.0),
    )
    assert batch[CAMERA].shape == (1, 3, 3, 448, 448) and batch[OBS_STATE].shape == (1, 3, 21)
    assert torch.allclose(batch[OBS_STATE][0, 0], torch.tensor([0.0] * 7 + [0.5] * 14))
    # the runtime hands over the chunk without its batch axis: [T, 7], normalized
    commands = postprocessor(torch.full((42, 7), 0.5))
    assert commands.shape == (42, 7)
    assert torch.allclose(commands[:, 0], 5.0 + torch.arange(1, 43))  # deltas of 1 from the last command
    assert torch.allclose(commands[:, 6], torch.ones(42))  # the gripper is the command itself


def test_runtime_policy_takes_the_runtime_arguments(monkeypatch):
    from lerobot.policies.flux3 import Flux3Policy
    from lerobot.rl.flux3.inference import Flux3RuntimePolicy

    monkeypatch.setattr(Flux3Policy, "predict_action_chunk", lambda self, batch: batch)
    policy = Flux3RuntimePolicy.__new__(Flux3RuntimePolicy)
    assert policy.predict_action_chunk("b", inference_delay=2, prev_chunk_left_over=None, execution_horizon=12) == "b"
