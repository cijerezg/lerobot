import pandas as pd

from examples.experiments.run_drtc_experiment import (
    _parse_experiment_dict,
    create_client_config,
    create_robot_config,
)
from lerobot.robots.squint_so101 import SquintSO101Robot, SquintSO101RobotConfig
from lerobot.robots.squint_so101.squint_so101 import (
    infer_squint_env_id,
    read_dataset_action_range,
    read_dataset_episode_actions,
    read_dataset_initial_state,
    read_dataset_task,
)
from lerobot.teleoperators.utils import TeleopEvents


def test_infer_squint_env_id_from_pick_place_task():
    task = "Pick up the orange cube and place it on the black X marker with the white background"

    assert infer_squint_env_id(task) == "SO101PlaceCubeMarker-v1"


def test_read_dataset_task_from_lerobot_v3_tasks_parquet(tmp_path):
    dataset_root = tmp_path / "dataset"
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True)
    task = "Pick up the orange cube and place it on the black X marker with the white background"
    pd.DataFrame({"task_index": [0]}, index=[task]).to_parquet(meta_dir / "tasks.parquet")

    assert read_dataset_task(str(dataset_root)) == task


def test_read_dataset_action_range_from_stats_json(tmp_path):
    dataset_root = tmp_path / "dataset"
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "stats.json").write_text(
        '{"action": {"min": [-1, -2, -3, -4, -5, 0], "max": [1, 2, 3, 4, 5, 6]}}'
    )

    low, high = read_dataset_action_range(str(dataset_root))

    assert low.tolist() == [-1, -2, -3, -4, -5, 0]
    assert high.tolist() == [1, 2, 3, 4, 5, 6]


def test_read_dataset_initial_state_from_lerobot_v3_data(tmp_path):
    dataset_root = tmp_path / "dataset"
    data_dir = dataset_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    expected = [3.0, -98.0, 99.0, 81.0, -27.0, 29.0]
    pd.DataFrame({"observation.state": [expected]}).to_parquet(data_dir / "file-000.parquet")

    state = read_dataset_initial_state(str(dataset_root))

    assert state.tolist() == expected


def test_read_dataset_episode_actions_from_lerobot_v3_data(tmp_path):
    dataset_root = tmp_path / "dataset"
    data_dir = dataset_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    first = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    skipped = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    second = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 1],
            "frame_index": [0, 1, 2, 0],
            "action": [first, skipped, second, skipped],
        }
    ).to_parquet(data_dir / "file-000.parquet")

    actions = read_dataset_episode_actions(str(dataset_root), episode_index=0, stride=2)

    assert actions.tolist() == [first, second]


def test_create_robot_config_builds_squint_config():
    cfg = _parse_experiment_dict(
        {
            "name": "sim",
            "estimator": "jk",
            "cooldown": True,
            "robot_type": "squint_so101",
            "robot_id": "sim_bot",
            "camera1_name": "side",
            "camera2_name": "top",
            "sim_dataset_root": "outputs/so101_pickplace_160_20260501a",
            "sim_video_dir": "outputs/test_videos",
            "sim_reset_seed_on_terminal": True,
            "sim_bootstrap_dataset_episode": 0,
            "sim_bootstrap_dataset_action_stride": 3,
        }
    )

    robot_cfg = create_robot_config(cfg)

    assert isinstance(robot_cfg, SquintSO101RobotConfig)
    assert robot_cfg.id == "sim_bot"
    assert robot_cfg.side_camera_name == "side"
    assert robot_cfg.top_camera_name == "top"
    assert robot_cfg.video_dir == "outputs/test_videos"
    assert robot_cfg.reset_seed_on_terminal is True
    assert robot_cfg.bootstrap_dataset_episode == 0
    assert robot_cfg.bootstrap_dataset_action_stride == 3


def test_rlt_paper_cadence_config_propagates_to_client(tmp_path):
    cfg = _parse_experiment_dict(
        {
            "name": "sim",
            "estimator": "jk",
            "cooldown": True,
            "robot_type": "squint_so101",
            "rlt_critic_updates_per_actor": 2,
            "rlt_success_sample_fraction": 0.5,
            "rlt_intervention_sample_fraction": 0.25,
            "rlt_intervention_reference_mode": "original",
        }
    )

    client_cfg = create_client_config(cfg, tmp_path / "metrics.csv")

    assert client_cfg.rlt_critic_updates_per_actor == 2
    assert client_cfg.rlt_success_sample_fraction == 0.5
    assert client_cfg.rlt_intervention_sample_fraction == 0.25
    assert client_cfg.rlt_intervention_reference_mode == "original"


def test_squint_robot_rlt_events_are_one_shot():
    robot = SquintSO101Robot(SquintSO101RobotConfig())

    robot._pending_start_event = True
    assert robot.get_rlt_events() == {TeleopEvents.START_EPISODE.value: True}
    assert robot.get_rlt_events() == {}

    robot._pending_terminal_event = "success"
    assert robot.get_rlt_events() == {TeleopEvents.SUCCESS.value: True}
    assert robot.get_rlt_events() == {}

    robot._pending_terminal_event = "failure"
    assert robot.get_rlt_events() == {
        TeleopEvents.TERMINATE_EPISODE.value: True,
        TeleopEvents.FAILURE.value: True,
    }
