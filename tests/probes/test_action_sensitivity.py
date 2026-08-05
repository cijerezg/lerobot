"""Contracts for subtask-conditioned action sensitivity."""

from types import SimpleNamespace

import numpy as np
import torch

from lerobot.probes import action_drift_jacobian as probe
from lerobot.probes.base import ActionSensitivityResult, AttentionCaptureResult


def test_action_groups_match_rebot_schema():
    assert probe.ACTION_GROUPS == {
        "proximal_arm": [0, 1, 2],
        "wrist": [3, 4, 5],
        "gripper": [6],
    }


def test_even_sampling_preserves_endpoints():
    values = [(index, 100 + index) for index in range(11)]
    sampled = probe._evenly_spaced(values, 3)
    assert sampled == [values[0], values[5], values[-1]]


class _Dataset:
    def __init__(self):
        self.hf_dataset = []
        for episode in range(2):
            for frame in range(12):
                self.hf_dataset.append(
                    {
                        "episode_index": torch.tensor(episode),
                        "frame_index": torch.tensor(frame),
                    }
                )

    def __len__(self):
        return len(self.hf_dataset)


def test_subtask_sampler_keeps_each_episode_subtask(monkeypatch):
    monkeypatch.setattr(probe, "probe_image_stride", lambda cfg: 2)
    monkeypatch.setattr(probe, "get_subtask_idx", lambda dataset, index: (index % 12) // 6)
    monkeypatch.setattr(probe, "get_subtask_str", lambda dataset, index: ["approach", "grasp"][index])
    cfg = SimpleNamespace(
        probe_parameters=SimpleNamespace(
            random_seed=3,
            max_episodes=2,
            n_frames_per_episode=8,
            action_sensitivity_frames_per_subtask=2,
            attn_eval_episodes=None,
        )
    )
    rows = probe.build_subtask_samples(_Dataset(), cfg)
    assert len(rows) == 8
    assert {(row["episode"], row["subtask"]) for row in rows} == {
        (0, "approach"),
        (0, "grasp"),
        (1, "approach"),
        (1, "grasp"),
    }
    assert all(row["frame"] % 2 == 0 for row in rows)


def _record(episode, subtask, grid_value, prompt_value):
    grid = np.full((2, 2), grid_value, dtype=np.float32)
    group = {
        "grids": {"img_wrist": grid},
        "prompt": [{"clause": "subtask", "label": "grasp", "value": prompt_value}],
    }
    return {
        "episode": episode,
        "subtask": subtask,
        "groups": dict.fromkeys(probe.ACTION_GROUPS, group),
    }


def test_aggregate_is_episode_balanced_not_frame_pooled():
    records = [
        _record(0, "grasp", 0.0, 0.0),
        _record(0, "grasp", 0.0, 0.0),
        _record(1, "grasp", 10.0, 10.0),
    ]
    mean_grid, n_episodes, n_frames = probe._episode_balanced_grid(
        records, "grasp", "gripper", "img_wrist"
    )
    assert np.allclose(mean_grid, 5.0)
    assert (n_episodes, n_frames) == (2, 3)
    prompt = probe._aggregate_prompt(records, "grasp", "gripper")
    assert np.isclose(prompt[("subtask", "grasp")], 5.0)


def test_score_view_keeps_raw_token_values():
    metadata = AttentionCaptureResult(
        cross_attn_by_layer={},
        self_attn_by_layer={},
        encoder_segments=[],
        encoder_pad_masks=None,
        image_tensors=[],
        patches_per_cam=0,
        task_tokens=None,
        subtask_tokens=None,
        tokenizer=None,
    )
    capture = ActionSensitivityResult(
        scores_by_group={"gripper": torch.tensor([1.0, 2.0, 3.0])},
        token_metadata=metadata,
        action_groups={"gripper": [6]},
        timestep=0.5,
        num_flow_samples=8,
        num_projections=4,
    )
    view = probe._score_view(capture, "gripper")
    assert view.cross_attn_by_layer[0].shape == (1, 1, 1, 3)
    assert torch.equal(view.cross_attn_by_layer[0].flatten(), torch.tensor([1.0, 2.0, 3.0]))


def test_detailed_and_aggregate_artifacts_render(tmp_path):
    image = np.full((24, 24, 3), 96, dtype=np.uint8)
    # The point-map depth block is a real panel here and is NOT square (12x16 in the
    # policy): every step from the grids to imshow has to carry its (rows, cols).
    panels = {"img_wrist": (2, 2), "depth": (2, 3)}
    records = []
    for episode, subtask, scale in [(0, "approach", 1.0), (1, "grasp", 2.0)]:
        groups = {}
        for group in probe.ACTION_GROUPS:
            grids, grid_images = {}, {}
            for panel, (rows, cols) in panels.items():
                grid = np.linspace(0.2, 0.8, rows * cols, dtype=np.float32).reshape(rows, cols) * scale
                grids[panel] = grid
                grid_images[panel] = {
                    "cam_name": panel,
                    "img_np": image,
                    "per_head_grid": torch.from_numpy(grid).reshape(1, -1),
                    "grid_hw": (rows, cols),
                }
            groups[group] = {
                "scores": np.arange(8, dtype=np.float32) * scale,
                "grids": grids,
                "grid_images": grid_images,
                "prompt": [
                    {"index": 0, "clause": "subtask", "label": subtask, "value": 0.5 * scale}
                ],
            }
        records.append(
            {
                "episode": episode,
                "frame": 10,
                "global_idx": 10 + 100 * episode,
                "subtask": subtask,
                "groups": groups,
            }
        )

    camera_scales, prompt_scales = probe._global_scales(records)
    html_records = probe._write_details(tmp_path, records)
    aggregate = probe._plot_aggregate_maps(tmp_path, records, camera_scales)
    prompts = probe._plot_prompt_maps(tmp_path, records)
    dashboard = probe._write_html(tmp_path, html_records, camera_scales, prompt_scales)

    assert len(aggregate) == 3
    assert len(prompts) == 3
    assert all((tmp_path / filename).is_file() for filename in aggregate + prompts + [dashboard])
    document = (tmp_path / dashboard).read_text()
    assert "Subtask-conditioned action sensitivity" in document
    assert "Proximal arm" in document
    assert "img_wrist" in document
    assert html_records[0]["cameras"]["depth"]["hw"] == [2, 3]
    raw = np.load(tmp_path / html_records[0]["raw"])
    assert raw["gripper__depth_grid"].shape == (2, 3)
