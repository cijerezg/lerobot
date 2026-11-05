"""The input-swap cube is built from ``probe_parameters.input_swap_factors``: key routing,
cell texts and the variance split must follow whatever switches are named."""

import numpy as np
import pytest
import torch

from lerobot.probes import input_swap as isw

OBS = {
    "observation.state": torch.zeros(1, 7),
    "observation.images.external_0": torch.zeros(1, 3, 8, 8),
    "observation.images.wrist_0": torch.zeros(1, 3, 8, 8),
    "probe_complementary.camera_is_present.observation.images.wrist_0": torch.tensor([True]),
    "observation.depth.wrist_0": torch.zeros(1, 1, 8, 8),
    "probe_complementary.depth.wrist_0.depth_is_present": torch.tensor([True]),
    "history.observation.state": torch.zeros(1, 3, 7),
    "history.observation.images.external_0": torch.zeros(1, 3, 3, 8, 8),
    "history.observation.images.wrist_0": torch.zeros(1, 3, 3, 8, 8),
    "history.depth.wrist_0.depth": torch.zeros(1, 3, 8, 8),
}


def test_default_cube_is_the_old_sit_cube():
    cube = isw.cube_from_spec(None)
    assert cube.factors == ("state", "image", "subtask")
    assert cube.letters == "SIT" and cube.n_cells == 8 and cube.text_bit == 4
    assert cube.code(cube.all_donor) == "SIT" and cube.label(2) == "I only"


def test_spec_validation():
    with pytest.raises(ValueError):
        isw.cube_from_spec("state,images")
    with pytest.raises(ValueError):
        isw.cube_from_spec("state,state")
    with pytest.raises(ValueError):
        isw.cube_from_spec("image")


def test_history_rides_with_its_stream_without_its_own_switch():
    groups, fixed = isw._factor_keys(OBS, isw.cube_from_spec("state,image,subtask"))
    assert set(groups) == {"state", "image"}
    assert "history.observation.state" in groups["state"]
    assert "history.observation.images.wrist_0" in groups["image"]
    assert "probe_complementary.camera_is_present.observation.images.wrist_0" in groups["image"]
    assert set(fixed) == {
        "observation.depth.wrist_0",
        "probe_complementary.depth.wrist_0.depth_is_present",
        "history.depth.wrist_0.depth",
    }


def test_history_switches_take_the_windows():
    cube = isw.cube_from_spec("state,image,state_history,image_history")
    assert cube.letters == "SIPH" and cube.n_cells == 16 and cube.text_bit == 0 and not cube.has_text
    groups, _ = isw._factor_keys(OBS, cube)
    assert groups["state"] == ["observation.state"]
    assert groups["state_history"] == ["history.observation.state"]
    assert sorted(groups["image_history"]) == [
        "history.observation.images.external_0",
        "history.observation.images.wrist_0",
    ]
    assert all(not k.startswith("history.") for k in groups["image"])
    # Every cell's prompt keeps the anchor's text when T is not a switch.
    assert isw._cell_subtasks("a", "b", cube) == ["a"] * 16
    assert isw._cell_subtasks("a", "b", isw.cube_from_spec("state,subtask")) == ["a", "a", "b", "b"]


def test_history_switch_without_memory_is_an_error():
    no_memory = {k: v for k, v in OBS.items() if not k.startswith("history.")}
    with pytest.raises(KeyError, match="image_history"):
        isw._factor_keys(no_memory, isw.cube_from_spec("state,image,image_history"))


def test_cell_sources_follow_the_mask():
    cube = isw.cube_from_spec("state,image,state_history,image_history")
    groups, fixed = isw._factor_keys(OBS, cube)
    donor = {k: torch.ones_like(v) if v.dtype != torch.bool else v for k, v in OBS.items()}
    cell = isw._cell(OBS, donor, cube.bit("image_history"), groups, fixed, cube)
    assert cell["history.observation.images.wrist_0"].sum() > 0
    assert cell["observation.images.wrist_0"].sum() == 0
    assert cell["history.observation.state"].sum() == 0
    assert cell["history.depth.wrist_0.depth"].sum() == 0


def test_variance_shares_sum_to_one_for_four_switches():
    cube = isw.cube_from_spec("state,image,state_history,image_history")
    rng = np.random.default_rng(0)
    cells = rng.normal(size=(16, 5, 3))
    shares = isw._factorial_shares(cells, cube)
    assert set(shares) == {*cube.factors, "two_way", "three_way", "four_way", "spread"}
    assert sum(v for k, v in shares.items() if k != "spread") == pytest.approx(1.0)
    effects = isw._main_effects(cells, cube)
    assert set(effects) == set(cube.factors)


def test_state_follow_reads_the_state_cell_wherever_it_sits():
    cube = isw.cube_from_spec("image,state")
    unnorm = np.zeros((4, 3, 2))
    unnorm[cube.bit("state")] = 2.0  # the S-only cell is bit 1 here
    follow = isw._state_follow(unnorm, np.zeros(2), np.array([1.0, 1.0]), np.ones(2), cube)
    assert np.allclose(follow, 2.0)
    assert (
        isw._state_follow(
            unnorm, np.zeros(2), np.ones(2), np.ones(2), isw.cube_from_spec("image,image_history")
        )
        is None
    )
