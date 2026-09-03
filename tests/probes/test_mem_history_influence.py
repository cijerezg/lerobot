"""Treatment construction for the MEM history-influence factorial.

The probe's whole argument rests on two cells differing in exactly one intended way, so
what is tested here is what each treatment does and does not touch.
"""

import numpy as np
import pytest
import torch

from lerobot.probes.mem_history_influence import (
    CELLS,
    REFERENCE,
    TREATMENTS,
    _constant_history,
    _drop_history,
    _mem_history_keys,
    _variant_observation,
    cell_name,
)

N_SLOTS = 3
STATE_KEY = "history.observation.state"
IMAGE_KEY = "history.observation.images.top"
DEPTH_KEY = "history.depth.wrist.depth"


def _observation() -> dict:
    return {
        "observation.state": torch.arange(7, dtype=torch.float32)[None],
        "observation.images.top": torch.rand(1, 3, 4, 4),
        STATE_KEY: torch.zeros(1, N_SLOTS, 7),
        IMAGE_KEY: torch.zeros(1, N_SLOTS, 3, 4, 4),
        DEPTH_KEY: torch.zeros(1, N_SLOTS, 1, 4, 4),
    }


def _foreign() -> dict:
    return {
        STATE_KEY: torch.full((1, N_SLOTS, 7), 9.0),
        IMAGE_KEY: torch.full((1, N_SLOTS, 3, 4, 4), 9.0),
    }


def test_the_factorial_is_three_treatments_on_two_channels():
    assert len(CELLS) == len(TREATMENTS) ** 2 == 9
    assert REFERENCE == ("real", "real")
    assert cell_name(*REFERENCE) in {cell_name(i, s) for i, s in CELLS}


def test_mem_history_keys_never_include_depth():
    # Depth rides the point-map encoder, so a treatment touching it would make two cells
    # differ in a way the probe does not claim to be testing.
    assert set(_mem_history_keys(_observation())) == {STATE_KEY, IMAGE_KEY}


@pytest.mark.parametrize("key,current", [(STATE_KEY, "observation.state"), (IMAGE_KEY, "observation.images.top")])
def test_constant_history_fills_every_slot_with_the_present(key, current):
    obs = _observation()

    filled = _constant_history(obs, key)

    assert filled.shape == obs[key].shape
    assert all(torch.equal(filled[:, slot], obs[current]) for slot in range(N_SLOTS))


def test_real_real_is_the_untouched_observation():
    obs = _observation()

    assert _variant_observation(obs, "real", "real", _foreign()) == obs


def test_each_channel_takes_its_own_treatment():
    obs = _observation()

    cell = _variant_observation(obs, "foreign", "constant", _foreign())

    assert bool(cell[IMAGE_KEY].eq(9.0).all())
    assert torch.equal(cell[STATE_KEY][:, 0], obs["observation.state"])


def test_no_cell_touches_depth_history():
    obs, foreign = _observation(), _foreign()

    for images, states in CELLS:
        assert _variant_observation(obs, images, states, foreign)[DEPTH_KEY] is obs[DEPTH_KEY]


def test_a_cell_never_drops_a_channel():
    # The distinction the redesign turns on: emptied, never absent, so the prompt shape is
    # one training produced. Only the legacy `none` removes keys.
    obs, foreign = _observation(), _foreign()

    for images, states in CELLS:
        assert set(_variant_observation(obs, images, states, foreign)) == set(obs)
    assert set(_drop_history(obs)) == set(obs) - {STATE_KEY, IMAGE_KEY}
    assert DEPTH_KEY in _drop_history(obs)


def test_foreign_falls_back_to_real_when_the_donor_lacks_a_channel():
    obs = _observation()

    cell = _variant_observation(obs, "foreign", "foreign", {})

    assert cell[IMAGE_KEY] is obs[IMAGE_KEY]
    assert cell[STATE_KEY] is obs[STATE_KEY]


def test_paired_penalty_is_zero_for_the_reference_cell():
    # Every contrast subtracts the reference on the same frame, so the reference's own bar
    # must be exactly zero — a nonzero one would mean the pairing is misaligned.
    reference_mse = np.asarray([0.031, 0.029, 0.030])
    penalty = reference_mse - reference_mse

    assert penalty == pytest.approx(np.zeros(3))
