"""The origin the MEM regime probe measures its verdicts from.

Every Delta, tau and verdict in this probe is a difference against one baseline condition,
so what that condition *is* decides what the verdicts mean.
"""

import numpy as np
import pytest
import torch

from lerobot.probes.mem_history_regime import _emptied_observation, _variant_observation

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


def test_emptied_keeps_every_history_key_present():
    # The whole point of the re-anchor: present and correctly shaped, never absent, so the
    # prompt is a shape training produced on every step.
    obs = _observation()

    emptied = _emptied_observation(obs)

    assert set(emptied) == set(obs)
    for key in (STATE_KEY, IMAGE_KEY):
        assert emptied[key].shape == obs[key].shape


def test_emptied_fills_every_slot_with_the_present():
    obs = _observation()

    emptied = _emptied_observation(obs)

    for key, current in ((STATE_KEY, "observation.state"), (IMAGE_KEY, "observation.images.top")):
        assert all(torch.equal(emptied[key][:, slot], obs[current]) for slot in range(N_SLOTS))


def test_emptied_leaves_depth_history_alone():
    # Depth rides the point-map encoder; touching it would make the origin differ from the
    # real window in a way this probe does not claim to test.
    obs = _observation()

    assert _emptied_observation(obs)[DEPTH_KEY] is obs[DEPTH_KEY]


def test_emptied_is_not_the_legacy_drop():
    obs = _observation()

    emptied = _emptied_observation(obs)
    dropped = _variant_observation(obs, False)

    assert STATE_KEY in emptied and STATE_KEY not in dropped
    assert IMAGE_KEY in emptied and IMAGE_KEY not in dropped
    assert DEPTH_KEY in dropped  # the drop spares depth too


def test_emptied_does_not_mutate_the_real_observation():
    obs = _observation()
    before = {key: value.clone() for key, value in obs.items()}

    _emptied_observation(obs)

    assert all(torch.equal(obs[key], before[key]) for key in obs)


def test_content_gain_is_unaffected_by_the_choice_of_origin():
    # G = MSE(stale) - MSE(full) never mentions the origin, which is why the probe's z-test
    # was the one number the old anchoring did not contaminate.
    mse = {"full": 0.0130, "stale": 0.0180, "emptied": 0.0172, "none": 0.0515}

    assert mse["stale"] - mse["full"] == pytest.approx(
        (mse["emptied"] - mse["full"]) - (mse["emptied"] - mse["stale"])
    )
    assert mse["stale"] - mse["full"] == pytest.approx(
        (mse["none"] - mse["full"]) - (mse["none"] - mse["stale"])
    )


def test_the_origin_changes_delta_and_therefore_the_verdicts():
    # rebot ckpt400 scale: the dropped prompt costs ~9x what the content does, so anchoring
    # there pushes frames past tau that the content never moved.
    mse_full, mse_emptied, mse_none = 0.0130, 0.0172, 0.0515
    tau = 0.0050

    delta_emptied = mse_emptied - mse_full
    delta_none = mse_none - mse_full

    assert delta_emptied < tau < delta_none  # "indistinguishable" becomes "helped"
    assert np.isclose(delta_none / delta_emptied, 9.17, atol=0.1)
