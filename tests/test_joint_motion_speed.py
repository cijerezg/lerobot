import numpy as np
import pytest
from lerobot.data_processing.annotate import joint_motion_speed as motion


def test_back_and_forth_motion_stays_fast_despite_zero_net_progress():
    q = np.array([[0.], [1.], [0.], [1.], [0.]])
    values, valid = motion.motion_trace(q, 10)
    assert q[-1, 0] == q[0, 0]
    assert np.all(values == 10) and valid.all()
    cell = motion.fit_group(np.linspace(0, 2, 1000))
    assert motion.summarize(values, valid, cell)['speed'] == 5


def test_rate_correction_matches_same_physical_motion():
    for rate in (10, 15, 30):
        q = (np.arange(rate * 3) / rate)[:, None]
        values, valid = motion.motion_trace(q, rate)
        assert valid.all()
        assert values == pytest.approx(np.ones(len(q)-1))


def test_stationary_motion_is_lowest_bucket_with_informative_reference():
    values, valid = motion.motion_trace(np.ones((30, 7)), 15)
    cell = motion.fit_group(np.linspace(0.01, 2, 1000))
    assert motion.summarize(values, valid, cell)['speed'] == 1


def test_nonfinite_states_default_instead_of_looking_stationary():
    q = np.zeros((30, 6)); q[15] = np.nan
    values, valid = motion.motion_trace(q, 30)
    row = motion.summarize(values, valid, motion.fit_group(np.linspace(0, 2, 1000)))
    assert row['speed'] == 3 and row['speed_source'] == 'default_unclear'
    assert 'nonfinite' in row['speed_default_reason']


def test_short_and_degenerate_data_default_to_three():
    cell = motion.fit_group(np.linspace(0, 2, 1000))
    assert motion.summarize(np.array([]), np.array([], dtype=bool), cell)['speed'] == 3
    assert not motion.fit_group(np.zeros(1000))['usable']
    assert motion.bucket(0, motion.fit_group(np.zeros(1000))) == 3


def test_typical_motion_uses_median_without_peak_domination():
    values = np.r_[np.full(90, .5), np.full(10, 10.)]
    cell = motion.fit_group(np.linspace(0, 1, 1000))
    row = motion.summarize(values, np.ones(100, dtype=bool), cell)
    assert row['motion_median_rad_s'] == .5
    assert row['speed'] == 3
