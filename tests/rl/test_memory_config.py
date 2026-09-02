"""Short-term memory window shape: MemoryConfig is the single source of truth, and
the slot ages it names must reach every consumer that stamps a frame with e(t)."""

import pytest

from lerobot.rl.shared_config import MemoryConfig

KEYS = ["observation.state", "observation.images.top", "depth.wrist.depth"]


def test_explicit_offsets_are_the_window_shape():
    """-6/-4/-2 s @ 30 fps, and the derived pair describes the same window."""
    cfg = MemoryConfig(history_keys=KEYS, history_offsets_seconds=[-6.0, -4.0, -2.0])
    assert cfg.history_times_seconds() == [6.0, 4.0, 2.0]  # oldest → newest
    assert cfg.history_offsets_frames(30) == [180, 120, 60]
    assert cfg.history_offsets(30) == dict.fromkeys(KEYS, [180, 120, 60])
    # Readers of the legacy pair (the probes' "stale" lag, the prompt token budget)
    # must not see a window the slots contradict.
    assert cfg.history_window_seconds == 6.0
    assert cfg.history_num_samples == 3


def test_defaults_match_the_explicit_window():
    """The uniform defaults spell -6/-4/-2 too, so an unset list changes nothing."""
    assert MemoryConfig(history_keys=KEYS).history_offsets_frames(30) == [180, 120, 60]


def test_legacy_uniform_window_is_unchanged():
    """Checkpoints written before history_offsets_seconds carry only the pair; they
    must still resolve to the 5 s / 5-sample ladder they were trained with."""
    cfg = MemoryConfig(history_keys=KEYS, history_window_seconds=5.0, history_num_samples=5)
    assert cfg.history_times_seconds() == [5.0, 4.0, 3.0, 2.0, 1.0]
    assert cfg.history_offsets_frames(30) == [150, 120, 90, 60, 30]


def test_offsets_are_sign_agnostic_and_ordered():
    """Seconds-before-now and lookback magnitudes are the same window, in any order."""
    negative = MemoryConfig(history_keys=KEYS, history_offsets_seconds=[-2.0, -6.0, -4.0])
    positive = MemoryConfig(history_keys=KEYS, history_offsets_seconds=[6, 4, 2])
    assert negative.history_times_seconds() == positive.history_times_seconds() == [6.0, 4.0, 2.0]


@pytest.mark.parametrize(
    ("offsets", "match"),
    [
        ([-6.0, 4.0, -2.0], "not mixed"),  # a sign typo is a different window, not a nudge
        ([-6.0, -6.0], "distinct"),
        ([-6.0, 0.0], "cannot contain 0"),
        ([], "must not be empty"),
    ],
)
def test_malformed_offsets_are_rejected(offsets, match):
    with pytest.raises(ValueError, match=match):
        MemoryConfig(history_keys=KEYS, history_offsets_seconds=offsets)


def test_offsets_that_collapse_on_rounding_are_rejected():
    """Two slots rounding to one step would silently shrink T_h below
    history_num_samples, which sizes the prompt's state-history clause."""
    cfg = MemoryConfig(history_keys=KEYS, history_offsets_seconds=[-0.02, -0.01])
    with pytest.raises(ValueError, match="collapse"):
        cfg.history_offsets_frames(30)


def test_offsets_land_on_the_image_stride_grid():
    """Image/depth history can only address stored rows (buffer._validate_history_stride);
    at image_stride 3 the -6/-4/-2 offsets must clear the grid."""
    from lerobot.rl.buffer import ReplayBuffer

    offsets = MemoryConfig(history_keys=KEYS, history_offsets_seconds=[-6.0, -4.0, -2.0]).history_offsets(30)
    ReplayBuffer._validate_history_stride(ReplayBuffer._normalize_history_offsets(offsets), 3)


def test_rl_config_syncs_slot_ages_to_every_stamping_consumer():
    """e(t) is read in real seconds, so the model must be stamped with the instants the
    buffer gathered. Nothing synced these before 2026-09-01: the MEM video encoder fell
    back to history_stride_seconds=1.0, correct only at the old 5 s / 5-sample window."""
    from lerobot.policies.depth_pointmap.configuration_pointmap import DepthPointmapConfig
    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLConfig

    cfg = MolmoAct2RLConfig(
        memory=MemoryConfig(history_keys=KEYS, history_offsets_seconds=[-6.0, -4.0, -2.0]),
        pointmap_config=DepthPointmapConfig(depth_key="wrist"),
    )
    assert cfg.history_times_seconds == [6.0, 4.0, 2.0]
    assert cfg.history_stride_seconds == 2.0  # the uniform fallback agrees here
    assert cfg.pointmap_config.history_times_seconds == [6.0, 4.0, 2.0]
    assert cfg.pointmap_config.history_num_samples == 3
    assert cfg.pointmap_config.history_window_seconds == 6.0


def test_rl_config_sync_preserves_the_legacy_window():
    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLConfig

    cfg = MolmoAct2RLConfig(
        memory=MemoryConfig(history_keys=KEYS, history_window_seconds=5.0, history_num_samples=5)
    )
    assert cfg.history_times_seconds == [5.0, 4.0, 3.0, 2.0, 1.0]
    assert cfg.history_stride_seconds == 1.0


def test_rl_config_leaves_stamps_unset_when_history_is_off():
    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLConfig

    cfg = MolmoAct2RLConfig(memory=MemoryConfig())
    assert cfg.history_times_seconds is None
