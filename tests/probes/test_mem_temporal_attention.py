"""Age-profile statistics and the positional controls of the MEM temporal probe."""

import numpy as np
import pytest
import torch

from lerobot.probes.mem_temporal_attention import (
    _age_shape,
    _control_batches,
    _delivered,
    _interior_min_fraction,
    _merge_samples,
    _mistake_age_enrichment,
    _mistake_context,
    _mistake_focus_samples,
    _profile_survival,
    _render_distribution,
    _render_mistake_sequence,
)

# Synthetic layer-mean age shares over the deployed three history frames, oldest → newest.
RECENCY = [0.1, 0.25, 0.65]  # strictly rising, no dip anywhere
TROUGH = [0.45, 0.1, 0.45]  # high at both ends, low in the middle
FLAT = [1 / 3, 1 / 3, 1 / 3]
SECONDS = np.array([6.0, 4.0, 2.0], dtype=np.float32)


def _head_age(*profiles, n_frames: int = 3, n_cameras: int = 2, n_heads: int = 4) -> np.ndarray:
    """(frames, layers, cameras, heads, ages) with one profile repeated per layer."""
    stack = np.asarray(profiles, dtype=np.float32)
    return np.tile(stack[None, :, None, None, :], (n_frames, 1, n_cameras, n_heads, 1))


def test_interior_min_separates_a_trough_from_steep_recency():
    # The reason this is a location and not a curvature: RECENCY is convex, so its
    # endpoints average above its interior and every curvature index scores it like a
    # trough. Its minimum is still at an end, and that is what must read as "no dip".
    fraction = _interior_min_fraction(_head_age(RECENCY, TROUGH, FLAT))

    assert fraction[0] == pytest.approx(0.0)
    assert fraction[1] == pytest.approx(1.0)


def test_interior_min_is_taken_per_head_not_on_the_head_mean():
    # One layer, half its heads peaked on the newest age and half on the oldest. Neither
    # group dips, but their mean does — averaging first would report a trough that no
    # head has.
    rising = np.asarray(RECENCY, dtype=np.float32)
    falling = rising[::-1].copy()
    head_age = np.stack([rising, falling, rising, falling])[None, None, None]

    assert _interior_min_fraction(head_age)[0] == pytest.approx(0.0)
    assert head_age.mean(axis=(0, 1, 2, 3)).argmin() == 1


def test_interior_min_is_zero_when_there_is_no_interior():
    assert _interior_min_fraction(_head_age([0.7, 0.3]))[0] == pytest.approx(0.0)


def test_survival_is_one_for_a_control_that_reproduces_the_profile():
    profile = np.asarray([TROUGH], dtype=np.float32)

    assert _profile_survival(profile, profile)[0] == pytest.approx(1.0)
    assert _profile_survival(profile, profile * 7.0)[0] == pytest.approx(1.0)  # level-free


def test_survival_is_zero_when_the_control_flattens_the_profile():
    survival = _profile_survival(np.asarray([TROUGH], dtype=np.float32), np.asarray([FLAT], dtype=np.float32))

    assert survival[0] == pytest.approx(0.0, abs=1e-6)


def test_survival_is_nan_when_the_real_profile_has_no_shape_to_lose():
    survival = _profile_survival(np.asarray([FLAT], dtype=np.float32), np.asarray([TROUGH], dtype=np.float32))

    assert np.isnan(survival[0])


def test_survival_is_nan_for_a_nearly_flat_profile_not_only_an_exactly_flat_one():
    # A guard that catches only exact flatness divides one small number by another and can
    # draw a large survival bar for a profile with no age preference to speak of.
    nearly_flat = np.asarray([[0.336, 0.331, 0.333]], dtype=np.float32)

    assert np.isnan(_profile_survival(nearly_flat, nearly_flat * 1.1)[0])
    assert not np.isnan(_profile_survival(np.asarray([TROUGH], dtype=np.float32), nearly_flat)[0])


def _batch(n_ages: int = 3) -> dict:
    frames = torch.arange(1 * 2 * n_ages * 4 * 3, dtype=torch.float32).reshape(1, 2, n_ages, 4, 3)
    return {
        "history_images": frames,
        "history_image_times": torch.as_tensor(SECONDS[:n_ages]),
        "history_images_mask": torch.tensor([True]),
    }


def test_constant_control_puts_the_same_frame_in_every_slot():
    batch = _batch()
    constant = _control_batches(batch, np.random.default_rng(0))["constant"]["history_images"]

    newest = batch["history_images"][:, :, -1]
    assert all(torch.equal(constant[:, :, age], newest) for age in range(constant.shape[2]))


def test_shuffled_control_reorders_the_same_frames():
    batch = _batch()
    shuffled = _control_batches(batch, np.random.default_rng(0))["shuffled"]["history_images"]

    real = batch["history_images"]
    assert not torch.equal(shuffled, real)  # never the identity: that is the real condition
    assert torch.equal(shuffled.sum(dim=2), real.sum(dim=2))  # same frames, moved


def test_controls_leave_the_age_embedding_and_the_original_batch_alone():
    batch = _batch()
    before = batch["history_images"].clone()
    controls = _control_batches(batch, np.random.default_rng(0))

    # times carries the age; holding it fixed is what makes position the only cue left.
    for control in controls.values():
        assert control["history_image_times"] is batch["history_image_times"]
        assert control["history_images_mask"] is batch["history_images_mask"]
    assert torch.equal(batch["history_images"], before)


def test_no_controls_when_the_window_holds_one_frame():
    assert _control_batches(_batch(n_ages=1), np.random.default_rng(0)) == {}


@pytest.mark.parametrize("with_control", [True, False])
def test_figure_renders_with_and_without_the_control_column(tmp_path, with_control):
    head_age = _head_age(RECENCY, TROUGH, FLAT, n_frames=4)
    age_mean = head_age.mean(axis=(0, 2, 3))
    survival = {"constant": _profile_survival(age_mean, age_mean)} if with_control else {}
    output = tmp_path / "temporal_attention.png"

    _render_distribution(
        np.linspace(0.5, 0.8, 4 * 3).reshape(4, 3).astype(np.float32),
        head_age,
        _age_shape(head_age, SECONDS),
        [3, 11, 23],
        ["top", "wrist"],
        SECONDS,
        [{"episode_idx": i // 2, "frame_idx": i * 30, "global_idx": i} for i in range(4)],
        3 / 4,
        "test provenance",
        survival,
        None,
        str(output),
    )

    assert output.is_file() and output.stat().st_size > 0


def test_age_shape_reports_the_dip_alongside_the_slope():
    shape = _age_shape(_head_age(RECENCY, TROUGH), SECONDS)

    assert shape["slope"][0] < 0  # recency
    assert shape["interior_min"][0] == pytest.approx(0.0)
    assert shape["interior_min"][1] == pytest.approx(1.0)


def _contribution(n_frames=3, n_layers=2, n_cameras=2, n_heads=4, n_keys=4) -> np.ndarray:
    """w_k ||v_k|| per frame, layer, camera, head, key — history first, current last."""
    return np.ones((n_frames, n_layers, n_cameras, n_heads, n_keys), dtype=np.float32)


def test_delivered_past_share_counts_every_key_but_the_current_frame():
    # Three history keys and the current frame, all delivering equally: history's share of
    # what left the block is 3/4, the same value an indifferent softmax puts on it.
    delivered = _delivered(_contribution())

    # (frames, layers), the same shape as temporal_mass so the two sit on one axis.
    assert delivered["past_share"].shape == (3, 2)
    assert delivered["past_share"] == pytest.approx(3 / 4)


def test_delivered_past_share_collapses_when_history_values_are_small():
    # The sink: every key attended equally, but the history keys carry nothing.
    contribution = _contribution()
    contribution[..., :-1] *= 0.01

    assert _delivered(contribution)["past_share"] == pytest.approx(0.03 / 1.03)


def test_delivered_age_share_is_renormalised_over_history_alone():
    # So it lands on the same axis as the attended age profile, current frame excluded.
    contribution = _contribution()
    contribution[..., :-1] = np.asarray(TROUGH, dtype=np.float32)
    contribution[..., -1] = 99.0  # the current frame must not enter the age profile

    age = _delivered(contribution)["age_share"]

    assert age.sum(axis=-1) == pytest.approx(np.ones(2))
    assert age[0] == pytest.approx(np.asarray(TROUGH) / np.sum(TROUGH), abs=1e-6)


def test_captured_contribution_is_the_weight_times_the_value_norm():
    # Guards the axis alignment in the capture: weights are (BC, heads, patches, q, k) and
    # v is (BC, k, patches, heads, dim), so the permute is the whole correctness argument.
    torch.manual_seed(0)
    bc, frames, patches, heads, dim = 2, 4, 7, 4, 8
    weights = torch.rand(bc, heads, patches, frames, frames)
    v = torch.randn(bc, frames, patches, heads, dim)

    value_norm = v.float().norm(dim=-1).permute(0, 3, 2, 1)
    contribution = weights[:, :, :, -1, :] * value_norm

    for key in range(frames):
        expected = weights[1, 2, 3, -1, key] * v[1, key, 3, 2].norm()
        assert contribution[1, 2, 3, key] == pytest.approx(float(expected), rel=1e-5)


def test_mistake_context_marks_current_and_exact_history_ages():
    spans = [
        {
            "episode_idx": 0,
            "from_index": 100,
            "to_index": 180,
            "mistake_type": "failed_close",
            "note": "test",
        }
    ]
    offsets = np.asarray([180, 120, 60])

    current = _mistake_context(0, 170, spans, offsets)
    recovery = _mistake_context(0, 230, spans, offsets)

    assert current["mistake_current"] is True
    assert current["mistake_history_age_mask"] == [False, False, True]
    assert recovery["mistake_current"] is False
    assert recovery["mistake_in_history"] is True
    assert recovery["mistake_history_age_mask"] == [False, True, True]
    assert recovery["mistake_types"] == ["failed_close"]


def test_mistake_age_enrichment_removes_the_number_of_labelled_slots():
    head_age = np.ones((2, 3, 2, 4, 3), dtype=np.float32)
    masks = np.asarray([[False, False, True], [False, True, True]], dtype=bool)

    share, enrichment = _mistake_age_enrichment(head_age, masks)

    assert share[0] == pytest.approx(np.full(3, 1 / 3))
    assert share[1] == pytest.approx(np.full(3, 2 / 3))
    assert enrichment == pytest.approx(np.ones((2, 3)))


def test_mistake_age_enrichment_is_nan_without_a_labelled_history_age():
    _share, enrichment = _mistake_age_enrichment(
        np.ones((1, 2, 1, 1, 3), dtype=np.float32), np.zeros((1, 3), dtype=bool)
    )

    assert np.isnan(enrichment).all()


class _TinyDataset:
    def __init__(self, n_frames: int = 200):
        self.hf_dataset = [
            {"episode_index": torch.tensor(0), "frame_index": torch.tensor(i)} for i in range(n_frames)
        ]

    def __len__(self):
        return len(self.hf_dataset)


def test_mistake_focus_sampling_adds_late_event_and_recovery_on_stride_grid():
    dataset = _TinyDataset()
    spans = [
        {
            "episode_idx": 0,
            "from_index": 60,
            "to_index": 120,
            "mistake_type": "failed_close",
            "note": "test",
        }
    ]

    samples = _mistake_focus_samples(dataset, spans, fps=30.0, stride=3)

    assert samples == [(0, 117, 117), (0, 150, 150)]


def test_merge_samples_deduplicates_and_orders_by_episode_time():
    assert _merge_samples(
        [(1, 9, 109), (0, 6, 6)],
        [(0, 3, 3), (0, 6, 6)],
    ) == [(0, 3, 3), (0, 6, 6), (1, 9, 109)]


def test_mistake_sequence_renders_every_history_age(tmp_path, monkeypatch):
    import lerobot.probes.mem_temporal_attention as temporal

    current = {
        "observation.images.top": torch.zeros(1, 3, 27, 27),
        "observation.images.wrist": torch.ones(1, 3, 27, 27),
    }
    history = {
        "history.observation.images.top": torch.rand(1, 3, 3, 27, 27),
        "history.observation.images.wrist": torch.rand(1, 3, 3, 27, 27),
    }
    monkeypatch.setattr(
        temporal,
        "get_frame_data",
        lambda *_args, **_kwargs: (current, None, None, "", "", 0, 117),
    )
    monkeypatch.setattr(
        temporal,
        "assemble_frame_history",
        lambda *_args, **_kwargs: history,
    )
    diagnostic = {
        "global_idx": 117,
        "camera_keys": ["observation.images.top", "observation.images.wrist"],
        "history_seconds": SECONDS,
        "head_age": np.full((2, 2, 4, 3), 0.1, dtype=np.float32),
        "patch_age": np.linspace(0.05, 0.25, 2 * 2 * 9 * 3, dtype=np.float32).reshape(2, 2, 9, 3),
        "mistake_current": True,
        "mistake_history_age_mask": [False, False, True],
        "mistake_types": ["failed_close"],
    }
    output = tmp_path / "mistake.png"

    _render_mistake_sequence(object(), object(), 30.0, diagnostic, 1 / 4, (0.05, 0.25), str(output))

    assert output.is_file() and output.stat().st_size > 0


def test_figure_renders_the_delivery_panel(tmp_path):
    head_age = _head_age(RECENCY, TROUGH, FLAT, n_frames=4)
    contribution = _contribution(n_frames=4, n_layers=3)
    output = tmp_path / "temporal_attention.png"

    _render_distribution(
        np.linspace(0.5, 0.8, 4 * 3).reshape(4, 3).astype(np.float32),
        head_age,
        _age_shape(head_age, SECONDS),
        [3, 11, 23],
        ["top", "wrist"],
        SECONDS,
        [{"episode_idx": i // 2, "frame_idx": i * 30, "global_idx": i} for i in range(4)],
        3 / 4,
        "test provenance",
        {},
        _delivered(contribution),
        str(output),
    )

    assert output.is_file() and output.stat().st_size > 0
