"""Per-sample depth presence, intrinsics and temporal memory (plan phase H).

Gate H: changing valid depth at any of -6/-4/-2 changes the fused depth memory;
changing a placeholder whose presence mask is false cannot change anything; and valid
current/history depth activates the temporal path and receives gradients.

The mixed corpus makes this a per-SAMPLE question for the first time: 53,257 of the
60,728 diverse anchors have no depth at all, so "the depth tensor is absent" stopped
being the right test.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="depth pointmap imports policy deps")

from lerobot.policies.depth_pointmap.configuration_pointmap import DepthPointmapConfig  # noqa: E402
from lerobot.policies.depth_pointmap.modeling_pointmap import (  # noqa: E402
    DepthPointmapEncoder,
    back_project,
)

CPU = torch.device("cpu")


def _config(**overrides):
    kwargs = {
        "image_size": (80, 80),
        "patch_size": 40,
        "depth_units_mm": 1.0,
        "history_num_samples": 3,
        "depth_key": "wrist_0",
    }
    kwargs.update(overrides)
    return DepthPointmapConfig(**kwargs)


def _encoder(**overrides):
    torch.manual_seed(0)
    return DepthPointmapEncoder(_config(**overrides), d_mem=16).eval()


def _batch(current, history=None, **extra):
    batch = {"observation.depth.wrist_0": current}
    if history is not None:
        batch["history.depth.wrist_0.depth"] = history
    batch.update(extra)
    return batch


# ── Per-sample intrinsics ────────────────────────────────────────────────────


def test_back_project_accepts_one_row_per_sample() -> None:
    depth = torch.full((2, 8, 8), 300.0)
    shared = back_project(
        depth, intrinsics=(100.0, 100.0, 4.0, 4.0), depth_units_mm=1.0, z_min_mm=0.0, z_max_mm=1e4
    )
    per_sample = back_project(
        depth,
        intrinsics=torch.tensor([[100.0, 100.0, 4.0, 4.0], [200.0, 200.0, 4.0, 4.0]]),
        depth_units_mm=1.0,
        z_min_mm=0.0,
        z_max_mm=1e4,
    )
    # Row 0 got the same camera; row 1 a different one, so its X/Y must differ.
    assert torch.allclose(shared[0], per_sample[0])
    assert not torch.allclose(shared[1], per_sample[1])
    # Z and the validity mask never depend on the focal length.
    assert torch.allclose(shared[:, 2:], per_sample[:, 2:])


def test_the_encoder_uses_the_batchs_intrinsics_when_it_carries_them() -> None:
    encoder = _encoder()
    current = torch.full((2, 80, 80), 300.0)
    default = encoder.memory_from_batch(_batch(current), batch_size=2, device=CPU)
    override = encoder.memory_from_batch(
        _batch(
            current,
            **{"depth.wrist_0.intrinsics": torch.tensor([[50.0, 50.0, 40.0, 40.0]] * 2)},
        ),
        batch_size=2,
        device=CPU,
    )
    assert not torch.allclose(default, override)


def test_a_wrong_sized_intrinsics_table_is_refused() -> None:
    encoder = _encoder()
    with pytest.raises(ValueError, match="intrinsics has"):
        encoder.memory_from_batch(
            _batch(
                torch.full((2, 80, 80), 300.0),
                **{"depth.wrist_0.intrinsics": torch.zeros(3, 4)},
            ),
            batch_size=2,
            device=CPU,
        )


# ── Gate H: per-sample presence ──────────────────────────────────────────────


def test_an_absent_sample_takes_the_null_bank_while_its_neighbour_does_not() -> None:
    encoder = _encoder()
    current = torch.stack([torch.full((80, 80), 300.0), torch.full((80, 80), 300.0)])
    present = torch.tensor([True, False])
    memory, valid = encoder.memory_from_batch(
        _batch(current, **{"depth.wrist_0.depth_is_present": present}),
        batch_size=2,
        device=CPU,
        return_valid_mask=True,
    )
    assert valid.tolist() == [True, False]
    assert torch.allclose(memory[1], encoder.null_memory(1)[0])
    assert not torch.allclose(memory[0], memory[1])


def test_a_placeholder_cannot_change_anything_when_its_presence_is_false() -> None:
    """The decisive one: garbage in an absent row must not move that row's tokens."""
    encoder = _encoder()
    present = torch.tensor([True, False])
    history_shape = (2, 3, 80, 80)
    base_current = torch.stack([torch.full((80, 80), 300.0), torch.full((80, 80), 300.0)])
    base_history = torch.full(history_shape, 250.0)

    def run(current, history):
        return encoder.memory_from_batch(
            _batch(current, history, **{"depth.wrist_0.depth_is_present": present}),
            batch_size=2,
            device=CPU,
        )

    baseline = run(base_current, base_history)
    poisoned_current = base_current.clone()
    poisoned_current[1] = 999.0
    poisoned_history = base_history.clone()
    poisoned_history[1] = 12.0
    poisoned = run(poisoned_current, poisoned_history)
    assert torch.allclose(baseline, poisoned)


def test_changing_valid_depth_at_any_history_age_changes_the_memory() -> None:
    encoder = _encoder()
    current = torch.full((1, 80, 80), 300.0)
    present = torch.tensor([True])
    history = torch.full((1, 3, 80, 80), 250.0)
    baseline = encoder.memory_from_batch(
        _batch(current, history, **{"depth.wrist_0.depth_is_present": present}),
        batch_size=1,
        device=CPU,
    )
    for slot in range(3):  # -6 s, -4 s, -2 s
        moved = history.clone()
        moved[:, slot] = 400.0
        changed = encoder.memory_from_batch(
            _batch(current, moved, **{"depth.wrist_0.depth_is_present": present}),
            batch_size=1,
            device=CPU,
        )
        assert not torch.allclose(baseline, changed), f"history slot {slot} does not reach the memory"


def test_valid_depth_receives_gradients_and_an_absent_row_does_not() -> None:
    encoder = DepthPointmapEncoder(_config(), d_mem=16).train()
    current = torch.stack([torch.full((80, 80), 300.0), torch.full((80, 80), 300.0)]).requires_grad_(True)
    history = torch.full((2, 3, 80, 80), 250.0).requires_grad_(True)
    memory = encoder.memory_from_batch(
        _batch(current, history, **{"depth.wrist_0.depth_is_present": torch.tensor([True, False])}),
        batch_size=2,
        device=CPU,
    )
    memory.sum().backward()
    assert current.grad[0].abs().sum() > 0
    assert current.grad[1].abs().sum() == 0
    assert history.grad[0].abs().sum() > 0
    assert history.grad[1].abs().sum() == 0


def test_the_depth_history_switch_is_read_from_its_own_key() -> None:
    """history_images_mask is (B, cameras) now; depth has a (B,) switch of its own."""
    encoder = _encoder()
    current = torch.full((1, 80, 80), 300.0)
    history = torch.full((1, 3, 80, 80), 250.0)
    on = encoder.memory_from_batch(
        _batch(current, history, history_depth_mask=torch.tensor([True])), batch_size=1, device=CPU
    )
    off = encoder.memory_from_batch(
        _batch(current, history, history_depth_mask=torch.tensor([False])), batch_size=1, device=CPU
    )
    missing = encoder.memory_from_batch(_batch(current), batch_size=1, device=CPU)
    assert not torch.allclose(on, off)
    # A masked window must compute what a missing one computes (same tolerance the
    # existing pointmap test uses: the masked-attention path is not bit-identical).
    assert torch.allclose(off, missing, atol=1e-5)


def test_a_two_dimensional_camera_mask_still_drives_the_depth_path() -> None:
    """Back-compat: without its own key, the (B, cameras) RGB mask is reduced with all."""
    encoder = _encoder()
    current = torch.full((1, 80, 80), 300.0)
    history = torch.full((1, 3, 80, 80), 250.0)
    reduced = encoder.memory_from_batch(
        _batch(current, history, history_images_mask=torch.tensor([[True, False, True]])),
        batch_size=1,
        device=CPU,
    )
    off = encoder.memory_from_batch(
        _batch(current, history, history_depth_mask=torch.tensor([False])), batch_size=1, device=CPU
    )
    assert torch.allclose(reduced, off, atol=1e-5)


def test_rgb_only_samples_keep_a_usable_forward() -> None:
    """A batch with no depth at all still produces tokens, as it always did."""
    encoder = _encoder()
    memory, valid = encoder.memory_from_batch({}, batch_size=3, device=CPU, return_valid_mask=True)
    assert memory.shape == (3, encoder.num_tokens, 16)
    assert not valid.any()
