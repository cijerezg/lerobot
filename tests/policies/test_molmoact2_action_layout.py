import pytest
import torch

from lerobot.policies.molmoact2.action_layout import (
    require_prefix_valid_mask,
    trim_to_native,
    valid_dim_mask,
)


def test_trim_to_native_is_a_slice_so_source_order_survives_the_round_trip():
    native = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]])
    padded = torch.cat([native, torch.zeros(1, 3)], dim=-1)

    assert torch.equal(trim_to_native(padded, native_dim=7), native)
    # Every robot keeps its own gripper slot: 6 for this 7-DoF arm, 5 for a 6-DoF one.
    assert trim_to_native(padded, native_dim=6).tolist() == [[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]


def test_trim_to_native_rejects_a_width_the_tensor_cannot_supply():
    with pytest.raises(ValueError, match="Cannot trim"):
        trim_to_native(torch.zeros(1, 5), native_dim=8)


def test_valid_dim_mask_marks_only_the_padded_suffix():
    mask = valid_dim_mask(2, 6, 8)
    assert mask.tolist() == [[False] * 6 + [True, True]] * 2


def test_prefix_valid_mask_accepts_mixed_widths_and_rejects_interior_holes():
    mixed = torch.tensor(
        [
            [False, False, False, False, False, False, True, True],
            [False, False, False, False, False, False, False, False],
        ]
    )
    assert torch.equal(require_prefix_valid_mask(mixed, "action_dim_is_pad"), mixed)

    hole = torch.tensor([[False, False, True, False, False, False, True, True]])
    with pytest.raises(ValueError, match="pad only a suffix"):
        require_prefix_valid_mask(hole, "action_dim_is_pad")
