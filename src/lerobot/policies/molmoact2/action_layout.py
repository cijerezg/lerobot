"""Low-dimensional vector conventions for mixed robot embodiments.

State and action vectors are ingested in **source order** and are never permuted.
Whatever order a corpus recorded its joints in is the order the model sees, so the
gripper sits wherever that robot puts it -- slot 5 on a six-DoF SO-101, slot 6 on an
ARX5, slot 7 on a Franka.  The one thing the pipeline requires is that a sample's
state and action describe the same quantities in the same order, because the anchor
encoding subtracts one from the other elementwise; feature *widths* may still differ,
so the shared prefix is what has to line up.

Every packed source is joint-space: measured or commanded joint positions followed by
a gripper channel.  There is deliberately no control-mode axis.  FMB was the one
Cartesian holdout, and rather than carry a coordinate-system switch through the whole
pipeline for a single corpus it now reads its measured Franka joints (see
``FMBCorpusEpisode.state``), which is the same 8-slot layout DROID records.  A future
genuinely Cartesian source would need that switch reintroduced -- and, more
importantly, its own answer for anchor encoding, which is only meaningful when state
and action live in the same space.

What is left here is bookkeeping for **width**: a batch is one tensor, so rows of
different DoF are right-padded to a common width and the padding is carried alongside
as a prefix-valid ``*_dim_is_pad`` mask.  Downstream losses and normalization read
that mask.
"""

from __future__ import annotations

import torch
from torch import Tensor


def trim_to_native(tensor: Tensor, *, native_dim: int) -> Tensor:
    """Drop the batching padding, leaving the robot's own ``native_dim`` values.

    Source order is preserved end to end, so recovering a deployable command is a
    slice: the policy's slot ``i`` is the robot's slot ``i``.
    """
    tensor = torch.as_tensor(tensor)
    if tensor.shape[-1] < native_dim:
        raise ValueError(f"Cannot trim to D={native_dim} from tensor shape {tuple(tensor.shape)}.")
    return tensor[..., :native_dim]


def valid_dim_mask(batch_size: int, native_dim: int, width: int, *, device=None) -> Tensor:
    if native_dim < 1 or native_dim > width:
        raise ValueError(f"native_dim must be in [1, {width}], got {native_dim}.")
    mask = torch.ones((batch_size, width), dtype=torch.bool, device=device)
    mask[:, :native_dim] = False
    return mask


def require_prefix_valid_mask(dim_is_pad: Tensor, key: str) -> Tensor:
    """Check every row pads only a suffix, and return the mask unchanged.

    Native widths differ per row, but padding is always appended, so a valid row is
    ``[False] * native_dim + [True] * pad``.  Consumers reduce these masks with a
    count (``(~mask).sum(-1)``) and would silently mislabel real dimensions as
    padding if a hole ever appeared, so the shape is enforced where it is produced.
    """
    mask = torch.as_tensor(dim_is_pad, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    widths = (~mask).sum(dim=-1)
    expected = torch.arange(mask.shape[-1], device=mask.device)[None] >= widths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError(f"{key} must pad only a suffix; got a row with interior padding.")
    return mask
