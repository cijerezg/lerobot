"""Low-dimensional vector conventions for mixed robot embodiments.

State and action vectors are ingested in **source order** and are never permuted.
Whatever order a corpus recorded its joints in is the order the model sees, so the
gripper sits wherever that robot puts it -- slot 5 on a six-DoF SO-101, slot 6 on an
ARX5, slot 7 on a Franka.  The one thing the pipeline requires is that a sample's
state and action describe the same quantities in the same order, because the anchor
encoding subtracts one from the other elementwise; feature *widths* may still differ,
so the shared prefix is what has to line up.

Every packed source but one is joint-space: measured or commanded joint positions
followed by a gripper channel.  FMB was the first Cartesian candidate and reads its
measured Franka joints instead (see ``FMBCorpusEpisode.state``), the same 8-slot layout
DROID records.  MolmoAct (layout 7, v2) is the genuinely Cartesian source: xyz metres,
an Euler triple, gripper ratio, and no joint channel in either release.  The switch
that tells the two apart is ``ActionLayout.control_mode`` ("joint" | "end_effector",
``datasets/diverse_actor_selection.py``): it lives on the layout record only, never on
a config field or a batch column of its own, and reaches the model as a prompt clause
(``The control mode is joint space.`` / ``... end-effector space.``) rendered right
after the embodiment clause on every row from the batch's ``action_layout_id``
(``processor_molmoact2._build_robot_text``).  Nothing numeric branches on it.

Anchor encoding has no layout branch either: ``AnchorEncodeStep`` subtracts the state
from the action elementwise for every row.  For layout 7 that is well-defined because
MolmoAct's action IS its state at the same timestep (``copy_state``), so both live in
the same end-effector frame and the encoded target is a pose displacement -- xyz in
metres, three Euler deltas in radians (the triple is unwrapped per episode at ingest,
so no delta crosses a +-pi seam), gripper ratio -- normalized under layout 7's own
stats row.  (A ``copy_state`` chunk starts one tick after the anchor,
``diverse_pilot.COPY_STATE_LEAD_S``, so its k=0 delta is a real displacement rather than
the identically-zero ``s[t0] - s[t0]``.)  What it does not do: no conversion between spaces, no check that a row's
state and action share one; that invariant is a property of the ingest, not of this
step.

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
