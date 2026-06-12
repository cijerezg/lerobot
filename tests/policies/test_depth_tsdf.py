# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TSDF geometry asserts (depth_tsdf_design.md §3.3) + buffer next_depth alignment.

Only what a real run can't catch: wrong projection math trains silently on garbage
geometry, and a misaligned next_depth silently feeds the critic target the wrong
frame. Bit-identity at gate=0 is verified on the real checkpoint by
lerobot/probes/tsdf_bit_identity.py (GPU + checkpoint), not here.
"""

import torch

from lerobot.policies.depth_tsdf.modeling_depth_tsdf import build_tsdf_grid

# Camera at the gripper origin looking along +z (identity extrinsic), unit focal
# scale: pixel (u, v) sees ray direction ((u-cx)/fx, (v-cy)/fy, 1).
INTRINSICS = (100.0, 100.0, 32.0, 32.0)
IDENTITY = torch.eye(4)


def _grid(depth_mm, *, box_min=(-8.0, -8.0, 50.0), n=16, voxel=1.0, tau=4.0):
    return build_tsdf_grid(
        depth_mm.unsqueeze(0),
        intrinsics=INTRINSICS,
        t_g_from_c=IDENTITY,
        box_min_mm=box_min,
        voxel_size_mm=voxel,
        grid_size=n,
        truncation_mm=tau,
        depth_units_mm=1.0,
    )


def test_plane_zero_crossing_free_occupied_unknown():
    # Fronto-parallel plane at z=58mm fills the image; the box spans z in [50, 66].
    depth = torch.full((64, 64), 58.0)
    grid = _grid(depth)
    phi, known = grid[0, 0], grid[0, 1]

    # Center column of the grid (x=y≈0 -> principal ray). Voxel k has center z = 50.5 + k.
    phi_col = phi[8, 8]
    known_col = known[8, 8]
    z = 50.5 + torch.arange(16.0)
    sdf = 58.0 - z

    # In front of the plane: known, positive, saturating to +1 beyond tau.
    expected = (sdf / 4.0).clamp(-1.0, 1.0)
    visible = sdf >= -4.0
    assert torch.equal(known_col, visible.float())
    assert torch.allclose(phi_col[visible], expected[visible], atol=1e-5)
    # The zero-crossing sits between the voxels straddling z=58.
    assert phi_col[7] > 0 > phi_col[8]
    # Occluded more than tau behind the surface: unknown, phi stored as 0.
    assert torch.equal(phi_col[~visible], torch.zeros((~visible).sum()))


def test_holes_are_unknown_not_surface():
    depth = torch.full((64, 64), 58.0)
    depth[:32, :] = 0.0  # holes over half the image (v < 32 -> y < 0 rays)
    grid = _grid(depth)
    known = grid[0, 1]
    # y-negative half of the box projects into the hole region: all unknown.
    assert known[:, :7, :].sum() == 0
    # The intact half is still observed.
    assert known[:, 9:, :].sum() > 0


def test_outside_frustum_is_unknown():
    # A box centered 200mm off-axis in x projects at u = 100*(200/z)+32 >> 64 for z in [50, 66].
    depth = torch.full((64, 64), 58.0)
    grid = _grid(depth, box_min=(192.0, -8.0, 50.0))
    assert grid[0, 1].sum() == 0  # entirely outside the camera frustum
    assert grid[0, 0].abs().sum() == 0  # and phi is 0 where unknown, not a sentinel


def test_extrinsic_translation_shifts_surface():
    # Camera 20mm behind the gripper origin along z: T_{G<-C} z-translation = -20
    # puts the gripper-frame plane z_G = 58 - 20 = 38 at camera depth 58.
    t = torch.eye(4)
    t[2, 3] = -20.0
    depth = torch.full((64, 64), 58.0)
    grid = build_tsdf_grid(
        depth.unsqueeze(0),
        intrinsics=INTRINSICS,
        t_g_from_c=t,
        box_min_mm=(-8.0, -8.0, 30.0),
        voxel_size_mm=1.0,
        grid_size=16,
        truncation_mm=4.0,
        depth_units_mm=1.0,
    )
    phi_col = grid[0, 0, 8, 8]
    # Voxel k center: z_G = 30.5 + k; surface at z_G = 38 -> crossing between k=7 and k=8.
    assert phi_col[7] > 0 > phi_col[8]


def test_uint16_units_and_batching():
    # Raw uint16 at 0.1mm/level: 580 -> 58mm. Second batch element sees a nearer plane.
    raw = torch.full((2, 64, 64), 580, dtype=torch.uint16)
    raw[1] = 540
    grid = build_tsdf_grid(
        raw,
        intrinsics=INTRINSICS,
        t_g_from_c=IDENTITY,
        box_min_mm=(-8.0, -8.0, 50.0),
        voxel_size_mm=1.0,
        grid_size=16,
        truncation_mm=4.0,
        depth_units_mm=0.1,
    )
    assert grid[0, 0, 8, 8, 7] > 0 > grid[0, 0, 8, 8, 8]  # surface at 58mm
    assert grid[1, 0, 8, 8, 3] > 0 > grid[1, 0, 8, 8, 4]  # surface at 54mm


def test_buffer_sample_emits_next_depth_aligned_with_next_state():
    from lerobot.rl.buffer import ReplayBuffer

    n, chunk = 12, 3
    buf = ReplayBuffer(
        capacity=n,
        device="cpu",
        state_keys=["observation.state"],
        storage_device="cpu",
        optimize_memory=True,
    )
    for i in range(n):
        buf.add(
            state={"observation.state": torch.full((1, 2), float(i))},
            action=torch.zeros(1, 2),
            reward=0.0,
            next_state={"observation.state": torch.full((1, 2), float(i + 1))},
            done=False,
            truncated=False,
            complementary_info={"depth.top.depth": torch.full((1, 8, 8), i, dtype=torch.uint16)},
        )
    batch = buf.sample(4, action_chunk_size=chunk)
    depth = batch["complementary_info"]["depth.top.depth"]
    next_depth = batch["complementary_info"]["next_depth.top.depth"]
    assert next_depth.shape == depth.shape and next_depth.dtype == torch.uint16
    # Depth values encode the frame index: next_depth must be the depth of the frame
    # next_state is derived from (idx + chunk, modulo capacity), per sample row.
    sampled_idx = depth[:, 0, 0].long()
    assert torch.equal(next_depth[:, 0, 0].long(), (sampled_idx + chunk) % n)
    # And that frame is exactly the one next_state reports.
    assert torch.equal(batch["next_state"]["observation.state"][:, 0].long(), (sampled_idx + chunk) % n)
