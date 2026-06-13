"""Gripper-frame truncated-signed-distance-function (TSDF) depth tokens (depth_tsdf_design.md).

Geometry core (design §3): single-frame projective TSDF — pure batched tensor
ops, no learned parameters, no pose (the grid lives in the gripper frame and the
camera is rigid to it, so only the constant T_{G←C} extrinsic enters).

Encoder (design §5): shallow 3D ResNet trunk + self-attention over the final
cells, Fourier positional encoding of metric cell centers, LayerNorm +
projection to the consuming policy's memory width ``d_mem``, a learned modality
embedding, a scalar tanh gate initialized to 0, and a learned null bank for
modality dropout / depth-missing.

Everything here is model-agnostic: the encoder emits (B, T, d_mem) memory
tokens plus a scalar gate; where those enter a policy's attention (the seam) is
that policy's business. Units are millimeters end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_depth_tsdf import DepthTsdfConfig


def build_tsdf_grid(
    depth: Tensor,
    *,
    intrinsics: tuple[float, float, float, float],
    t_g_from_c: Tensor,
    box_min_mm: tuple[float, float, float],
    voxel_size_mm: float,
    grid_size: int,
    truncation_mm: float,
    depth_units_mm: float,
) -> Tensor:
    """Single-frame projective TSDF in the gripper frame (design §3.3).

    Args:
        depth: (B, H, W) or (B, 1, H, W) raw depth; 0 = hole. uint16 or float.
        intrinsics: (fx, fy, cx, cy) of the raw depth stream.
        t_g_from_c: (4, 4) camera→gripper extrinsic, translation in mm.
        box_min_mm: gripper-frame box origin; the box spans box_min + N·δ per axis.
        voxel_size_mm: voxel pitch δ.
        grid_size: side length N in voxels.
        truncation_mm: TSDF truncation τ.
        depth_units_mm: raw value × this = mm.

    Returns:
        (B, 2, N, N, N) float32. Channel 0: φ = clip(sdf/τ, -1, 1), zeroed where
        unknown. Channel 1: known mask. Axis order (x, y, z), ij-indexed; voxel
        (i, j, k) has center box_min + (i+0.5, j+0.5, k+0.5)·δ.

    The three known=0 cases (design §3.3): voxel outside the camera frustum,
    hole pixel, voxel occluded more than τ behind an observed surface. The
    network must read validity from the known channel, never from φ's 0.
    """
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    if depth.ndim != 3:
        raise ValueError(f"TSDF builder expects (B, H, W) depth, got {tuple(depth.shape)}.")
    device = depth.device
    depth_mm = depth.to(torch.float32) * depth_units_mm
    batch_size, height, width = depth_mm.shape
    fx, fy, cx, cy = intrinsics
    n = grid_size

    # Voxel centers in the gripper frame, then into the camera frame.
    idx = torch.arange(n, device=device, dtype=torch.float32)
    centers = torch.stack(torch.meshgrid(idx, idx, idx, indexing="ij"), dim=-1).reshape(-1, 3)
    box_min = torch.tensor(box_min_mm, device=device, dtype=torch.float32)
    p_g = box_min + (centers + 0.5) * voxel_size_mm
    t_c_from_g = torch.linalg.inv(t_g_from_c.to(device=device, dtype=torch.float32))
    q = p_g @ t_c_from_g[:3, :3].T + t_c_from_g[:3, 3]
    z = q[:, 2]

    # Project to the nearest pixel (one image lookup per voxel; bilinear would
    # interpolate across depth discontinuities and invent surfaces).
    z_safe = z.clamp(min=1e-6)
    u = (fx * q[:, 0] / z_safe + cx).round().long()
    v = (fy * q[:, 1] / z_safe + cy).round().long()
    in_frustum = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    flat = v.clamp(0, height - 1) * width + u.clamp(0, width - 1)
    measured = depth_mm.reshape(batch_size, -1)[:, flat]

    sdf = measured - z
    known = in_frustum & (measured > 0) & (sdf >= -truncation_mm)
    known = known.to(torch.float32)
    phi = (sdf / truncation_mm).clamp(-1.0, 1.0) * known
    return torch.stack([phi, known], dim=1).reshape(batch_size, 2, n, n, n)


def fourier_position_encoding(
    centers_mm: Tensor, *, lambda_max_mm: float, lambda_min_mm: float, num_wavelengths: int
) -> Tensor:
    """Geometric sin/cos ladder over metric positions (design §5.3).

    centers_mm: (T, 3) → returns (T, 6 * num_wavelengths).
    """
    ratio = lambda_min_mm / lambda_max_mm
    steps = torch.arange(num_wavelengths, dtype=torch.float32) / max(num_wavelengths - 1, 1)
    wavelengths = lambda_max_mm * ratio**steps
    angles = 2 * torch.pi * centers_mm.unsqueeze(-1) / wavelengths  # (T, 3, L)
    return torch.cat([angles.sin(), angles.cos()], dim=-1).reshape(centers_mm.shape[0], -1)


class TsdfResidualBlock3d(nn.Module):
    """Stride-2 residual downsampling block, GroupNorm/SiLU (design §5.2)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.skip = nn.Conv3d(c_in, c_out, kernel_size=1, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class TsdfSelfAttentionBlock(nn.Module):
    """Pre-LN self-attention + MLP over the trunk's output cells."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x))


class DepthTsdfEncoder(nn.Module):
    """TSDF grid → policy memory tokens (design §5).

    Input : (B, 2, N, N, N) from build_tsdf_grid.
    Output: (B, (N/8)³, d_mem) ungated memory tokens; the scalar tanh(gate) is
    applied by the consuming policy at its attention read site (gated additive
    read), not to the tokens, so that gate=0 is bit-identical to depth-free.

    null_tokens is the learned null bank substituted under modality dropout or
    when depth is missing (design §6.3).
    """

    TRUNK_CHANNELS = (32, 64, 128, 256)
    NUM_FOURIER_WAVELENGTHS = 6

    def __init__(
        self,
        config: DepthTsdfConfig,
        *,
        d_mem: int,
        num_read_layers: int | None = None,
        num_read_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        c0, c1, c2, c3 = self.TRUNK_CHANNELS
        self.num_tokens = (config.grid_size // 8) ** 3
        self.stem = nn.Sequential(nn.Conv3d(2, c0, kernel_size=3, padding=1), nn.GroupNorm(8, c0), nn.SiLU())
        self.stages = nn.Sequential(
            TsdfResidualBlock3d(c0, c1), TsdfResidualBlock3d(c1, c2), TsdfResidualBlock3d(c2, c3)
        )
        self.attn_blocks = nn.Sequential(TsdfSelfAttentionBlock(c3), TsdfSelfAttentionBlock(c3))
        self.out_norm = nn.LayerNorm(c3)
        self.out_proj = nn.Linear(c3, d_mem)
        self.modality_embed = nn.Parameter(torch.randn(d_mem) * 0.02)
        self.null_tokens = nn.Parameter(torch.randn(self.num_tokens, d_mem) * 0.02)
        # Scalar Flamingo-style gate; tanh(0) = 0 at init (design §5.4).
        self.gate = nn.Parameter(torch.zeros(()))

        self.register_buffer(
            "t_g_from_c", torch.tensor(config.t_g_from_c, dtype=torch.float32).reshape(4, 4), persistent=False
        )
        # Fourier encoding of the metric centers of the output cells; each cell
        # covers 8³ voxels. λ_max ≈ 2× box side, λ_min ≈ cell spacing.
        cell_size = 8 * config.voxel_size_mm
        side = config.grid_size * config.voxel_size_mm
        idx = torch.arange(config.grid_size // 8, dtype=torch.float32)
        cells = torch.stack(torch.meshgrid(idx, idx, idx, indexing="ij"), dim=-1).reshape(-1, 3)
        cell_centers = torch.tensor(config.box_min_mm) + (cells + 0.5) * cell_size
        pos = fourier_position_encoding(
            cell_centers,
            lambda_max_mm=2 * side,
            lambda_min_mm=cell_size,
            num_wavelengths=self.NUM_FOURIER_WAVELENGTHS,
        )
        self.register_buffer("pos_encoding", pos, persistent=False)
        self.pos_proj = nn.Linear(pos.shape[-1], c3)

    def gate_value(self) -> Tensor:
        return torch.tanh(self.gate)

    def null_memory(self, batch_size: int) -> Tensor:
        return self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, grid: Tensor) -> Tensor:
        x = self.stem(grid.to(self.out_proj.weight.dtype))
        x = self.stages(x)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C) in the cell centers' ij flatten order
        x = x + self.pos_proj(self.pos_encoding)
        x = self.attn_blocks(x)
        return self.out_proj(self.out_norm(x)) + self.modality_embed

    def memory_from_batch(
        self, batch: dict[str, Tensor], *, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Depth memory from a policy batch: grid → encode, once per call (design §6.2).

        Consumes raw metric depth from observation.depth.{depth_key} (a [0,1]
        depth normalizer must not run on this path). Swaps in the learned null
        bank under modality dropout at train time and whenever depth is missing
        (design §6.3), keeping shapes static.

        Returns (memory (B, T, d_mem) in encoder dtype, scalar tanh-gate).
        """
        cfg = self.config
        depth = batch.get(f"observation.depth.{cfg.depth_key}")
        if depth is None:
            return self.null_memory(batch_size), self.gate_value()
        depth = torch.as_tensor(depth).to(device=device)
        grid = build_tsdf_grid(
            depth,
            intrinsics=tuple(cfg.intrinsics),
            t_g_from_c=self.t_g_from_c,
            box_min_mm=tuple(cfg.box_min_mm),
            voxel_size_mm=cfg.voxel_size_mm,
            grid_size=cfg.grid_size,
            truncation_mm=cfg.truncation_mm,
            depth_units_mm=cfg.depth_units_mm,
        )
        memory = self(grid)
        if self.training and cfg.dropout_prob > 0:
            dropped = torch.rand(memory.shape[0], device=memory.device) < cfg.dropout_prob
            memory = torch.where(
                dropped[:, None, None], self.null_memory(memory.shape[0]).to(memory.dtype), memory
            )
        return memory, self.gate_value()
