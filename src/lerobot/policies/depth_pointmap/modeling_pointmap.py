"""Back-projected point-map depth tokens (depth_pointmap_design.md).

Pipeline:

  depth image ──back_project──▶ 4-channel point map [X, Y, Z, m] (camera frame, mm)
            ──patchify──▶ 192 non-overlapping P×P patches
            ──per patch──▶ token = f (local-shape 2D CNN, recentered) + g (Fourier
                           PE of the absolute centroid)

A single depth frame is a heightfield (one Z per pixel), so the image plane is its
natural dense domain: the CNN is 2D in *structure* (dense, hole-friendly via the
mask channel, no quantization) but 3D in *content* (each pixel carries its metric
position). Recentering each patch to its centroid removes position but keeps metric
scale (a near patch has small Δ, a far one large), so f is translation-invariant
local shape; the absolute centroid goes to the position encoding g instead.

Emits (B, N, d_mem) tokens that feed the co-evolving DepthStream (modeling_stream.py);
the per-layer read gate + sink live on that stream, not here. Units: millimeters.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_pointmap import DepthPointmapConfig


def back_project(
    depth: Tensor,
    *,
    intrinsics: tuple[float, float, float, float],
    depth_units_mm: float,
    z_min_mm: float,
    z_max_mm: float,
) -> Tensor:
    """Depth image → metric point map in the camera frame (design §1).

    Args:
        depth: (H, W), (B, H, W) or (B, 1, H, W) raw depth; uint16 or float. 0 = hole.
        intrinsics: (fx, fy, cx, cy) of the raw depth stream, in pixels.
        depth_units_mm: raw value × this = mm.
        z_min_mm, z_max_mm: valid-depth band; outside → mask 0 (near deadzone, far cutoff).

    Returns:
        (B, 4, H, W) float32 = [X, Y, Z, m]. Z is the depth; X=(u-cx)Z/fx,
        Y=(v-cy)Z/fy; m is the validity mask. Invalid pixels have X=Y=Z=0.
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)  # single live frame (H, W) → (1, H, W)
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    if depth.ndim != 3:
        raise ValueError(f"back_project expects (B, H, W) depth, got {tuple(depth.shape)}.")
    z = depth.to(torch.float32) * depth_units_mm  # (B, H, W) mm
    _, height, width = z.shape
    fx, fy, cx, cy = intrinsics

    m = (z >= z_min_mm) & (z <= z_max_mm)
    mf = m.to(torch.float32)
    vv, uu = torch.meshgrid(
        torch.arange(height, device=z.device, dtype=torch.float32),
        torch.arange(width, device=z.device, dtype=torch.float32),
        indexing="ij",
    )
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return torch.stack([x * mf, y * mf, z * mf, mf], dim=1)


def patchify(pointmap: Tensor, patch_size: int) -> Tensor:
    """(B, C, H, W) → (B, N, C, P, P), non-overlapping, row-major patch order."""
    b, c, h, w = pointmap.shape
    p = patch_size
    nh, nw = h // p, w // p
    x = pointmap.reshape(b, c, nh, p, nw, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, nh * nw, c, p, p)


def fourier_position_encoding(
    centers_mm: Tensor, *, lambda_max_mm: float, lambda_min_mm: float, num_wavelengths: int
) -> Tensor:
    """Geometric sin/cos ladder over metric positions (design §4).

    centers_mm: (T, 3) → (T, 6 * num_wavelengths). A wavelength λ_k is the distance
    over which sinusoid k completes one cycle.
    """
    device = centers_mm.device
    ratio = lambda_min_mm / lambda_max_mm
    steps = torch.arange(num_wavelengths, device=device, dtype=torch.float32) / max(num_wavelengths - 1, 1)
    wavelengths = lambda_max_mm * ratio**steps
    angles = 2 * torch.pi * centers_mm.to(torch.float32).unsqueeze(-1) / wavelengths  # (T, 3, L)
    return torch.cat([angles.sin(), angles.cos()], dim=-1).reshape(centers_mm.shape[0], -1)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


class PatchResidualBlock2d(nn.Module):
    """Stride-2 residual downsampling block, GroupNorm/SiLU (design §3)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
        self.norm1 = _group_norm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = _group_norm(c_out)
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class PatchShapeCNN(nn.Module):
    """Per-patch 2D CNN over the recentered point map → one feature vector.

    Shared across all patches (a conv shares its filters by construction). Applied
    to (M, C_in, P, P) and global-average-pooled to (M, d_out). For P=40 the three
    stride-2 blocks downsample 40→20→10→5 before pooling.
    """

    def __init__(self, in_channels: int, hidden: tuple[int, ...], d_out: int) -> None:
        super().__init__()
        dims = (in_channels, *hidden, d_out)
        self.blocks = nn.Sequential(
            *(PatchResidualBlock2d(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x).mean(dim=(-1, -2))


class DepthPointmapEncoder(nn.Module):
    """Point map → depth-stream input tokens (design §3–5).

    Input : (B, 4, H, W) from build via back_project.
    Output: (B, N, d_mem) tokens, N = (H/P)(W/P), where d_mem is the depth-stream
    width. These tokens then co-evolve through the DepthStream (modeling_stream.py),
    which owns the per-layer read gate; this encoder is gate-free.

    null_tokens is the learned per-patch bank substituted for empty patches (all
    pixels invalid) and under whole-sample modality dropout / depth-missing.
    """

    NUM_HIDDEN = (32, 64)

    def __init__(self, config: DepthPointmapConfig, *, d_mem: int) -> None:
        super().__init__()
        self.config = config
        height, width = config.image_size
        self.num_tokens = (height // config.patch_size) * (width // config.patch_size)

        in_channels = 4 + (1 if config.include_centroid_depth else 0)
        self.cnn = PatchShapeCNN(in_channels, self.NUM_HIDDEN, d_mem)

        # PE bounds derived so they track the far cutoff (design §4): λ_min = near
        # token spacing P·z_min/fx, λ_max = 2·z_max (must span the scene or alias).
        fx = config.intrinsics[0]
        self.lambda_min_mm = config.patch_size * config.z_min_mm / fx
        self.lambda_max_mm = 2.0 * config.z_max_mm
        self.pos_proj = nn.Linear(6 * config.num_wavelengths, d_mem)

        self.modality_embed = nn.Parameter(torch.randn(d_mem) * 0.02)
        self.null_tokens = nn.Parameter(torch.randn(self.num_tokens, d_mem) * 0.02)
        # Temporal slots: [0 .. T_h-1] = past frames oldest → newest, [-1] = current.
        # Only created when history is on, so historyless checkpoints load unchanged.
        if config.history_num_samples > 0:
            self.time_embed = nn.Parameter(torch.randn(config.history_num_samples + 1, d_mem) * 0.02)

    def null_memory(self, batch_size: int) -> Tensor:
        """Null bank for the full token assembly (history slots included when on)."""
        null = self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        t_h = self.config.history_num_samples
        if t_h == 0:
            return null
        return torch.cat([null + self.time_embed[i] for i in [*range(t_h), -1]], dim=1)

    def forward(self, pointmap: Tensor) -> Tensor:
        cfg = self.config
        p = cfg.patch_size
        pointmap = pointmap.to(self.pos_proj.weight.dtype)
        patches = patchify(pointmap, p)  # (B, N, 4, P, P)
        b, n = patches.shape[:2]
        coords = patches[:, :, :3]  # (B, N, 3, P, P)
        mask = patches[:, :, 3:4]  # (B, N, 1, P, P)

        count = mask.sum(dim=(-1, -2))  # (B, N, 1)
        centroid = (coords * mask).sum(dim=(-1, -2)) / count.clamp(min=1.0)  # (B, N, 3)
        empty = count.squeeze(-1) == 0  # (B, N)

        delta = (coords - centroid[..., None, None]) * mask / cfg.coord_scale_mm
        cnn_in = torch.cat([delta, mask], dim=2)  # (B, N, 4, P, P)
        if cfg.include_centroid_depth:
            zbar = (centroid[:, :, 2:3] / cfg.coord_scale_mm)[..., None, None]  # (B, N, 1, 1, 1)
            cnn_in = torch.cat([cnn_in, zbar.expand(b, n, 1, p, p)], dim=2)

        f = self.cnn(cnn_in.reshape(b * n, cnn_in.shape[2], p, p)).reshape(b, n, -1)
        pe = fourier_position_encoding(
            centroid.reshape(b * n, 3),
            lambda_max_mm=self.lambda_max_mm,
            lambda_min_mm=self.lambda_min_mm,
            num_wavelengths=cfg.num_wavelengths,
        )
        g = self.pos_proj(pe.to(self.pos_proj.weight.dtype)).reshape(b, n, -1)

        token = f + g + self.modality_embed
        null = self.null_tokens.unsqueeze(0).to(token.dtype)
        return torch.where(empty[..., None], null, token)

    def memory_from_batch(
        self, batch: dict[str, Tensor], *, batch_size: int, device: torch.device
    ) -> Tensor:
        """Depth tokens from a policy batch: back-project → encode (design §1, §5).

        Consumes raw metric depth from observation.depth.{depth_key} (no [0,1]
        normalizer on this path). Swaps in the learned null bank under modality
        dropout at train time and whenever depth is missing, keeping shapes static.

        Returns memory (B, N, d_mem) — the DepthStream's initial tokens.
        """
        cfg = self.config
        memory = self._encode_depth(batch.get(f"observation.depth.{cfg.depth_key}"), batch_size, device)
        if cfg.history_num_samples > 0:
            memory = torch.cat(
                [self._history_memory(batch, batch_size, device), memory + self.time_embed[-1]], dim=1
            )
        if self.training and cfg.dropout_prob > 0:
            dropped = torch.rand(memory.shape[0], device=memory.device) < cfg.dropout_prob
            memory = torch.where(
                dropped[:, None, None], self.null_memory(memory.shape[0]).to(memory.dtype), memory
            )
        return memory

    def _encode_depth(self, depth: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        """One frame slot (B, N, d_mem): back-project → encode; null bank when missing."""
        cfg = self.config
        if depth is None:
            return self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        depth = torch.as_tensor(depth).to(device=device)
        pointmap = back_project(
            depth,
            intrinsics=tuple(cfg.intrinsics),
            depth_units_mm=cfg.depth_units_mm,
            z_min_mm=cfg.z_min_mm,
            z_max_mm=cfg.z_max_mm,
        )
        return self(pointmap)

    def _history_memory(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        """Past-frame slots (B, T_h·N, d_mem), oldest → newest, each with its slot's
        time embedding. Reads the buffer-canonical history window
        history.depth.{depth_key}.depth (B, T_h, H, W); padded slots (episode start,
        see the _is_pad mask) and a missing window take the null bank instead."""
        cfg = self.config
        t_h = cfg.history_num_samples
        null = self.null_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        window = batch.get(f"history.depth.{cfg.depth_key}.depth")
        if window is None:
            return torch.cat([null + self.time_embed[i] for i in range(t_h)], dim=1)
        window = torch.as_tensor(window).to(device=device)
        if window.shape[1] != t_h:
            raise ValueError(
                f"depth history window has {window.shape[1]} frames, expected "
                f"history_num_samples={t_h}."
            )
        flat = window.reshape(batch_size * t_h, *window.shape[2:])
        tokens = self._encode_depth(flat, batch_size * t_h, device)
        tokens = tokens.reshape(batch_size, t_h, self.num_tokens, -1)
        pad = batch.get(f"history.depth.{cfg.depth_key}.depth_is_pad")
        if pad is not None:
            pad = torch.as_tensor(pad).to(device=tokens.device, dtype=torch.bool)
            tokens = torch.where(pad[:, :, None, None], null.unsqueeze(1).to(tokens.dtype), tokens)
        tokens = tokens + self.time_embed[:t_h, None, :]
        return tokens.reshape(batch_size, t_h * self.num_tokens, -1)
