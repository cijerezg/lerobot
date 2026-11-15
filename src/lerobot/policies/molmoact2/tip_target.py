"""The tip target adapter (docs/notes/tip_target_adapter.md): where in the wrist image is the point of the next
gripper event. Wrist tokens (image feature + depth points), top tokens, subtask tokens and one "no target" token go
through a small transformer; the output is one distribution over the 27 x 27 wrist patches plus "no target". The
weighted mean of the patches' depth points is the vector from the gripper tip to the target."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

GRID = 27  # wrist patches per side; the 14 x 14 tokens each cover 2 x 2 of them (the last row and column are padding)
TIP_TEXT_LENGTH = 16  # subtask tokens the adapter reads; longer texts are cut
TIP_WIDTH, TIP_LAYERS, TIP_HEADS = 384, 6, 6  # the adapter inside the policy, 13M parameters
TIP_LR = 2e-4
TIP_GRAD_CLIP_NORM = 1.0
TIP_WRIST_CAMERA = "observation.images.wrist_0"
TIP_TOP_CAMERA = "observation.images.external_0"
TIP_DEPTH_COLUMN = "depth.wrist_0.depth"  # the wrist depth frames in the replay buffer (role name)
TIP_DEPTH_OBSERVATION = "observation.depth.wrist_0"  # the wrist depth frame in the actor's observation
LABEL_SIGMA = 0.75  # width of the label around the target pixel, in wrist patches
POINT_SCALE = 0.1  # m; the depth points enter the adapter in this unit (about their median distance)
FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966  # D405 at 640 x 480
# The wrist camera is fixed to the gripper: tool = CAMERA_TO_TOOL_R @ camera + CAMERA_TO_TOOL_T (m).
CAMERA_TO_TOOL_R = ((-0.9976, 0.0042, -0.0698), (0.0177, -0.9506, -0.3099), (-0.0677, -0.3104, 0.9482))
CAMERA_TO_TOOL_T = (0.0081, 0.0504, -0.054)


def to_cells(x):
    """[B, 27, 27, C] per patch -> [B, 196, 4 * C] per token, padded at the bottom and right like the image pooling."""
    b, c = x.shape[0], x.shape[-1]
    x = F.pad(x, (0, 0, 0, 1, 0, 1))
    return x.view(b, 14, 2, 14, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(b, 196, 4 * c)


def to_patches(x):
    """[B, 196, 4] per token -> [B, 729] per patch, the inverse of to_cells."""
    b = x.shape[0]
    return x.view(b, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(b, 28, 28)[:, :GRID, :GRID].reshape(b, GRID * GRID)


class Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        hidden = width * 8 // 3
        self.norm1, self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width, bias=False), nn.Linear(width, width, bias=False)
        self.up, self.down = nn.Linear(width, 2 * hidden, bias=False), nn.Linear(hidden, width, bias=False)

    def forward(self, x, cos, sin, mask):
        b, s, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(b, s, w)), self.dropout, self.training)
        gate, value = self.up(self.norm2(x)).chunk(2, -1)
        return x + F.dropout(self.down(F.silu(gate) * value), self.dropout, self.training)  # SwiGLU


def rotate(x, cos, sin):
    """Rotary positions: rotate the pairs of x [B, heads, S, D] by the angles of each token [S, D / 2]."""
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


class Adapter(nn.Module):
    def __init__(self, text_width, text_length, image_width=1152, width=768, layers=12, heads=12, dropout=0.1):
        super().__init__()
        self.image_norm, self.image_in = nn.RMSNorm(image_width), nn.Linear(image_width, width)
        self.depth_in = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.text_norm, self.text_in = nn.RMSNorm(text_width), nn.Linear(text_width, width)
        self.text_position = nn.Parameter(torch.zeros(text_length, width))
        self.top, self.none = nn.Parameter(torch.zeros(width)), nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList([Block(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.RMSNorm(width)
        self.patch_head, self.none_head = nn.Linear(width, 4), nn.Linear(width, 1)
        for p in [self.text_position, self.top, self.none]:
            nn.init.normal_(p, std=0.02)
        # rotary angles: row and column of the 14 x 14 grid for image tokens (both cameras), zero for the other tokens
        n = width // heads // 4
        freq = 100.0 ** (-torch.arange(n) / n)
        r, c = torch.meshgrid(torch.arange(14.0), torch.arange(14.0), indexing="ij")
        grid = torch.cat([r.reshape(-1, 1) * freq, c.reshape(-1, 1) * freq], 1)  # [196, D / 2]
        self.register_buffer("angles", torch.cat([torch.zeros(1 + text_length, 2 * n), grid, grid]), persistent=False)

    def forward(self, wrist, top, depth, text, text_mask):
        """wrist, top [B, 196, image_width]; depth [B, 27, 27, 4] (scaled x, y, z, valid share); text [B, L, text_width];
        text_mask [B, L] True on real tokens. Returns the scores [B, 730]: the 729 wrist patches, then "no target"."""
        b = wrist.shape[0]
        x = torch.cat([
            self.none.expand(b, 1, -1),
            self.text_in(self.text_norm(text)) + self.text_position,
            self.image_in(self.image_norm(wrist)) + self.depth_in(to_cells(depth)),
            self.image_in(self.image_norm(top)) + self.top,
        ], 1)
        mask = torch.cat([text_mask.new_ones(b, 1), text_mask, text_mask.new_ones(b, 392)], 1)[:, None, None, :]
        cos, sin = self.angles.cos(), self.angles.sin()
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        x = self.norm(x)
        n = 1 + text.shape[1]
        return torch.cat([to_patches(self.patch_head(x[:, n : n + 196])), self.none_head(x[:, 0])], 1)


def patch_points(depth):
    """Raw wrist depth frames [B, 480, 640] (1e-4 m, 0 = no reading) -> [B, 27, 27, 4]: the mean point of each wrist
    patch in the tool frame (m) over its pixels with a reading, and the share of its pixels that have one."""
    b, device = depth.shape[0], depth.device
    v, u = torch.meshgrid(torch.arange(480, device=device), torch.arange(640, device=device), indexing="ij")
    rays = torch.stack([(u - CX) / FX, (v - CY) / FY, torch.ones_like(u, dtype=torch.float32)], -1).reshape(-1, 3)
    cell = ((v * GRID // 480) * GRID + u * GRID // 640).reshape(-1)  # the wrist patch of each depth pixel
    metres = depth.reshape(b, -1).float() * 1e-4
    valid = (metres > 0).float()
    rotation = torch.tensor(CAMERA_TO_TOOL_R, device=device)
    points = (rays * metres[..., None]) @ rotation.T + torch.tensor(CAMERA_TO_TOOL_T, device=device)
    total = torch.zeros(b, GRID * GRID, 3, device=device).index_add_(1, cell, points * valid[..., None])
    count = torch.zeros(b, GRID * GRID, device=device).index_add_(1, cell, valid)
    pixels = torch.zeros(GRID * GRID, device=device).index_add_(0, cell, torch.ones(len(cell), device=device))
    return torch.cat([total / count.clip(1)[..., None], (count / pixels)[..., None]], -1).view(b, GRID, GRID, 4)


def scaled_points(points):
    """patch_points as the adapter reads them: x, y, z in POINT_SCALE, zero where the patch has no reading."""
    return torch.cat([points[..., :3] / POINT_SCALE * (points[..., 3:] > 0), points[..., 3:]], -1)


def tip_vector(scores, points):
    """scores [B, 730] (float32); points [B, 27, 27, 4] from patch_points. Returns the probability of "no target" [B]
    and the vector from the gripper tip to the target [B, 3] (m, tool frame): the mean of the patches' depth points
    weighted by their probability, over the patches with a reading."""
    b = scores.shape[0]
    p = scores.softmax(-1)
    weights = p[:, :-1] * (points[..., 3].reshape(b, -1) > 0)
    weights = weights / weights.sum(1, keepdim=True).clip(1e-9)
    return p[:, -1], (weights[..., None] * points[..., :3].reshape(b, -1, 3)).sum(1)


def tip_target_loss(scores, target, points, reduction):
    """scores [B, 730] (float32); target [B, 6] rows of utils/tip_target.py; points [B, 27, 27, 4] from patch_points.

    The label is a bump around the target pixel over the wrist patches, or everything on "no target". A NaN target
    row (a source without targets) has loss 0 and is left out of the numbers. Returns the loss and the metrics:
    tip_loss and tip_none_accuracy are means over the tip_frames rows with a target row; tip_peak_on_patch,
    tip_peak_within_one and tip_vector_error_cm (adapter vector against the target offset) are means over the
    tip_target_frames rows whose target is in the wrist view."""
    b = scores.shape[0]
    known = ~torch.isnan(target).any(-1)
    in_view = known & (target[:, 0] > 0)
    gu, gv = target[:, 4] * GRID / 640, target[:, 5] * GRID / 480
    centre = torch.arange(GRID, device=scores.device) + 0.5
    bump = torch.exp(
        -((centre[None, :, None] - gv[:, None, None]) ** 2 + (centre[None, None, :] - gu[:, None, None]) ** 2)
        / (2 * LABEL_SIGMA**2)
    ).reshape(b, -1)
    label = torch.zeros_like(scores)
    label[in_view, :-1] = (bump / bump.sum(1, keepdim=True))[in_view]
    label[~in_view, -1] = 1
    per_frame = -(label * scores.log_softmax(-1)).sum(-1) * known

    with torch.no_grad():
        p = scores.softmax(-1)
        peak = p[:, :-1].argmax(-1)
        off = torch.maximum((peak // GRID - gv.floor()).abs(), (peak % GRID - gu.floor()).abs())
        vector = tip_vector(scores, points)[1]
        error_cm = 100 * (vector - target[:, 1:4]).norm(dim=-1)

        def mean(values, rows):
            return (values[rows].float().sum() / rows.sum().clip(1)).item()

        metrics = {
            "tip_loss": mean(per_frame, known),
            "tip_none_accuracy": mean((p[:, -1] > 0.5) == ~in_view, known),
            "tip_peak_on_patch": mean(off == 0, in_view),
            "tip_peak_within_one": mean(off <= 1, in_view),
            "tip_vector_error_cm": mean(error_cm, in_view),
            "tip_frames": int(known.sum()),
            "tip_target_frames": int(in_view.sum()),
        }
    return per_frame if reduction == "none" else per_frame.sum() / known.sum().clip(1), metrics
