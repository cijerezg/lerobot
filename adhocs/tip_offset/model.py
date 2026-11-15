"""The tip-offset adapter: wrist tokens (image feature + depth points), top tokens, subtask tokens and one "no target"
token through a transformer; the output is one distribution over the 27 x 27 wrist patches plus "no target"."""
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 27  # wrist patches per side; the 14 x 14 tokens each cover 2 x 2 of them (the last row and column are padding)


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
        """wrist, top [B, 196, 1152]; depth [B, 27, 27, 4] (scaled x, y, z, valid share); text [B, L, text_width];
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
