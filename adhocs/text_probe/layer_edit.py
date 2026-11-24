"""Reads and edits inside the language model for the layer study (second round).

``high_norm``: which positions have an L2 norm over ``FACTOR`` x the layer median. ``edited``: change the output
of language-model blocks during a forward (clamp the norm, undo a block's update, replace by given vectors).
``attention``: the attention weights of every language-model block from chosen query positions. ``updates``: the
norm of what the attention and the MLP of every block add to each position.

An edit only applies when a block sees the whole prompt, so the cached decoding steps of a chat answer run on
the edited prompt but are not edited themselves.
"""
from __future__ import annotations

import torch

FACTOR = 10.0


def high_norm(encoder: torch.Tensor, groups: list[str], factor: float = FACTOR) -> torch.Tensor:
    """[layers, positions] bool: norm of the block output over ``factor`` x the median norm of the attended
    positions of that layer."""
    norm = encoder.float().norm(dim=-1)
    attended = torch.tensor([g != "masked" for g in groups])
    out = norm > factor * norm[:, attended].median(dim=1, keepdim=True).values
    out[:, ~attended] = False
    return out


def blocks(policy):
    return policy._backbone().transformer.blocks


class edited:
    """``fn(layer, before, after) -> new output or None`` on every block that sees ``positions`` positions.
    ``before`` is the block's input, ``after`` its output, both [..., positions, width]."""

    def __init__(self, policy, positions: int, fn):
        self.policy, self.positions, self.fn = policy, positions, fn

    def __enter__(self):
        self.handles, before = [], {}

        def pre(index):
            def hook(_module, args, kwargs):
                before[index] = args[0] if args else kwargs["hidden_states"]

            return hook

        def post(index):
            def hook(_module, _args, _kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[-2] != self.positions:
                    return None
                new = self.fn(index, before[index], hidden)
                if new is None:
                    return None
                return (new, *output[1:]) if isinstance(output, tuple) else new

            return hook

        for index, block in enumerate(blocks(self.policy)):
            self.handles.append(block.register_forward_pre_hook(pre(index), with_kwargs=True))
            self.handles.append(block.register_forward_hook(post(index), with_kwargs=True))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


def _over(hidden: torch.Tensor, attended: torch.Tensor, factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(positions over factor x the median norm [..., S], the median [..., 1])"""
    norm = hidden.float().norm(dim=-1)
    median = norm[..., attended].median(dim=-1, keepdim=True).values
    over = norm > factor * median
    over[..., ~attended] = False
    return over, median


def clamp(attended: torch.Tensor, factor: float = FACTOR, only: torch.Tensor | None = None):
    """Every position over ``factor`` x the layer median is scaled down to the median norm (direction kept).
    ``only`` [S] bool restricts it to some positions."""

    def fn(_layer, _before, after):
        over, median = _over(after, attended.to(after.device), factor)
        if only is not None:
            over = over & only.to(after.device)
        norm = after.float().norm(dim=-1)
        scale = torch.where(over, median / norm, torch.ones_like(norm))
        return after * scale.unsqueeze(-1).to(after.dtype)

    return fn


def block_writes(attended: torch.Tensor, factor: float = FACTOR, only: torch.Tensor | None = None):
    """Where a block's output is over ``factor`` x the layer median, the block's update is undone at that
    position (output = input). ``only`` [S] bool restricts it to some positions."""

    def fn(_layer, before, after):
        over, _ = _over(after, attended.to(after.device), factor)
        if only is not None:
            over = over & only.to(after.device)
        return torch.where(over.unsqueeze(-1), before, after)

    return fn


def replace(mask: torch.Tensor, values: torch.Tensor):
    """Positions ``mask`` [layers, S] of each block's output are set to ``values`` [layers, S, width]."""

    def fn(layer, _before, after):
        if not bool(mask[layer].any()):
            return None
        at = mask[layer].to(after.device)
        new = after.clone()
        new[..., at, :] = values[layer][mask[layer]].to(after.device, after.dtype)
        return new

    return fn


class updates:
    """Norm of what the attention and the MLP of each block add, per position: ``attn`` and ``mlp`` [layers, S]
    (first call of each block that sees ``positions`` positions)."""

    def __init__(self, policy, positions: int):
        self.policy, self.positions = policy, positions
        self.kept = {"attn": {}, "mlp": {}}

    def __enter__(self):
        def keep(name, index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[-2] == self.positions:
                    self.kept[name].setdefault(index, hidden.detach().float().reshape(-1, *hidden.shape[-2:])[0].norm(dim=-1).cpu())

            return hook

        self.handles = []
        for index, block in enumerate(blocks(self.policy)):
            self.handles.append(block.self_attn.register_forward_hook(keep("attn", index)))
            self.handles.append(block.mlp.register_forward_hook(keep("mlp", index)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.attn = torch.stack([self.kept["attn"][i] for i in sorted(self.kept["attn"])])
        self.mlp = torch.stack([self.kept["mlp"][i] for i in sorted(self.kept["mlp"])])


class attention:
    """Attention weights of every language-model block, mean over heads, from ``queries`` to every position:
    ``mass`` [layers, queries, S] (first call of each block that sees ``positions`` positions)."""

    def __init__(self, policy, positions: int, queries: list[int]):
        self.policy, self.positions, self.queries = policy, positions, queries
        self.layer, self.kept = None, {}

    def __enter__(self):
        functional = torch.nn.functional
        self.original = functional.scaled_dot_product_attention

        def attend(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            if self.layer is not None and self.layer not in self.kept and query.shape[-2] == self.positions and key.shape[-2] == self.positions:
                rows = torch.tensor(self.queries, device=query.device)
                q = query.reshape(-1, *query.shape[-3:])[0][:, rows].float()  # [H, Q, d]
                k = key.reshape(-1, *key.shape[-3:])[0].float()
                scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
                if attn_mask is not None:
                    mask = attn_mask.reshape(-1, *attn_mask.shape[-2:])[0][rows]
                    scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask.float()
                elif is_causal:
                    future = torch.arange(self.positions, device=query.device)[None, :] > rows[:, None]
                    scores = scores.masked_fill(future, -torch.inf)
                self.kept[self.layer] = scores.softmax(dim=-1).mean(dim=0).half().cpu()
            return self.original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

        functional.scaled_dot_product_attention = attend
        self.handles = []
        for index, block in enumerate(blocks(self.policy)):
            self.handles.append(block.self_attn.register_forward_pre_hook(lambda _m, _a, index=index: setattr(self, "layer", index)))
            self.handles.append(block.self_attn.register_forward_hook(lambda _m, _a, _o: setattr(self, "layer", None)))
        return self

    def __exit__(self, *_):
        torch.nn.functional.scaled_dot_product_attention = self.original
        for handle in self.handles:
            handle.remove()
        self.mass = torch.stack([self.kept[i] for i in sorted(self.kept)]) if self.kept else None


def mean_token(attended: torch.Tensor, source: torch.Tensor, factor: float = FACTOR):
    """At the first block's output, every position over ``factor`` x the layer median is set to the mean of the
    ``source`` positions [S] bool that are not over it (the high norm that comes in with the image tokens)."""

    def fn(layer, _before, after):
        if layer != 0:
            return None
        over, _ = _over(after, attended.to(after.device), factor)
        rest = source.to(after.device) & ~over.reshape(-1, over.shape[-1])[0]
        mean = after[..., rest, :].float().mean(dim=-2, keepdim=True).to(after.dtype)
        return torch.where(over.unsqueeze(-1), mean, after)

    return fn
