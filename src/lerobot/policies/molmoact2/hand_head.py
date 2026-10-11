"""Separate hand parameters: the expert's hand head and the hand framing tokens' input rows.

Two fresh pieces, both behind ``HandBlockConfig`` flags, both spliced in with forward hooks so the
checkpoint's own modules and parameter names are untouched and the flag off removes everything.

**Expert head** (``hand.expert_head``). The hand block occupies slots 8..14 of the 32-wide action
chunk. Without this head it rides columns 8..14 of the checkpoint's ``action_embed`` (32 -> d) and
rows 8..14 of ``final_layer.linear`` (d -> 32), columns that were zero-input and masked in
pretraining. :class:`HandExpertHead` gives the hand block its own ``hand_in`` (7 -> d) and
``hand_out`` (d -> 7):

- a pre-hook on ``action_embed`` zeroes slots 8..14 of its input, and a forward hook adds
  ``hand_in(x[..., 8:15])`` to its output. ``hand_in`` starts as a copy of columns 8..14 of
  ``action_embed`` (never updated in pretraining, so at the checkpoint's init scale), which makes
  the input side identical to the single-projection path at step 0; from then on the original
  columns see zero input and get no gradient.
- a forward hook on ``final_layer.linear`` overwrites slots 8..14 of its output with
  ``hand_out(h)``. Zero-initialized like the checkpoint's own final layer: the hand velocity is
  exactly zero at step 0, the weight still gets gradient ``dL/dy . h^T``.

The expert's output stays one 32-wide tensor, so the loss, the masks, the RTC loop and the HF
sampling loop see what they see today.

**Framing token rows** (``hand.learn_tokens``). ``<hand_start>`` / ``<hand_end>`` are added to the
tokenizer with input rows copied from ``<action_start>`` / ``<action_end>``, and the embedding table
is frozen (``freeze_embedding``), so at the input side the model cannot tell the hand span from the
joint span. :class:`HandTokenRows` holds the two rows as one free parameter initialized to those
copies, used as ``r * w / ||w||`` with ``r`` the joint row's norm (pinned: no learning rate can
blow the magnitude up, only the direction moves), and swapped into the embedding output by a
forward hook on ``wte`` wherever the input id is a hand marker. The gradient is the loss gradient
projected onto the tangent plane, scaled by ``r / ||w||``; :meth:`HandTokenRows.renormalize` after
each optimizer step keeps ``||w|| = r`` so that scale stays stationary (the forward is invariant to
it). :meth:`HandTokenRows.cosines` is the telemetry: the cosine of each row to its joint seed, 1.0
meaning the model has not told the spans apart.

Spec: ``docs/ee_mixture_loss/TODO.md`` (Phase 3b).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lerobot.policies.molmoact2.hand_block import HAND_ACTION_SLOTS, HAND_ACTION_WIDTH, HAND_DELTA_DIM

HAND_HEAD_ATTR = "hand_head"
HAND_TOKEN_ROWS_ATTR = "hand_token_rows"


class HandExpertHead(nn.Module):
    """``hand_in`` / ``hand_out`` and the hooks that splice them around the expert's projections."""

    def __init__(self, action_embed: nn.Linear, final_linear: nn.Linear) -> None:
        super().__init__()
        if not isinstance(action_embed, nn.Linear) or int(action_embed.in_features) < HAND_ACTION_WIDTH:
            raise RuntimeError(
                "HandExpertHead needs action_embed to be a Linear over >= "
                f"{HAND_ACTION_WIDTH} action slots, got {type(action_embed).__name__}."
            )
        if not isinstance(final_linear, nn.Linear) or int(final_linear.out_features) < HAND_ACTION_WIDTH:
            raise RuntimeError(
                "HandExpertHead needs final_layer.linear to be a Linear over >= "
                f"{HAND_ACTION_WIDTH} action slots, got {type(final_linear).__name__}."
            )
        width = int(action_embed.out_features)
        device, dtype = action_embed.weight.device, action_embed.weight.dtype
        self.hand_in = nn.Linear(HAND_DELTA_DIM, width, bias=False, device=device, dtype=dtype)
        self.hand_out = nn.Linear(int(final_linear.in_features), HAND_DELTA_DIM, device=device, dtype=dtype)
        with torch.no_grad():
            self.hand_in.weight.copy_(action_embed.weight[:, HAND_ACTION_SLOTS])
            self.hand_out.weight.zero_()
            self.hand_out.bias.zero_()
        self._stash: Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # ── hooks ────────────────────────────────────────────────────────────────

    def _embed_pre_hook(self, module: nn.Module, args: tuple) -> tuple:
        x = args[0]
        self._stash = x[..., HAND_ACTION_SLOTS]
        zeroed = torch.cat(
            [
                x[..., : HAND_ACTION_SLOTS.start],
                torch.zeros_like(self._stash),
                x[..., HAND_ACTION_SLOTS.stop :],
            ],
            dim=-1,
        )
        return (zeroed, *args[1:])

    def _embed_hook(self, module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        hand = self._stash
        self._stash = None
        if hand is None:
            raise RuntimeError("HandExpertHead: action_embed ran without its pre-hook.")
        return output + self.hand_in(hand.to(self.hand_in.weight.dtype)).to(output.dtype)

    def _final_hook(self, module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        hidden = args[0]
        hand = self.hand_out(hidden.to(self.hand_out.weight.dtype)).to(output.dtype)
        return torch.cat(
            [output[..., : HAND_ACTION_SLOTS.start], hand, output[..., HAND_ACTION_SLOTS.stop :]], dim=-1
        )

    def attach(self, action_embed: nn.Module, final_linear: nn.Module) -> None:
        self.detach_hooks()
        self._handles = [
            action_embed.register_forward_pre_hook(self._embed_pre_hook),
            action_embed.register_forward_hook(self._embed_hook),
            final_linear.register_forward_hook(self._final_hook),
        ]

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def install_hand_expert_head(action_expert: nn.Module) -> HandExpertHead:
    """Register a :class:`HandExpertHead` on ``action_expert`` (as ``action_expert.hand_head``, so
    its parameters sit under the expert's name for the freeze and optimizer rules) and attach its
    hooks. Idempotent."""
    existing = getattr(action_expert, HAND_HEAD_ATTR, None)
    if isinstance(existing, HandExpertHead):
        return existing
    action_embed = getattr(action_expert, "action_embed", None)
    final_layer = getattr(action_expert, "final_layer", None)
    final_linear = getattr(final_layer, "linear", None)
    if action_embed is None or final_linear is None:
        raise RuntimeError(
            "install_hand_expert_head: the action expert exposes no action_embed / final_layer.linear."
        )
    head = HandExpertHead(action_embed, final_linear)
    setattr(action_expert, HAND_HEAD_ATTR, head)
    head.attach(action_embed, final_linear)
    return head


class HandTokenRows(nn.Module):
    """The hand framing tokens' input rows: one free ``(2, d)`` parameter, norm-pinned at use."""

    def __init__(self, token_ids: tuple[int, int], seed_rows: Tensor) -> None:
        super().__init__()
        seed_rows = seed_rows.detach().to(torch.float32)
        if tuple(seed_rows.shape[:1]) != (2,) or seed_rows.ndim != 2:
            raise ValueError(f"HandTokenRows expects two seed rows (2, d), got {tuple(seed_rows.shape)}.")
        self.token_ids = (int(token_ids[0]), int(token_ids[1]))
        self.rows = nn.Parameter(seed_rows.clone())
        self.register_buffer("norm", seed_rows.norm(dim=-1).clone())
        self.register_buffer("seed", seed_rows.clone())
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def effective(self) -> Tensor:
        """``r * w / ||w||`` per row, float32."""
        return self.rows * (self.norm / self.rows.norm(dim=-1).clamp_min(1e-12))[:, None]

    @torch.no_grad()
    def renormalize(self) -> None:
        """Put ``w`` back on its sphere; a no-op for the function, it only resets the gradient scale."""
        self.rows.mul_((self.norm / self.rows.norm(dim=-1).clamp_min(1e-12))[:, None])

    @torch.no_grad()
    def cosines(self) -> dict[str, float]:
        cos = nn.functional.cosine_similarity(self.rows.float(), self.seed.float(), dim=-1)
        return {"hand_token_cos_start": float(cos[0]), "hand_token_cos_end": float(cos[1])}

    def _wte_hook(self, module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        ids = args[0]
        if not torch.is_tensor(ids) or ids.shape != output.shape[:-1]:
            return output
        rows = self.effective().to(device=output.device, dtype=output.dtype)
        for which, token_id in enumerate(self.token_ids):
            output = torch.where((ids == token_id)[..., None], rows[which], output)
        return output

    def attach(self, wte: nn.Module) -> None:
        self.detach_hook()
        self._handle = wte.register_forward_hook(self._wte_hook)

    def detach_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def install_hand_token_rows(wte: nn.Module, token_ids: tuple[int, int]) -> HandTokenRows:
    """Build :class:`HandTokenRows` seeded from what ``wte`` currently returns for ``token_ids``
    (the copies of the joint framing rows) and attach its hook. The caller registers the module."""
    device = next(iter(wte.parameters())).device
    with torch.no_grad():
        seed = wte(torch.tensor(list(token_ids), device=device, dtype=torch.long))
    rows = HandTokenRows(token_ids, seed.reshape(2, -1))
    rows.to(device)
    rows.attach(wte)
    return rows
